import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_directory: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 32,
    ):
        self.persist_directory = persist_directory
        self.pipeline = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

    @property
    def _faiss_path(self) -> str:
        return os.path.join(self.persist_directory, "faiss.index")

    @property
    def _metadata_path(self) -> str:
        return os.path.join(self.persist_directory, "metadata.pkl")

    def exists(self) -> bool:
        return os.path.isfile(self._faiss_path) and os.path.isfile(self._metadata_path)

    def _ensure_index(self, dimension: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)

    def _document_to_metadata(self, chunk: Document) -> Dict[str, Any]:
        return {
            "content": chunk.page_content,
            "source": chunk.metadata.get("source"),
            **{k: v for k, v in chunk.metadata.items() if k != "source"},
        }

    def store_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if embeddings.size == 0:
            return

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._ensure_index(embeddings.shape[1])
        self.index.add(embeddings)
        if chunk_metadata:
            self.metadata.extend(chunk_metadata)

    def build_from_documents(self, documents: List[Document]) -> None:
        if not documents:
            print("No documents to index.")
            return

        chunks = self.pipeline.chunk_documents(documents)
        if not chunks:
            print("No text chunks produced from documents.")
            return

        embeddings = self.pipeline.embed_chunks(chunks)
        metadata = [self._document_to_metadata(chunk) for chunk in chunks]
        self.store_embeddings(embeddings, metadata)
        print(f"Indexed {len(chunks)} chunks from {len(documents)} documents.")

    def add_embeddings(
        self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]
    ) -> None:
        self.store_embeddings(embeddings, metadata)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No index to save. Build or load the vector store first.")

        os.makedirs(self.persist_directory, exist_ok=True)
        faiss.write_index(self.index, self._faiss_path)
        with open(self._metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"Vector store not found in {self.persist_directory}")

        self.index = faiss.read_index(self._faiss_path)
        with open(self._metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "index": int(idx),
                    "score": float(score),
                    "metadata": self.metadata[idx],
                }
            )
        return results

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.pipeline.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self.search(query_embedding, top_k)
