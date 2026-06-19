from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = True,
    ):
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.model = SentenceTransformer(model_name)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)
        return [chunk for chunk in chunks if chunk.page_content.strip()]

    def embed_chunks(self, chunks: List[Document]) -> np.ndarray:
        if not chunks:
            dimension = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype=np.float32)

        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32, copy=False)
