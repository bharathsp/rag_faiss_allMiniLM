import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.data_loader import load_all_documents
from src.vector_store import FaissVectorStore

load_dotenv()

DEFAULT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the provided context. "
            "If the context does not contain enough information, say so briefly.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {query}",
        ),
    ]
)


class RAGSearch:
    def __init__(
        self,
        persist_directory: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0,
        min_score: Optional[float] = None,
    ):
        self.min_score = min_score
        self.vector_store = FaissVectorStore(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._llm_model = llm_model
        self._temperature = temperature
        self._llm: Optional[ChatOpenAI] = None
        self._chain = None

        if self.vector_store.exists():
            self.vector_store.load()
        else:
            documents = load_all_documents()
            if not documents:
                raise ValueError("No documents found in data/. Add files and try again.")
            self.vector_store.build_from_documents(documents)
            self.vector_store.save()

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(model=self._llm_model, temperature=self._temperature)
        return self._llm

    @property
    def chain(self):
        if self._chain is None:
            self._chain = DEFAULT_PROMPT | self.llm | StrOutputParser()
        return self._chain

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.vector_store.query(query, top_k=top_k)
        if self.min_score is None:
            return results
        return [result for result in results if result["score"] >= self.min_score]

    @staticmethod
    def _format_context(results: List[Dict[str, Any]], max_chars: int = 12000) -> str:
        seen: set[str] = set()
        blocks: List[str] = []
        total = 0

        for result in results:
            meta = result["metadata"]
            content = meta.get("content", "").strip()
            if not content or content in seen:
                continue
            seen.add(content)

            source = meta.get("source", "unknown")
            block = f"[source: {os.path.basename(str(source))} | score: {result['score']:.3f}]\n{content}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)

        return "\n\n".join(blocks)

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "No relevant context found for that question."

        context = self._format_context(results)
        return self.chain.invoke({"context": context, "query": query})
