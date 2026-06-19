import os
import socket
import sys

import gradio as gr

from src.search import RAGSearch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PERSIST_DIR = "faiss_store"
TOP_K = 8
HOST = "127.0.0.1"
PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

rag: RAGSearch | None = None


def find_available_port(host: str, start_port: int, max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(
        f"No available port found in range {start_port}-{start_port + max_attempts - 1}"
    )


def get_rag() -> RAGSearch:
    global rag
    if rag is None:
        print("Loading RAG index and embedding model...")
        rag = RAGSearch(
            persist_directory=PERSIST_DIR,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size=1000,
            chunk_overlap=200,
        )
        print("RAG index ready.")
    return rag


def _format_retrieval_fallback(results: list) -> str:
    chunks = []
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        source = os.path.basename(str(meta.get("source", "unknown")))
        preview = meta.get("content", "")[:600]
        chunks.append(f"**[{i}] {source}** (score={result['score']:.3f})\n{preview}")

    return (
        "OPENAI_API_KEY is not set, so here are the top retrieved chunks:\n\n"
        + "\n\n".join(chunks)
    )


def chat(message: str, history: list):
    message = message.strip()
    if not message:
        yield "Please enter a question."
        return

    engine = get_rag()

    if not os.getenv("OPENAI_API_KEY"):
        results = engine.search(message, top_k=TOP_K)
        if not results:
            yield "No relevant context found. Add documents to data/ and try again."
            return
        yield _format_retrieval_fallback(results)
        return

    yield from engine.stream_search_and_summarize(message, top_k=TOP_K)


def main() -> None:
    get_rag()

    demo = gr.ChatInterface(
        fn=chat,
        title="RAG Document Chatbot",
        description=(
            "Ask questions about the documents in your `data/` folder. "
            "Answers are generated in detail from retrieved context and stream in real time."
        ),
        examples=[
            "What is this document about? Explain in detail.",
            "Summarize the main topics and key takeaways.",
            "What are the important concepts and how do they relate?",
        ],
    )

    port = find_available_port(HOST, PORT)
    if port != PORT:
        print(f"Port {PORT} is in use, using {port} instead.")

    print(f"Starting chatbot at http://{HOST}:{port}")
    demo.launch(server_name=HOST, server_port=port, share=False)


if __name__ == "__main__":
    main()
