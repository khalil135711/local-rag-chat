"""Core RAG logic: ingest, retrieve, route, generate. Fully local via Ollama + ChromaDB."""
import ast
import hashlib
import io
import operator as op
import os
from typing import Iterator

import chromadb
import ollama
from pypdf import PdfReader

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("DB_PATH", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

llm = ollama.Client(host=OLLAMA_HOST)
_db = chromadb.PersistentClient(path=DB_PATH)
collection = _db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


# ---------- Ingestion ----------
def load_pages(name: str, data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for a PDF, TXT or MD file."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    return [(1, data.decode("utf-8", errors="ignore"))]


def chunk_text(text: str) -> list[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    return llm.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def ingest_file(name: str, data: bytes) -> int:
    """Chunk, embed and store a file. Returns number of chunks stored."""
    ids, docs, metas = [], [], []
    for page, text in load_pages(name, data):
        for i, chunk in enumerate(chunk_text(text)):
            if len(chunk.strip()) < 20:
                continue
            ids.append(hashlib.md5(f"{name}-{page}-{i}-{chunk[:50]}".encode()).hexdigest())
            docs.append(chunk)
            metas.append({"source": name, "page": page})
    if not docs:
        return 0
    for i in range(0, len(docs), 32):  # batch embeddings
        batch = slice(i, i + 32)
        collection.upsert(
            ids=ids[batch],
            documents=docs[batch],
            metadatas=metas[batch],
            embeddings=embed(docs[batch]),
        )
    return len(docs)


def list_sources() -> list[str]:
    metas = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def delete_source(name: str) -> None:
    collection.delete(where={"source": name})


def reset() -> None:
    global collection
    _db.delete_collection("docs")
    collection = _db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


# ---------- Retrieval ----------
def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    if collection.count() == 0:
        return []
    res = collection.query(
        query_embeddings=embed([question]),
        n_results=min(k, collection.count()),
    )
    return [
        {"text": d, "source": m["source"], "page": m["page"], "score": 1 - dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# ---------- Tools (agent mode) ----------
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg}


def calculator(expression: str) -> str:
    """Safe arithmetic evaluator (no eval)."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Calculator error: {e}"


def route(question: str) -> str:
    """Ask the local LLM which tool to use: DOCS, CALC or CHAT. Defaults to DOCS."""
    prompt = (
        "Classify the user message. Reply with exactly one word.\n"
        "DOCS = needs information from the user's uploaded documents\n"
        "CALC = a pure arithmetic calculation\n"
        "CHAT = greeting or general chit-chat\n\n"
        f"Message: {question}\nAnswer:"
    )
    out = llm.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                   options={"temperature": 0})["message"]["content"].upper()
    for label in ("CALC", "CHAT", "DOCS"):
        if label in out:
            return label
    return "DOCS"


# ---------- Generation ----------
SYSTEM = (
    "You are a helpful assistant. Answer ONLY using the provided context. "
    "If the answer is not in the context, say you don't know. Be concise."
)


def answer(question: str, history: list[dict], agent: bool = False) -> tuple[Iterator[str], list[dict], str]:
    """Return (token_stream, sources, tool_used)."""
    tool = route(question) if agent else "DOCS"
    sources: list[dict] = []

    if tool == "CALC":
        expr = llm.chat(model=LLM_MODEL, options={"temperature": 0}, messages=[
            {"role": "user", "content": f"Convert to a plain arithmetic expression, output only the expression: {question}"}
        ])["message"]["content"].strip().strip("`")
        result = calculator(expr)
        return iter([f"`{expr}` = **{result}**"]), sources, "calculator"

    if tool == "CHAT":
        messages = [{"role": "system", "content": "You are a friendly assistant."}] + history[-6:]
        messages.append({"role": "user", "content": question})
    else:
        sources = retrieve(question)
        context = "\n\n".join(f"[{s['source']} p.{s['page']}]\n{s['text']}" for s in sources) or "(no documents indexed)"
        messages = [{"role": "system", "content": SYSTEM}] + history[-6:]
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    def stream() -> Iterator[str]:
        for part in llm.chat(model=LLM_MODEL, messages=messages, stream=True):
            yield part["message"]["content"]

    return stream(), sources, "chat" if tool == "CHAT" else "documents"
