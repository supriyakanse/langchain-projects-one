# app/rag_chain.py

import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from app.vectorstore import load_faiss_index
from datetime import datetime

# --- LLM options ---
# We are switching fully to OLLAMA (recommended)
USE_LLAMA_CPP = False

# Retriever / embedding model (must match embedding model used when building index)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load FAISS index & metadata
index, metadata = load_faiss_index()
if index is None:
    print("Warning: FAISS index not found. build-index must be called first.")

# --- Simple in-memory conversational memory per session_id ---
CONVERSATION_MEMORY: Dict[str, List[Dict[str, str]]] = {}


# --- Semantic search helper ---
def semantic_search(query: str, k: int = 5):
    """
    Returns list of metadata items for top-k results and distances.
    """
    if index is None:
        return []

    q_vec = embedding_model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k)
    results = []

    indices = I[0].tolist()
    distances = D[0].tolist()

    for idx, dist in zip(indices, distances):
        try:
            meta = metadata[idx].item() if isinstance(metadata[idx], np.ndarray) else metadata[idx]
        except Exception:
            meta = metadata[idx]

        results.append({"meta": meta, "distance": float(dist)})

    return results


# --- Prompt construction ---
def build_prompt(retrieved: List[Dict[str, Any]], conversation_history: List[Dict[str, str]], user_query: str) -> str:
    system = (
        "You are an assistant that answers questions based ONLY on the provided email excerpts. "
        "If the answer is not present in the snippets, say: "
        "\"I don't have that information in the provided emails.\" "
        "Cite snippets as [source #n]."
    )

    parts = [f"SYSTEM: {system}", "\n---Retrieved email snippets---\n"]

    for i, r in enumerate(retrieved, start=1):
        meta = r.get("meta", {})
        subject = meta.get("subject", "<no-subject>")
        sender = meta.get("sender", "<unknown>")
        date = meta.get("date", "<unknown>")
        body = meta.get("body", "").strip()

        if len(body) > 1500:
            body = body[:1500] + " [...]"

        parts.append(
            f"[source #{i}] Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{body}\n"
        )

    # last 6 conversation turns
    parts.append("\n---Conversation history---\n")
    for turn in conversation_history[-6:]:
        role = turn.get("role")
        text = turn.get("text")
        parts.append(f"{role.capitalize()}: {text}")

    parts.append("\n---User question---\n")
    parts.append(f"User: {user_query}\n")
    parts.append("Answer concisely and cite as [source #n] when needed.\n")

    return "\n".join(parts)


# --- LLM call wrapper (OLLAMA) ---
def call_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Uses Ollama local model for inference.
    """
    import requests

    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "") or data.get("text", "")
    except Exception as e:
        return f"[LLM ERROR] {str(e)}"


# --- Main public API: answer a query ---
def answer_query(session_id: str, user_query: str, k: int = 5) -> Dict[str, Any]:
    """
    Steps:
        - Ensure memory for session
        - Retrieve top-k docs from FAISS
        - Build prompt
        - Call LLM
        - Save memory
        - Return structured answer
    """

    # init session memory
    mem = CONVERSATION_MEMORY.setdefault(session_id, [])

    # retrieve docs
    retrieved = semantic_search(user_query, k=k)

    # build prompt
    prompt = build_prompt(retrieved, mem, user_query)

    # run LLM
    answer_text = call_llm(prompt)

    # append memory
    mem.append({"role": "user", "text": user_query})
    mem.append({"role": "assistant", "text": answer_text})

    # compact sources for frontend
    sources = []
    for i, r in enumerate(retrieved, start=1):
        meta = r.get("meta", {})
        sources.append({
            "source_id": i,
            "subject": meta.get("subject"),
            "from": meta.get("sender"),
            "date": str(meta.get("date")),
            "distance": r.get("distance")
        })

    return {"success": True, "answer": answer_text, "sources": sources}
