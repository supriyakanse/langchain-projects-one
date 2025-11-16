# app/rag_chain.py

import os
from typing import Dict, Any

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

from app.vectorstore import load_faiss_index


# Store per-session conversational memories
CONVERSATION_MEMORY: Dict[str, ConversationBufferMemory] = {}


def get_memory(session_id: str):
    """Returns or creates memory for session."""
    if session_id not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",       # 👈 IMPORTANT FIX
            return_messages=True
        )
    return CONVERSATION_MEMORY[session_id]


def answer_query(session_id: str, question: str, k: int = 5) -> Dict[str, Any]:
    vectorstore, metadata = load_faiss_index()

    if not vectorstore:
        return {"success": False, "error": "FAISS index not loaded"}

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434")
    )

    memory = get_memory(session_id)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"      # 👈 SECOND FIX
    )

    result = chain({"question": question})

    # --- format sources ---
    sources = []
    for doc in result["source_documents"]:
        sources.append({
            "snippet": doc.page_content[:200] + "...",
            "subject": doc.metadata.get("subject"),
            "sender": doc.metadata.get("sender"),
            "date": doc.metadata.get("date"),
        })

    return {
        "success": True,
        "answer": result["answer"],
        "sources": sources
    }
