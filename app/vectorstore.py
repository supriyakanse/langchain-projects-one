# app/vectorstore.py
import os
import numpy as np
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

INDEX_DIR = "vectorstore"
META_PATH = os.path.join(INDEX_DIR, "email_metadata.npy")

os.makedirs(INDEX_DIR, exist_ok=True)

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def build_faiss_index(emails: List[Dict]):
    """
    Builds a LangChain FAISS vectorstore from emails.
    """

    if not emails:
        return False

    texts = []
    metadatas = []

    for email in emails:
        body_text = (
            f"Subject: {email['subject']}\n"
            f"From: {email['sender']}\n"
            f"Date: {email['date']}\n\n"
            f"{email['body']}"
        )

        texts.append(body_text)
        metadatas.append({
            "subject": email["subject"],
            "sender": email["sender"],
            "date": email["date"]
        })

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    vectorstore.save_local(INDEX_DIR)
    np.save(META_PATH, metadatas, allow_pickle=True)

    return True


def load_faiss_index():
    """
    Loads LangChain vectorstore + metadata.
    """

    if not os.path.exists(INDEX_DIR):
        return None, None

    if not os.path.exists(META_PATH):
        return None, None

    vectorstore = FAISS.load_local(
        INDEX_DIR, 
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    metadata = np.load(META_PATH, allow_pickle=True)

    return vectorstore, metadata
