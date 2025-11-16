import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Store index file
INDEX_PATH = "email_index.faiss"
META_PATH = "email_metadata.npy"

# Load model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def encode_text(text: str):
    """Generate embedding vector for text."""
    return embedding_model.encode([text])[0]


def build_faiss_index(emails: list):
    """
    Build FAISS vector store from email list.
    emails = list of {subject, sender, date, body}
    """
    if not emails:
        return False

    vectors = []
    metadata = []

    for email_data in emails:
        text = (
            f"Subject: {email_data['subject']}\n"
            f"From: {email_data['sender']}\n"
            f"Date: {email_data['date']}\n\n"
            f"{email_data['body']}"
        )

        vec = encode_text(text)
        vectors.append(vec)
        metadata.append(email_data)

    vectors = np.array(vectors).astype("float32")

    # Create FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save index & metadata
    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, metadata, allow_pickle=True)

    return True


def load_faiss_index():
    """Load FAISS index + metadata from disk"""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    metadata = np.load(META_PATH, allow_pickle=True)

    return index, metadata
