from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from app.email_utils import fetch_emails_between_dates
from app.vectorstore import build_faiss_index


from typing import Optional
from app.rag_chain import answer_query, CONVERSATION_MEMORY
from app.vectorstore import load_faiss_index

app = FastAPI()

# Temporary storage
EMAIL_CACHE = []

vectorstore = None
rag_chain = None



class FetchRequest(BaseModel):
    email: str
    password: str
    start_date: str
    end_date: str


@app.post("/fetch-emails")
def fetch_emails(req: FetchRequest):
    global EMAIL_CACHE
    EMAIL_CACHE = fetch_emails_between_dates(
        req.email,
        req.password,
        req.start_date,
        req.end_date
    )
    return {"count": len(EMAIL_CACHE), "emails": EMAIL_CACHE}


@app.post("/build-index")
def build_index():
    global EMAIL_CACHE, INDEX_LOADED

    if len(EMAIL_CACHE) == 0:
        return {"success": False, "message": "No emails fetched yet"}

    status = build_faiss_index(EMAIL_CACHE)

    index,metadata=load_faiss_index()
    INDEX_LOADED = index is not None

    return {"success": status, "index_loaded": INDEX_LOADED}


# Make sure index loaded (optional/diagnostic)
def is_index_loaded():
    index, _ = load_faiss_index()
    return index is not None

class ChatRequest(BaseModel):
    session_id: str  # client supplies a session id to keep conversation state
    question: str
    top_k: Optional[int] = 5

@app.post("/chat")
def chat(req: ChatRequest):
    # Always load fresh FAISS index (fixes old-data issue)
    index, metadata = load_faiss_index()

    if index is None:
        raise HTTPException(status_code=400, detail="FAISS index not loaded. Run /build-index first.")

    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    # Pass dynamically loaded index to your answer_query()
    resp = answer_query(req.session_id, req.question, k=req.top_k or 5)

    if not resp.get("success"):
        raise HTTPException(status_code=500, detail=resp.get("error", "unknown error"))

    return {"answer": resp["answer"], "sources": resp["sources"]}
