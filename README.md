# Email-RAG

Email-RAG is a small FastAPI-based service that fetches emails via IMAP, builds a FAISS vector index using Sentence-Transformers, and answers user questions over those emails using a local LLM (via Ollama). It provides endpoints to fetch messages, build/load a FAISS index, and run conversational semantic-search + generation with session-scoped memory and cited sources.

## Table of contents
- Project overview
- Prerequisites
- Installation
- Quick start
- API reference
- Environment variables
- Index files
- Security & limitations
- Troubleshooting
- Contributing

## Project overview

The service is implemented in the `app/` package. Key responsibilities:
- Fetch emails over IMAP and convert them to structured records (`app/email_utils.py`).
- Encode email text with `sentence-transformers` and store vectors in a FAISS index (`app/vectorstore.py`).
- Perform semantic search and generate answers using a local LLM via Ollama (`app/rag_chain.py`).
- Expose HTTP endpoints with FastAPI in `app/main.py`.

## Prerequisites

- Python 3.10 (Dockerfile uses `python:3.10-slim`).
- A working Ollama installation if you want local LLM inference (default host: `http://localhost:11434`).
- IMAP access to your mailbox (the code currently assumes `imap.gmail.com` for Gmail; verify or update `app/email_utils.py` for other providers).
- Recommended: create a Python virtual environment for local development.

Dependencies are listed in `requirements.txt`. For reproducible installs pin versions in `requirements.txt` as needed.

## Installation

1. Create and activate a virtual environment.
2. Install runtime dependencies from `requirements.txt`.

Refer to `requirements.txt` for the packages required by the project.

## Quick start

- Run locally (development): start the FastAPI app from the repository root by running `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`.
- Run in Docker (development): start with `docker-compose up --build` which uses the provided `Dockerfile` and `docker-compose.yml`.
- Build the image manually: build with `docker build -t email-rag .` and run with `docker run -p 8000:8000 email-rag`.

## API reference

All endpoints are exposed by the FastAPI app in `app/main.py`.

- `POST /fetch-emails`
  - Request JSON fields: `email` (string), `password` (string), `start_date` (string, YYYY-MM-DD), `end_date` (string, YYYY-MM-DD).
  - Behavior: logs into IMAP, fetches messages between dates, and returns a JSON payload with `count` and an array of email objects (`subject`, `sender`, `date`, `body`).

- `POST /build-index`
  - No request body.
  - Behavior: uses the server's in-memory email cache populated by `/fetch-emails` to build and save a FAISS index and associated metadata files. Returns success status and whether the index was loaded.

- `POST /chat`
  - Request JSON fields: `session_id` (string), `question` (string), `top_k` (integer, optional, default 5).
  - Behavior: loads the FAISS index from disk, performs a semantic search for the question, constructs a prompt combining retrieved email snippets and session memory, calls the configured LLM (via Ollama), stores conversation turns in in-memory session memory, and returns `answer` plus compact `sources` metadata.

Usage examples (inline JSON bodies):
- Fetch emails: call `POST /fetch-emails` with a JSON body like `{"email":"user@example.com","password":"APP_PASSWORD","start_date":"2025-01-01","end_date":"2025-02-01"}`.
- Chat: call `POST /chat` with a JSON body like `{"session_id":"session-1","question":"What did Alice say about the roadmap?","top_k":5}`.

## Environment variables

- `OLLAMA_URL` — Base URL for the Ollama API. Default in code: `http://localhost:11434`.
- `OLLAMA_MODEL` — Model identifier for Ollama. Default in code: `llama3.1:8b`.

These environment variables are read by `app/rag_chain.py` when calling the LLM.

## Index files

The FAISS index and metadata are persisted to the repository root as:
- `email_index.faiss` — the FAISS binary index file.
- `email_metadata.npy` — Numpy file storing email metadata aligned with vectors.

These files are ignored by `.gitignore` by default. For production usage consider storing them in a durable object store or a mounted volume.

## Security & limitations

- IMAP credentials are sent to the server in the `POST /fetch-emails` request; the current implementation expects plaintext credentials. Use app-specific passwords or OAuth in production, and never commit credentials.
- Conversation memory is stored in-memory per session (no persistence). Restarting the server clears session memory.
- No authentication is implemented for the API endpoints. Add proper auth, rate limiting, and monitoring before exposing to untrusted networks.
- Ollama must be running and reachable for LLM calls to succeed. If Ollama is not available the `/chat` call will return an LLM error string.

## Troubleshooting

- If `/build-index` reports no emails fetched, confirm you called `/fetch-emails` and that the IMAP credentials and date range are correct.
- If embedding or FAISS errors occur, ensure `sentence-transformers` and `faiss-cpu` are installed and compatible with your environment.
- If LLM generation fails, verify `OLLAMA_URL`, `OLLAMA_MODEL`, and that the Ollama server is running.

## Contributing

Contributions are welcome. Typical next improvements:
- Add API authentication and secure credential handling (e.g., OAuth).
- Persist session memory to a database.
- Add unit and integration tests.
- Pin dependency versions in `requirements.txt` for reproducible installs.

## License

Add a license file if you plan to publish or distribute this project.
