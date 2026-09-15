"""
FastAPI backend for the Dermatology chatbot.

Endpoints:
  POST /chat          — send a message, get answer + sources
  GET  /chat/{id}     — retrieve full conversation history
  DELETE /chat/{id}   — clear a conversation
"""

import uuid
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import llm
import retriever
import verifier
import config

app = FastAPI(title="Dermatology Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (replace with Redis/DB for production)
_sessions: dict[str, list[dict]] = {}


# ── Request / Response models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: Annotated[str, Field(min_length=1, max_length=2000)]
    conversation_id: str | None = None   # omit to start a new conversation


class Source(BaseModel):
    chapter:    str
    section:    str
    subsection: str
    page_start: int | None
    page_end:   int | None
    excerpt:    str
    score:      float


class Verification(BaseModel):
    is_abstention:       bool
    n_claims:            int
    n_grounded:          int
    groundedness_score:  float | None   # None = nothing to check (abstention, or no chunks)
    error:               str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer:  str
    sources: list[Source]
    verification: Verification


# ── Routes ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conv_id = req.conversation_id or str(uuid.uuid4())
    history = _sessions.setdefault(conv_id, [])

    # Retrieve relevant chunks
    chunks = retriever.retrieve(req.question, top_k=config.TOP_K)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    # Generate answer
    answer_text = llm.answer(req.question, chunks, history)

    # Verifier agent (Phase 5): re-check the answer's citations against the
    # same chunks the generator saw. Never let a verifier failure break the
    # chat turn — verify() already degrades to an "error" result internally.
    verification = verifier.verify(answer_text, chunks)

    # Update history
    history.append({"role": "user",      "content": req.question})
    history.append({"role": "assistant", "content": answer_text})

    sources = [
        Source(
            chapter=c["chapter"],
            section=c["section"],
            subsection=c["subsection"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            excerpt=c["text"][:400],   # first 400 chars as preview
            score=c["score"],
        )
        for c in chunks
    ]

    return ChatResponse(
        conversation_id=conv_id,
        answer=answer_text,
        sources=sources,
        verification=Verification(
            is_abstention=verification["is_abstention"],
            n_claims=verification["n_claims"],
            n_grounded=verification["n_grounded"],
            groundedness_score=verification["groundedness_score"],
            error=verification.get("error"),
        ),
    )


@app.get("/chat/{conversation_id}")
def get_history(conversation_id: str):
    history = _sessions.get(conversation_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation_id": conversation_id, "history": history}


@app.delete("/chat/{conversation_id}")
def clear_conversation(conversation_id: str):
    _sessions.pop(conversation_id, None)
    return {"status": "cleared"}


@app.get("/health")
def health():
    return {"status": "ok", "model": config.CLAUDE_MODEL}
