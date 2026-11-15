"""FastAPI Server exposing Hillock as an OpenAI-compatible endpoint."""

import json
import logging
import re
import threading
import time
import uuid
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import DB_FILE, HILLOCK_VERSION
from engine import IntegratedHillock

logger = logging.getLogger("Hillock.API")

# Initialize the FastAPI app, memory engine, and thread lock
app = FastAPI(title="Hillock API", version=HILLOCK_VERSION)
engine = IntegratedHillock(DB_FILE)
engine_lock = threading.Lock()


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str = Field(..., max_length=32000)


class ChatRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    messages: List[Message] = Field(..., min_length=1, max_length=200)
    stream: Optional[bool] = False
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)


def clean_response(text: str) -> str:
    """Strips the CLI prefix (e.g., 'Hillock > ') from the engine's response."""
    return re.sub(r"^Hillock.*? > ", "", text).strip()


@app.get("/")
def health_check():
    """Provides a friendly welcome message when users open the API in a browser."""
    return {
        "status": "online",
        "message": "Hillock API is running! Point your AI client to /v1/chat/completions",
        "version": HILLOCK_VERSION
    }


@app.get("/v1/models")
def list_models():
    """Dummy endpoint required by AnythingLLM to save workspaces."""
    return {
        "object": "list",
        "data": [
            {
                "id": "hillock",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hillock-engine"
            }
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    # Extract the last non-empty user message from the OpenAI messages array
    user_messages = [
        m.content.strip()
        for m in req.messages
        if m.role == "user" and m.content.strip()
    ]
    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="At least one non-empty user message is required."
        )
    query = user_messages[-1]

    # Guard mutable HDC state and SQLite writes across FastAPI worker threads
    with engine_lock:
        reply_raw, _, _, _ = engine.execute_chat_turn(
            query,
            stream_generator=bool(req.stream)
        )

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    if req.stream:
        def stream_generator():
            if isinstance(reply_raw, str):
                text = clean_response(reply_raw)
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            else:
                for token in reply_raw:
                    if not token:
                        continue
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    reply_text = clean_response(reply_raw)
    return JSONResponse(content={
        "id": response_id,
        "object": "chat.completion",
        "created": created_time,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Binding Hillock API to '%s' without authentication exposes local memory on the network.",
            host
        )
    print(f"Starting Hillock OpenAI-Compatible API Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()