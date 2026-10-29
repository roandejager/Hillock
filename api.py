"""FastAPI Server exposing Hillock as an OpenAI-compatible endpoint."""

import time
import uuid
import re
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from engine import IntegratedHillock
from config import DB_FILE

# Initialize the FastAPI app and the Hillock Memory Engine
app = FastAPI(title="Hillock API", version="1.0.2")
engine = IntegratedHillock(DB_FILE)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.0

def clean_response(text: str) -> str:
    """Strips the CLI prefix (e.g., 'Hillock > ') from the engine's response."""
    return re.sub(r"^Hillock.*? > ", "", text).strip()

@app.get("/")
def health_check():
    """Provides a friendly welcome message when users open the API in a browser."""
    return {
        "status": "online",
        "message": "Hillock API is running! Point your AI client to /v1/chat/completions",
        "version": "1.0.2"
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
    # Extract the last user message from the OpenAI messages array
    user_messages = [m.content for m in req.messages if m.role == "user"]
    query = user_messages[-1] if user_messages else ""

    # Execute the turn through Hillock's memory engine
    reply_raw, _, _, _ = engine.execute_chat_turn(query, stream_generator=req.stream)

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    if req.stream:
        def stream_generator():
            # If the engine returned a hardcoded string (e.g., STRICT mode refusal)
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
                # If the engine returned the true token generator
                for token in reply_raw:
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
    
    else:
        # If not streaming, reply_raw is guaranteed to be a string
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

if __name__ == "__main__":
    import uvicorn
    print("Starting Hillock OpenAI-Compatible API Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)