"""
Bridge API endpoints - Clean /v1/chat implementation
Dedicated file for API routes to keep architecture clear
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core_client import CoreClient

logger = logging.getLogger(__name__)

app = FastAPI(title="Jeffrey Bridge API", version="2.0")

# CORS configuration for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = "jeffrey-core"
    temperature: float | None = 0.7
    max_tokens: int | None = 1000


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


# Initialize CoreClient
core_client = CoreClient()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    core_healthy = await core_client.health_check()
    return {
        "status": "healthy" if core_healthy else "degraded",
        "core_connected": core_healthy,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint
    Routes requests to Jeffrey Core via Unix socket
    """
    try:
        # Prepare query for Core
        core_query = {
            "action": "chat",
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        # Send to Core and get response
        response = await core_client.ask_core(core_query)

        if not response:
            raise HTTPException(status_code=503, detail="Core service unavailable")

        # Format as OpenAI-compatible response
        chat_response = ChatResponse(
            id=f"chatcmpl-{datetime.utcnow().timestamp():.0f}",
            created=int(datetime.utcnow().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.get("content", "")},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
                "total_tokens": response.get("total_tokens", 0),
            },
        )

        return chat_response

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat")
async def simple_chat(messages: list[dict[str, str]]):
    """
    Simplified chat endpoint for testing
    Direct JSON input without Pydantic validation
    """
    try:
        query = {"action": "chat", "messages": messages}
        response = await core_client.ask_core(query)
        return {"response": response.get("content", "") if response else "Core unavailable"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
