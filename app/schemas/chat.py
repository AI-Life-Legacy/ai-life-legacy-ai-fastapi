from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="The ID of the user (optional, passed from backend)")
    viewer_id: Optional[str] = Field(None, description="The ID of the viewer (optional, for tracking)")
    session_id: Optional[str] = Field(None, description="The session ID (optional, generated if missing)")
    role_id: Optional[str] = Field(None, description="The role ID of the avatar (optional, defaults to curator)")
    role: Optional[str] = Field(None, description="The fallback role description or name (optional)")
    message: str = Field(..., description="The user's message (required, cannot be empty)")
    mode: Optional[str] = Field("writer", description="The chat mode: 'writer' or 'viewer' (optional, defaults to 'writer')")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    role_id: str
    context_used: bool

class VoiceChatRequest(BaseModel):
    text: str
    role_id: str
