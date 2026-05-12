from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    role_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    context_used: bool

class VoiceChatRequest(BaseModel):
    text: str
    role_id: str
