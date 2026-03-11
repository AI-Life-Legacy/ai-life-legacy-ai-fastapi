from pydantic import BaseModel

class ChatRequest(BaseModel):
    userId: str
    message: str
    role: str = "친한 친구"

class ChatResponse(BaseModel):
    response: str
