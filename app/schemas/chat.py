from pydantic import BaseModel

class ChatRequest(BaseModel):
    userId: str
    message: str

class ChatResponse(BaseModel):
    response: str
