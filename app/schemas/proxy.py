from pydantic import BaseModel
from typing import List, Any, Optional

class CaseRequest(BaseModel):
    data: str

class CaseResponse(BaseModel):
    case: str
    summary: str
    recommended_chapters: List[str]

class ChatMessage(BaseModel):
    role: str
    content: str

class QuestionRequest(BaseModel):
    question: Optional[str] = None
    data: Optional[str] = None
    toc_id: Optional[int] = None
    current_answer: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = None

class CombineRequest(BaseModel):
    question1: str
    data1: str
    question2: str
    data2: str

class AutobiographyRequest(BaseModel):
    answers: List[Any]

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    role_id: str
    message: str

class SearchRequest(BaseModel):
    query: str
