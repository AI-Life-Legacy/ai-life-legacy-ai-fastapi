from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QuestionRequest(BaseModel):
    user_id: Optional[str] = None
    toc_id: int
    current_answer: str
    chat_history: List[dict]

class QuestionResponse(BaseModel):
    question: str

class AutobiographyRequest(BaseModel):
    userId: Optional[str] = None
    user_id: Optional[str] = None
    userName: Optional[str] = "사용자"
    user_name: Optional[str] = None
    answers: Optional[List[Any]] = None
    chapters: Optional[List[Any]] = None
    toc: Optional[List[Any]] = None
    questions: Optional[List[Any]] = None
    force: bool = False

class AutobiographyResponse(BaseModel):
    status: str
    pdf_url: str
    page_count: int
    cached: bool = False


