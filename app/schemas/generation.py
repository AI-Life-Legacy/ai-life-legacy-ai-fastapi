from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    toc_id: int
    current_answer: str
    chat_history: List[dict]

class QuestionResponse(BaseModel):
    question: str

class AutobiographyRequest(BaseModel):
    userId: str
    userName: str = "사용자"
    force: bool = False

class AutobiographyResponse(BaseModel):
    status: str
    pdf_url: str
    page_count: int
    cached: bool = False

