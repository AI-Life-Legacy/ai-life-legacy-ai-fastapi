from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    userId: str
    originalQuestion: str
    userAnswer: str

class QuestionResponse(BaseModel):
    question: str

class QaPair(BaseModel):
    question: str
    answer: str

class AutobiographyRequest(BaseModel):
    userId: str
    pairs: List[QaPair]

class AutobiographyResponse(BaseModel):
    content: str
