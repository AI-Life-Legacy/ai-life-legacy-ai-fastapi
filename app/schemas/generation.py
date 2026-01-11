from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    originalQuestion: str
    userAnswer: str

class QuestionResponse(BaseModel):
    question: str

class QaPair(BaseModel):
    question: str
    answer: str

class AutobiographyRequest(BaseModel):
    pairs: List[QaPair]

class AutobiographyResponse(BaseModel):
    content: str
