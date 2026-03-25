from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    userId: str
    originalQuestion: str
    userAnswer: str

class QuestionResponse(BaseModel):
    question: str

class AutobiographyRequest(BaseModel):
    userId: str
    userName: str = "사용자"

class AutobiographyResponse(BaseModel):
    status: str
    mdPath: str
    pdfPath: str
