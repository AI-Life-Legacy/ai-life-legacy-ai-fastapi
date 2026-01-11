from fastapi import APIRouter, HTTPException
from app.schemas.generation import (
    QuestionRequest, QuestionResponse, 
    AutobiographyRequest, AutobiographyResponse
)
from app.services.openai_service import generate_follow_up_question, combine_answers_to_autobiography

router = APIRouter()

@router.post("/question", response_model=QuestionResponse)
async def create_follow_up_question(request: QuestionRequest):
    try:
        question_text = await generate_follow_up_question(request.originalQuestion, request.userAnswer)
        return QuestionResponse(question=question_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/autobiography", response_model=AutobiographyResponse)
async def create_autobiography(request: AutobiographyRequest):
    try:
        content = await combine_answers_to_autobiography(request.pairs)
        return AutobiographyResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
