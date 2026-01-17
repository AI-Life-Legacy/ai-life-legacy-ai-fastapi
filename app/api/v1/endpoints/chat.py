from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.openai_service import generate_avatar_response

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_avatar(request: ChatRequest):
    try:
        response_text = await generate_avatar_response(request.userId, request.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
