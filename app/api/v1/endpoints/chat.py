from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.schemas.chat import ChatRequest, ChatResponse, VoiceChatRequest
from app.services.openai_service import generate_avatar_response, generate_voice_response

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_avatar(request: ChatRequest):
    try:
        response_data = await generate_avatar_response(
            request.user_id, 
            request.message, 
            request.role_id, 
            request.session_id
        )
        return ChatResponse(**response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/voice")
async def chat_voice(request: VoiceChatRequest):
    try:
        audio_content = await generate_voice_response(request.text, request.role_id)
        return Response(content=audio_content, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
