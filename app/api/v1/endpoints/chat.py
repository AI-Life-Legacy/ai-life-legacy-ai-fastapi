from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.schemas.chat import ChatRequest, ChatResponse, VoiceChatRequest
from app.services.openai_service import generate_avatar_response, generate_voice_response
from openai import AuthenticationError, PermissionDeniedError, RateLimitError

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_avatar(request: ChatRequest):
    # message가 비어 있으면 400 반환
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어 있습니다. 올바른 메시지를 입력해 주세요.")

    try:
        response_data = await generate_avatar_response(
            user_id=request.user_id, 
            user_message=request.message, 
            role_id=request.role_id, 
            session_id=request.session_id,
            role=request.role,
            viewer_id=request.viewer_id,
            mode=request.mode
        )
        return ChatResponse(**response_data)
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"OpenAI Authentication Error: {str(e)}")
    except PermissionDeniedError as e:
        raise HTTPException(status_code=403, detail=f"OpenAI Permission/Access Error: {str(e)}")
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"OpenAI Rate Limit / Quota Error: {str(e)}")
    except Exception as e:
        err_msg = str(e).lower()
        if "authentication" in err_msg or "apikey" in err_msg or "api_key" in err_msg:
            raise HTTPException(status_code=401, detail=f"OpenAI Authentication Error: {str(e)}")
        elif "permission" in err_msg or "model_not_found" in err_msg or "access" in err_msg:
            raise HTTPException(status_code=403, detail=f"OpenAI Permission/Access Error: {str(e)}")
        elif "quota" in err_msg or "rate_limit" in err_msg or "rate limit" in err_msg:
            raise HTTPException(status_code=429, detail=f"OpenAI Rate Limit / Quota Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/voice")
async def chat_voice(request: VoiceChatRequest):
    try:
        audio_content = await generate_voice_response(request.text, request.role_id)
        return Response(content=audio_content, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
