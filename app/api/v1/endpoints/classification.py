from fastapi import APIRouter, HTTPException
from app.schemas.classification import UserCaseRequest, UserCaseResponse
from app.services.openai_service import classify_user_case
import json

router = APIRouter()

@router.post("/case", response_model=UserCaseResponse)
async def predict_user_case(request: UserCaseRequest):
    try:
        raw_result = await classify_user_case(request.introText)
        # GPT로부터 받은 표준 JSON 문자열 파싱
        parsed = json.loads(raw_result)
        return UserCaseResponse(**parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
