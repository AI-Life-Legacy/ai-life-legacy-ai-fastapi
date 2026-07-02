from fastapi import APIRouter, HTTPException
from app.schemas.proxy import (
    CaseRequest, CaseResponse, 
    QuestionRequest, CombineRequest, 
    AutobiographyRequest, ChatRequest, SearchRequest
)
from app.services.openai_service import classify_user_case, generate_avatar_response
from app.services.vector_store import search_context
import json

router = APIRouter()

@router.post("/case", response_model=CaseResponse)
async def classify_case_proxy(request: CaseRequest):
    try:
        # 실시간 분류 로직 호출 (선택 사항, 일단 더미와 결합)
        # raw_result = await classify_user_case(request.data)
        # parsed = json.loads(raw_result)
        # case_val = parsed.get("case", "case1")
        
        # 유저가 요청한 더미 응답 형식 준수
        return CaseResponse(
            case="case1",
            summary="사용자의 자기소개를 기반으로 기본 자서전 작성 유형으로 분류했습니다.",
            recommended_chapters=[
                "어린 시절과 첫 기억",
                "청소년기와 학창 시절",
                "가족과 관계",
                "일과 삶",
                "삶의 전환점",
                "기억에 남는 사람들",
                "남기고 싶은 말"
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/question")
async def create_question_proxy(request: QuestionRequest):
    original_question = request.question
    current_answer = request.data or request.current_answer

    # chat_history가 있는 경우 데이터 보완
    if request.chat_history:
        if not original_question:
            # 마지막 AI 메시지를 원본 질문으로 간주
            for msg in reversed(request.chat_history):
                if msg.role == "ai":
                    original_question = msg.content
                    break
        
        if not current_answer:
            # 마지막 User 메시지를 현재 답변으로 간주
            for msg in reversed(request.chat_history):
                if msg.role == "user":
                    current_answer = msg.content
                    break

    if not current_answer:
        raise HTTPException(status_code=400, detail="current_answer is required")

    # 더미 응답 (사용자가 요청한 예시 형식)
    return {
        "question": "그 시절 가족에게 들은 이야기가 더 있나요?"
    }

@router.post("/combine")
async def combine_answers_proxy(request: CombineRequest):
    # 더미 응답
    return {
        "combined_text": f"{request.data1}\n\n{request.data2}"
    }

@router.post("/autobiography")
async def create_autobiography_proxy(request: AutobiographyRequest):
    # 더미 응답
    return {
        "status": "COMPLETED",
        "pdf_url": "http://localhost:8000/storage/data/dummy_autobiography.pdf",
        "page_count": 10
    }

@router.post("/chat")
async def chat_proxy(request: ChatRequest):
    try:
        # 실제 아바타 응답 서비스 호출 시도
        response_data = await generate_avatar_response(
            request.user_id,
            request.message,
            request.role_id,
            request.session_id
        )
        return response_data
    except Exception:
        # 실패 시 더미 응답
        return {
            "answer": f"안녕! {request.message}라고 했니? 반갑구나.",
            "session_id": request.session_id,
            "context_used": False
        }

@router.post("/search")
async def search_proxy(request: SearchRequest):
    try:
        # 실제 검색 로직 호출 시도 (user_id가 필요한데 SearchRequest에는 없음)
        # 일단 더미 데이터 반환
        return {
            "results": [
                {"text": "사용자의 과거 기억 1...", "score": 0.95},
                {"text": "사용자의 과거 기억 2...", "score": 0.88}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
