from openai import AsyncOpenAI
from app.core.config import settings
from app.prompts.templates import PROMPTS
from app.services.vector_store import search_context
import json
from typing import Optional


client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def classify_user_case(intro_text: str) -> str:
    # 데이터가 너무 적을 경우(공백 포함 5자 미만) 디폴트 case1 반환
    if not intro_text or len(intro_text.strip()) < 5:
        return json.dumps({"case": "case1", "reasoning": "Input too short, defaulted to case1"})

    # 사용자 프롬프트에 데이터를 주입
    prompt_content = PROMPTS["CASE_CLASSIFICATION_USER"].format(user_intro_text=intro_text)
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt_content}
        ],
        temperature=0.0
    )
    # 프롬프트가 'case1' 형태의 문자열만 반환하도록 지시하므로 그대로 리턴
    content = response.choices[0].message.content.strip()
    
    # API 응답 모델(UserCaseResponse)이 JSON({case: "..."})을 기대하므로,
    # 여기서 텍스트("case1")를 JSON 형식이 되도록 변환하여 리턴하거나, 
    # 호출부(endpoint)에서 처리하도록 할 수 있습니다. 
    # 현재 엔드포인트 코드는 json.loads()를 수행하므로, 여기서 JSON 문자열을 만들어줍니다.
    # 만약 AI가 실수로 다른 말을 덧붙였을 경우를 대비해 정규식 등으로 파싱하는 게 안전하지만,
    # 일단 프롬프트를 믿고 단순 래핑합니다.
    
    return json.dumps({"case": content, "reasoning": "Classified by AI"})

async def generate_follow_up_question(user_id: str, current_answer: str, chat_history: list) -> str:
    # 1. RAG: 관련 문맥 검색 (최근 답변 위주로)
    results = await search_context(user_id, current_answer, n_results=3)
    context_text = "\n".join([f"- {doc.page_content}" for doc, _ in results])
    if not context_text:
        context_text = "관련된 과거 기록이 없습니다."

    # 2. 대화 내역 포맷팅
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])

    # 3. 프롬프트 구성
    prompt_content = f"""
    [대화 내역]
    {history_text}
    
    [최근 답변]
    {current_answer}
    
    [참고 문맥]
    {context_text}
    
    위의 대화 내역과 최근 답변, 그리고 참고 문맥을 바탕으로 자연스러운 꼬리 질문을 하나 만들어줘.
    결과물에는 따옴표나 추가 설명 없이 오직 질문만 작성해줘.
    """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt_content}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def map_role_id(role_id: Optional[str], role: Optional[str]) -> str:
    r_id = role_id.strip() if role_id else ""
    r_name = role.strip() if role else ""
    
    mapping = {
        "curator": "curator",
        "큐레이터": "curator",
        "father": "father",
        "아버지": "father",
        "mother": "mother",
        "어머니": "mother",
        "self": "self",
        "self": "self",
        "나": "self",
        "sister": "sister",
        "누나": "sister",
        "언니": "sister",
        "여동생": "sister",
        "brother": "brother",
        "형": "brother",
        "오빠": "brother",
        "남동생": "brother"
    }
    
    if r_id:
        mapped = mapping.get(r_id.lower())
        if mapped:
            return mapped
        if r_id.lower() in ["curator", "father", "mother", "self", "sister", "brother"]:
            return r_id.lower()
            
    if r_name:
        mapped = mapping.get(r_name.lower())
        if mapped:
            return mapped
        if r_name.lower() in ["curator", "father", "mother", "self", "sister", "brother"]:
            return r_name.lower()
            
    return "curator"

async def generate_avatar_response(
    user_id: Optional[str] = None,
    user_message: str = "",
    role_id: Optional[str] = None,
    session_id: Optional[str] = None,
    role: Optional[str] = None,
    viewer_id: Optional[str] = None,
    mode: str = "writer"
) -> dict:
    import uuid
    
    final_user_id = user_id or "anonymous"
    final_role_id = map_role_id(role_id, role)
    final_session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
    
    if viewer_id:
        print(f"[Chat] Mode: {mode}, Author: {final_user_id}, Viewer: {viewer_id}, Session: {final_session_id}")
    else:
        print(f"[Chat] Mode: {mode}, Author: {final_user_id}, Session: {final_session_id}")
    
    # 1. RAG 검색
    context_text = ""
    context_used = False
    
    try:
        # anonymous, unknown 또는 비어있는 user_id는 RAG 검색을 건너뜀
        if final_user_id not in [None, "", "anonymous", "unknown"]:
            results = await search_context(final_user_id, user_message, n_results=3)
            if results:
                context_chunks = [f"- {doc.page_content}" for doc, _ in results if doc and doc.page_content]
                context_text = "\n".join(context_chunks)
                context_used = len(context_text.strip()) > 0
    except Exception as e:
        print(f"Warning: Failed to retrieve RAG context for user {final_user_id}: {e}")
        context_text = ""
        context_used = False

    if not context_text:
        context_text = "제공된 과거 기억이나 자서전 기록이 없습니다. 일상적인 대화 어조로 성심껏 응답하세요."

    # 2. 역할 설정 및 페르소나 선택
    prompt_key = f"AVATAR_SYSTEM_{final_role_id.upper()}"
    system_prompt = PROMPTS.get(prompt_key, PROMPTS["AVATAR_SYSTEM_CURATOR"])

    # 3. 프롬프트 구성
    prompt_content = PROMPTS["AVATAR_USER_PROMPT"].format(
        context=context_text,
        user_message=user_message
    )
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_content}
        ],
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    
    return {
        "answer": answer,
        "session_id": final_session_id,
        "role_id": final_role_id,
        "context_used": context_used
    }

async def generate_voice_response(text: str, role_id: str):
    # OpenAI TTS 사용
    # role_id에 따라 목소리 매핑
    voices = {
        "father": "echo",
        "mother": "nova",
        "curator": "onyx",
        "friend": "alloy"
    }
    voice = voices.get(role_id, "alloy")
    
    response = await client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    return response.content
