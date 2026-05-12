from openai import AsyncOpenAI
from app.core.config import settings
from app.prompts.templates import PROMPTS
from app.services.vector_store import search_context
import json

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

async def generate_avatar_response(user_id: str, user_message: str, role_id: str, session_id: str) -> dict:
    # 1. RAG 검색
    results = await search_context(user_id, user_message, n_results=3)
    context_text = "\n".join([f"- {doc.page_content}" for doc, _ in results])
    context_used = len(results) > 0
    if not context_text:
        context_text = "특별한 과거 기록이 없습니다."

    # 2. 역할 설정 (간단한 매핑, 실제로는 프롬프트 템플릿 확장이 좋음)
    role_names = {
        "father": "아버지",
        "mother": "어머니",
        "curator": "큐레이터",
        "friend": "친한 친구"
    }
    role_name = role_names.get(role_id, "아버지")

    # 3. 프롬프트 구성
    # AVATAR_CHAT_PROMPT를 조금 더 유연하게 수정하여 사용
    system_prompt = f"당신은 {role_name}입니다. 제공된 과거 기억만을 바탕으로 대화하세요. 지어내지 마세요."
    
    prompt_content = PROMPTS["AVATAR_CHAT_PROMPT"].format(
        context=context_text,
        user_message=user_message
    )
    # 실제 구현에서는 session_id를 사용하여 이전 대화 내역을 가져오는 로직이 필요하지만,
    # 여기서는 간단하게 시스템 프롬프트와 현재 메시지만 처리합니다.
    
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
        "session_id": session_id,
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
