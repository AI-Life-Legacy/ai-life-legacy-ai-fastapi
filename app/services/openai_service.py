from openai import AsyncOpenAI
from app.core.config import settings
from app.prompts.templates import PROMPTS
import json

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def classify_user_case(intro_text: str) -> str:
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

async def generate_follow_up_question(original_question: str, user_answer: str) -> str:
    prompt_content = PROMPTS["QUESTION_GENERATION_USER"].format(
        question=original_question, 
        answer=user_answer
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt_content}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

async def combine_answers_to_autobiography(pairs: list) -> str:
    # pairs 리스트에서 첫 번째와 두 번째 질문/답변을 추출한다고 가정 (명세상 question1, question2 등)
    # 리스트 길이가 2 이상이어야 함. 안전하게 처리.
    q1 = pairs[0].question if len(pairs) > 0 else ""
    a1 = pairs[0].answer if len(pairs) > 0 else ""
    q2 = pairs[1].question if len(pairs) > 1 else ""
    a2 = pairs[1].answer if len(pairs) > 1 else ""
    
    # 더 많은 질문이 있을 경우 어떻게 할지 명세에는 없으므로, 일단 2개만 처리하거나 반복문으로 합쳐야 함.
    # 현재 `combine.prompt.ts`는 명시적으로 2개의 질문/답변을 인자로 받음.
    
    prompt_content = PROMPTS["AUTOBIOGRAPHY_COMBINATION_USER"].format(
        q1=q1, a1=a1, q2=q2, a2=a2
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt_content}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
