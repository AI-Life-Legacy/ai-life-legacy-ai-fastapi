from openai import AsyncOpenAI
from app.core.config import settings
from app.services.vector_store import retrieve_all_user_contexts
import os

class AutobiographyService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_autobiography_memoir(self, user_id: str, user_name: str) -> str:
        """
        RAG 기반으로 사용자의 생애 데이터를 검색하여 자연스러운 1인칭 자서전을 생성합니다.
        """
        # 1. 벡터 데이터베이스에서 전체 컨텍스트 검색
        retrieved_context = await retrieve_all_user_contexts(user_id=user_id, limit=30)
        
        if not retrieved_context:
            return "검색된 사용자 데이터가 없습니다. 자서전을 생성할 수 없습니다."

        # 2. 시스템 프롬프트 설정 (AI 작가 페르소나 - 고볼륨 & 극사실주의)
        system_prompt = """You are a master biographer specialized in high-volume, event-driven realistic memoirs.

Your goal is to write a full-length autobiography (5000+ Korean characters) that feels like a real book. 

STRICT RULES FOR VOLUME & DETAIL:
1. MINIMUM 4-5 LONG PARAGRAPHS PER CHAPTER: Each chapter must be a substantial story. Never write just one paragraph for a chapter.
2. SENSORY & INTERNAL DETAIL: For every event, describe:
   - What the narrator saw, heard, and smelled.
   - The narrator's exact internal thoughts at that moment.
   - The specific dialogue or words exchanged (if applicable).
3. 100% FACT PRESERVATION: Do not omit any specific noun (names, objects, places, dates) from the RAG context.
4. NO SUMMARIES: Instead of saying "I had a happy childhood," describe a specific afternoon in the 1969 "감나무 마당" with the "나무 기차" and "친구 철수".
5. CHRONOLOGICAL 10 CHAPTERS: Ensure the story covers birth to future dreams in exactly 8-10 distinct, long chapters.

TONE:
- Realistic, grounded, and sincere. 
- Avoid poetic exaggeration. No "beautiful journey" or "ocean of time" talk.
- Use the voice of "나" (first-person).

Output structure:
제목: <user_name>의 자서전

## <장 제목 1>
[4-5개 이상의 풍부한 문단]

... (반복)

## <장 제목 10>
[4-5개 이상의 풍부한 문단]

Goal: 5000~6000 Korean characters (excluding whitespace). Show us a real book.
"""

        user_prompt = f"""사용자 이름: {user_name}

아래는 RAG 시스템이 검색한 사용자의 생애 관련 정보(인터뷰 및 답변)입니다.
이 정보를 '있는 그대로' 생생하게 살려, 1인칭 회고 형식의 자서전을 작성해주세요.

검색된 컨텍스트:
{retrieved_context}

작성 지침:
- **사실 보존**: 데이터에 포함된 구체적인 연도, 장소, 친구 이름, 사물, 사건을 하나도 빠뜨리지 말고 문맥 속에 자연스럽게 포함시키세요.
- **현실적 톤**: 미사여구로 포장하기보다 담백하고 진솔하게 사실을 서술하세요. (예: "내 인생은 아름다운 여행이었다" -> "1969년 서울에서 태어난 내 삶은 굴곡이 많았지만 그만큼 단단해지는 과정이었다.")
- **분량**: 전체 3500자 이상의 풍부한 분량을 확보하세요. 요약하지 말고, 각 일화를 충분히 상세하게 풀어쓰세요.
- **연대기 순**: 탄생부터 현재, 그리고 미래의 꿈까지 시간 순서대로 구성하세요.
"""

        # 3. AI 생성 요청
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8
        )

        return response.choices[0].message.content

autobiography_service = AutobiographyService()
