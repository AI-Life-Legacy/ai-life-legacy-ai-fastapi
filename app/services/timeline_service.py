import json
from typing import List, Dict, Any
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.story import LifeEvent

class TimelineService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def reconstruct_timeline(self, retrieved_context: str) -> List[LifeEvent]:
        """
        RAG에서 검색된 문맥(context)을 기반으로 생애 사건(LifeEvent) 목록을 추출하고 시간순으로 정렬하여 반환합니다.
        """
        system_prompt = """당신은 사람의 기억을 시간순으로 구조화하는 '타임라인 분석가'입니다.
제공된 인터뷰(또는 회고) 텍스트를 분석하여 중요한 인생 사건(LifeEvent)들을 추출하세요.

[사건 추출 원칙]
1. 구체적인 연도나 나이가 있다면 적극 반영하세요. (단, 정확한 수치가 없다면 문맥을 보고 estimated_age 나 estimated_year를 합리적으로 추정하세요.)
2. 각 사건의 핵심(event_summary)은 1~2문장으로 간결하게 요약하세요.
3. 인물(people), 장소(location), 핵심 감정(emotion)을 명확하게 뽑아주세요.
4. event_type은 다음 중 하나로 배정하세요: childhood, school, youth, family, career, crisis, hobby, romance, marriage, self_reflection
5. 시간 순서대로 정렬(Chronological Sorting)해서 반환해야 합니다. (어릴 때 -> 나이 들었을 때)
6. 특정 사건이 원인이 되어 뒤의 사건이 발생했다면, 해당 앞 사건의 leads_to 배열에 뒤 사건의 id를 넣으세요.
7. [매우 중요] 생애 주기(life_stage) 필드를 반드시 명시하세요. 허용된 값은 다음 7개뿐입니다:
   - childhood, youth, university, early_career, career_crisis, present, future
   - 불확실하거나 애매한 경우 가장 가까운 과거 스테이지나 present로 분류하세요. 없는 정보를 지어내지 마세요.

출력은 반드시 다음 JSON 스키마를 따라야 합니다. JSON Root는 객체(Object)이며, 'events' 필드 안에 배열로 사건을 담으세요.
{
  "events": [
    {
      "id": "event_1",
      "estimated_age": 7,
      "estimated_year": 1990,
      "location": "서울 성북구",
      "people": ["아버지", "어머니"],
      "event_summary": "초등학교 입학 전, 부모님과 함께 동네 뒷산에 매주 올랐던 기억",
      "emotion": "따뜻함, 안도감",
      "event_type": "family",
      "life_stage": "childhood",
      "leads_to": ["event_2"]
    }
  ]
}
"""
        user_prompt = f"""[사용자의 과거 기억 기록]
{retrieved_context}

위 기록을 분석하여 인생 사건들을 시간순으로 추출해 JSON 형태로 반환해주세요.
"""

        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_AUTOBIOGRAPHY_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # 사건 추출이므로 낮은 temperature
                response_format={ "type": "json_object" } 
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            life_events = []
            for item in data.get("events", []):
                life_events.append(LifeEvent(**item))
                
            return life_events
        except Exception as e:
            print(f"Error in timeline reconstruction: {e}")
            return []

timeline_service = TimelineService()

