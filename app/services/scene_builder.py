from typing import List, Dict
from app.schemas.story import LifeEvent, Scene, ChapterData
from openai import AsyncOpenAI
from app.core.config import settings
import json

class SceneBuilder:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # 8단계 생애 주기
        self.chapters_outline = [
            {"chapter_num": 1, "chapter_type": "childhood", "title_hint": "유년기와 가족"},
            {"chapter_num": 2, "chapter_type": "school", "title_hint": "학창시절과 청소년기"},
            {"chapter_num": 3, "chapter_type": "youth", "title_hint": "대학과 사회 진출"},
            {"chapter_num": 4, "chapter_type": "marriage", "title_hint": "결혼과 가정 형성"},
            {"chapter_num": 5, "chapter_type": "career", "title_hint": "직장 도전과 위기"},
            {"chapter_num": 6, "chapter_type": "hobby", "title_hint": "현재 삶과 취미"},
            {"chapter_num": 7, "chapter_type": "self_reflection", "title_hint": "건강과 삶의 태도"},
            {"chapter_num": 8, "chapter_type": "family", "title_hint": "미래 계획과 인생 철학"}
        ]

    def _assign_events_to_chapters(self, events: List[LifeEvent]) -> Dict[str, List[LifeEvent]]:
        """
        LifeEvent들을 allowed life_stage에 기반하여 챕터 타입별로 간략히 분류합니다.
        """
        chapter_buckets = {ch["chapter_type"]: [] for ch in self.chapters_outline}
        
        # 챕터별 허용 라이프 스테이지 매핑
        allowed_stages = {
            "childhood": ["childhood"],
            "school": ["youth"],
            "youth": ["university", "early_career"],
            "marriage": ["early_career", "present"],
            "career": ["career_crisis", "early_career"],
            "hobby": ["present"],
            "self_reflection": ["present", "future"],
            "family": ["present", "future"]
        }
        
        for ev in events:
            stage = getattr(ev, 'life_stage', 'present')
            
            # 이벤트 분배 (중복 허용: 한 사건이 여러 챕터의 재료가 될 수 있지만 Scene Builder가 걸러줌)
            # 여기서는 가장 적합한 챕터를 찾아 1곳에만 넣는 방식 (선착순 또는 특정 우선순위)
            assigned = False
            
            # 1. 명확한 매칭 시도
            if ev.event_type == "marriage" and stage in allowed_stages["marriage"]:
                chapter_buckets["marriage"].append(ev)
                assigned = True
            elif ev.event_type in ["crisis", "career"] and stage in allowed_stages["career"]:
                chapter_buckets["career"].append(ev)
                assigned = True
            elif ev.event_type == "school" and stage in allowed_stages["school"]:
                chapter_buckets["school"].append(ev)
                assigned = True
            
            if not assigned:
                # 2. Stage 기반 매칭
                for ch_type, stages in allowed_stages.items():
                    if stage in stages:
                        chapter_buckets[ch_type].append(ev)
                        assigned = True
                        break # 첫 매칭에 분배
            
            if not assigned:
                # 3. Fallback
                chapter_buckets["family"].append(ev)
                
        return chapter_buckets

    async def build_scenes_for_chapter(self, chapter_type: str, events: List[LifeEvent]) -> List[Scene]:
        """
        한 챕터에 속한 여러 사건(LifeEvent)들을 GPT를 이용해 의미 있는 씬(Scene) 단위로 병합 및 구성합니다.
        최소 2개 이상의 사건이 1개의 씬에 포함되도록 지시합니다.
        """
        if not events:
            return []
            
        events_json = [ev.model_dump() for ev in events]
        
        system_prompt = """당신은 수집된 인생 사건들을 '소설이나 영화의 장면(Scene)'으로 재구성하는 서사 기획자입니다.
주어진 LifeEvent 목록을 읽고, 의미상 연결되는 사건들을 묶어 Scene들의 배열로 만들어주세요.

[Scene 구성 원칙]
1. 하나의 Scene에는 가급적 2개 이상의 LifeEvent를 포함시켜 다채로운 서사를 만드세요.
2. 각 Scene의 title은 감성적이고 소제목답게 지어주세요.
3. conflict(내적/외적 갈등)와 turning_point(전환점)를 명확히 찾거나 문맥상 부여하세요.
4. emotion_intensity는 1부터 10 사이의 숫자입니다. 감정적으로 매우 중요하고 격동적인 씬이면 8~10을 부여하세요.
5. spread_hint: emotion_intensity가 8 이상이면 "needs_quote_after"라고 적어, 레이아웃 엔진이 이 씬 다음에 인용구 템플릿을 넣도록 유도하세요.
6. JSON 구조의 'scenes' 배열로 반환하세요.

출력 예시:
{
  "scenes": [
    {
      "title": "뒷산에서의 작은 모험",
      "setting": "봄날의 따스한 주말 아침, 동네 뒷산",
      "characters": ["나", "아버지"],
      "conflict": "처음 올라가보는 높은 산길에 대한 두려움",
      "turning_point": "아버지가 꽉 잡아준 손의 온기",
      "resolution": "두려움을 극복하고 정상에 오름",
      "reflection": "그때 배운 용기가 삶의 작은 고비마다 나를 지탱했다.",
      "emotion_intensity": 7,
      "spread_hint": "",
      "linked_events": [{"id": "event_1", ...}] // 원본 LifeEvent 객체들
    }
  ]
}
"""
        user_prompt = f"""[해당 챕터: {chapter_type}의 LifeEvents]
{json.dumps(events_json, ensure_ascii=False, indent=2)}

이 사건들을 바탕으로 자연스럽게 이어지는 Scene들을 만들어 JSON으로 반환하세요.
"""

        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_AUTOBIOGRAPHY_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
            
            data = json.loads(response.choices[0].message.content)
            
            scenes = []
            for s in data.get("scenes", []):
                # JSON 리턴받은 linked_events가 dict의 리스트이므로 Scene 객체 생성시 
                # Pydantic이 List[LifeEvent]로 자동 캐스팅합니다.
                scenes.append(Scene(**s))
                
            return scenes
        except Exception as e:
            print(f"Error building scenes for {chapter_type}: {e}")
            return []

    async def build_full_story_structure(self, events: List[LifeEvent]) -> List[ChapterData]:
        chapter_buckets = self._assign_events_to_chapters(events)
        
        full_chapters = []
        for ch_outline in self.chapters_outline:
            ch_type = ch_outline["chapter_type"]
            ch_events = chapter_buckets.get(ch_type, [])
            
            # 씬 구축
            scenes = await self.build_scenes_for_chapter(ch_type, ch_events)
            
            full_chapters.append(ChapterData(
                chapter_num=ch_outline["chapter_num"],
                chapter_title=ch_outline["title_hint"],
                chapter_type=ch_type,
                mood=ch_type, # 기본적으로 chapter_type을 mood로 사용. 추후 임베딩 기반 개선 가능
                scenes=scenes
            ))
            
        return full_chapters

scene_builder = SceneBuilder()
