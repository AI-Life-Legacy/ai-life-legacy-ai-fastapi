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

    async def generate_dynamic_chapters_outline(self, events: List[LifeEvent]) -> List[dict]:
        """
        사용자의 생애 이벤트들을 기반으로 AI가 맞춤형 목차(TOC)를 동적으로 디자인합니다.
        """
        if not events:
            return []
            
        events_json = [ev.model_dump() for ev in events]
        
        system_prompt = """당신은 소설이나 자서전의 목차를 구상하는 베테랑 출판 기획자입니다.
제공된 인물의 실제 생애 사건 리스트(LifeEvent)를 상세히 분석하여, 이 사람의 생애 특징을 관통하는 최적의 맞춤형 목차(TOC) 아웃라인을 구상해 주세요.

[목차 구상 원칙]
1. 획일적인 8단 구성에서 벗어나, 사건들의 밀도와 고유한 스토리(예: 군대 생활이 길었거나, 창업 도전기, 혹은 특정 취미/여행에 대한 이야기 등)를 파악해 최소 3개에서 최대 6개 사이의 맞춤형 챕터로 구성하세요.
2. 만약 특정 시기(예: 유년시절)의 기억이 너무 적다면 청소년기와 합쳐서 한 장으로 묶거나 과감히 생략하고, 사건이 많은 시기는 세분화하십시오.
3. 각 챕터의 제목(chapter_title)은 단순 명사가 아닌 소설처럼 감성적이고 깊이 있게 지어주세요 (예: '군산 바다의 짠내와 따뜻했던 품').
4. 각 챕터에는 다음 필드를 포함해 주세요:
   - chapter_num: 1부터 시작하는 순차 정수
   - chapter_title: 챕터의 제목
   - chapter_type: childhood, school, youth, career, marriage, hobby, self_reflection, family 중 가장 정서가 어울리는 기존 무드 코드 1개 선택 (CSS 스타일링 매핑용)
   - description: 챕터가 다룰 주요 인생 여정 묘사
   - mood: 정서 톤 (chapter_type과 동일한 값 권장)
   - event_ids: 이 챕터에 포함할 LifeEvent의 id들의 리스트
5. 제공된 모든 LifeEvent ID가 최소 한 번은 챕터의 'event_ids'에 빠짐없이 매핑되어야 합니다. 누락되는 인생 기억이 없도록 하십시오.
6. JSON 형식의 'chapters' 배열로 반환하세요.
"""
        user_prompt = f"""[추출된 사용자 인생 사건 (LifeEvents)]
{json.dumps(events_json, ensure_ascii=False, indent=2)}

위 사건들을 바탕으로 이 유저만을 위한 맞춤형 목차 아웃라인을 생성해 주세요.
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
            return data.get("chapters", [])
        except Exception as e:
            print(f"Error generating dynamic chapters outline: {e}")
            return []

    async def build_full_story_structure(self, events: List[LifeEvent]) -> List[ChapterData]:
        # 1. AI를 통한 동적 목차 생성 시도
        dynamic_outline = await self.generate_dynamic_chapters_outline(events)
        
        full_chapters = []
        
        if dynamic_outline:
            print(f"[Dynamic TOC] Successfully generated {len(dynamic_outline)} custom chapters.")
            for ch in dynamic_outline:
                ch_num = ch["chapter_num"]
                ch_title = ch["chapter_title"]
                ch_type = ch.get("chapter_type", "family")
                ch_mood = ch.get("mood", ch_type)
                event_ids = ch.get("event_ids", [])
                
                # 챕터에 할당된 LifeEvent들 추출
                ch_events = [ev for ev in events if ev.id in event_ids]
                
                # 해당 챕터에 할당된 이벤트가 없는 경우, 방어 코드로 폴백 적용
                if not ch_events and events:
                    ch_events = [events[0]]
                
                # 씬 구축
                scenes = await self.build_scenes_for_chapter(ch_type, ch_events)
                
                full_chapters.append(ChapterData(
                    chapter_num=ch_num,
                    chapter_title=ch_title,
                    chapter_type=ch_type,
                    mood=ch_mood,
                    scenes=scenes
                ))
        else:
            # 2. 실패 시 정적 백업 목차 사용
            print("[Dynamic TOC] Falling back to static chapters outline.")
            chapter_buckets = self._assign_events_to_chapters(events)
            for ch_outline in self.chapters_outline:
                ch_type = ch_outline["chapter_type"]
                ch_events = chapter_buckets.get(ch_type, [])
                
                scenes = await self.build_scenes_for_chapter(ch_type, ch_events)
                
                full_chapters.append(ChapterData(
                    chapter_num=ch_outline["chapter_num"],
                    chapter_title=ch_outline["title_hint"],
                    chapter_type=ch_type,
                    mood=ch_type,
                    scenes=scenes
                ))
                
        return full_chapters

scene_builder = SceneBuilder()
