from openai import AsyncOpenAI
from app.core.config import settings
from app.services.vector_store import retrieve_all_user_contexts, retrieve_chapter_contexts
from app.services.timeline_service import timeline_service
from app.services.scene_builder import scene_builder
import os
import json

CHAPTER_ROLES = {
    "childhood": {
        "theme": "성장 배경, 가족 분위기, 최초 기억 형성",
        "avoid": "정치적 이슈, 직업적 커리어, 연애",
        "tone": "따뜻함, 호기심, 애틋함",
        "role": "자서전의 서막, 가치관의 뿌리 제시"
    },
    "school": {
        "theme": "친구 관계, 정체성 탐색, 작은 반항과 학업/도전",
        "avoid": "심각한 어른의 고충, 재정적 파탄, 은퇴 후 이야기",
        "tone": "활기참, 풋풋함, 변화",
        "role": "자아 형성 및 관계의 확장"
    },
    "youth": {
        "theme": "진로 선택, 현실 진입, 첫 사회생활의 낯섦과 적응",
        "avoid": "노년의 회한, 가족의 소멸 등 후반부 주제",
        "tone": "열정, 좌충우돌, 긴장감",
        "role": "성인으로서의 첫 발돋움과 홀로서기"
    },
    "marriage": {
        "theme": "배우자와의 만남, 관계의 확장, 책임감, 안정",
        "avoid": "가족 밖의 지엽적인 업무 에피소드 집중",
        "tone": "포용, 사랑, 헌신, 따뜻함",
        "role": "가족 형성 및 개인에서 공동체로의 변화"
    },
    "career": {
        "theme": "사회적 압박, 직장에서의 갈등/위기, 실패와 버팀, 성취",
        "avoid": "지나치게 평온하고 안일한 회상 위주 서술",
        "tone": "치열함, 결단, 성취감",
        "role": "인생의 전성기 및 극복의 서사"
    },
    "hobby": {
        "theme": "일상의 균형, 자아 발견, 회복, 취미 생활",
        "avoid": "과도한 업무 스트레스 묘사",
        "tone": "여유, 즐거움, 소소한 행복",
        "role": "삶의 숨고르기 및 개인적 내면 탐구"
    },
    "self_reflection": {
        "theme": "성찰, 가치관 변화, 삶과 건강에 대한 태도",
        "avoid": "단순한 사건 나열",
        "tone": "차분함, 달관, 진지함",
        "role": "인생의 깊이와 성숙한 시각의 표현"
    },
    "family": {
        "theme": "앞으로의 다짐, 남기고 싶은 말, 인생 철학, 가족의 미래",
        "avoid": "과거 사건 중심의 긴 회상",
        "tone": "소망, 감사, 단단함",
        "role": "자서전의 울림 있는 마무리"
    }
}

def group_answers_by_chapter(answers: list) -> dict:
    grouped = {
        "childhood": [],
        "school": [],
        "youth": [],
        "marriage": [],
        "career": [],
        "hobby": [],
        "self_reflection": [],
        "family": []
    }
    
    # 챕터 감지를 위한 간단한 키워드 맵 (질문이나 답변에 이 단어가 들어있으면 해당 챕터로 분류)
    keyword_map = {
        "childhood": ["유년", "어린 시절", "태어난", "고향", "부모님", "아버지", "어머니", "형제", "자매"],
        "school": ["학교", "학창", "선생님", "친구", "소풍", "공부", "사춘기", "중학교", "고등학교", "초등학교"],
        "youth": ["대학", "20대", "청년", "첫 직장", "군대", "진로", "전공", "취업"],
        "marriage": ["결혼", "배우자", "남편", "아내", "연애", "신혼", "첫째", "출산", "아이들", "자식"],
        "career": ["직장", "회사", "업무", "성취", "도전", "실패", "퇴사", "승진", "동료", "상사", "사업"],
        "hobby": ["취미", "여가", "주말", "운동", "그림", "음악", "여행", "휴식", "좋아하는"],
        "self_reflection": ["건강", "나이", "깨달음", "가치관", "인생", "태도", "후회", "보람", "성찰"],
        "family": ["미래", "계획", "자녀", "손주", "가족", "남기고", "다짐", "꿈", "철학"]
    }
    
    for item in answers:
        item_text = ""
        item_chapter = None
        
        if isinstance(item, dict):
            # toc_id 등을 통한 매칭 (1 -> childhood, 2 -> school 등)
            toc_id = item.get("toc_id") or item.get("tocId")
            if toc_id is not None:
                toc_id_map = {
                    1: "childhood",
                    2: "school",
                    3: "youth",
                    4: "marriage",
                    5: "career",
                    6: "hobby",
                    7: "self_reflection",
                    8: "family"
                }
                item_chapter = toc_id_map.get(int(toc_id))
            
            if not item_chapter:
                item_chapter = item.get("chapter_type") or item.get("chapterType")
                
            q_text = item.get("question_text") or item.get("questionText") or item.get("question") or ""
            if isinstance(q_text, dict):
                q_text = q_text.get("question_text") or q_text.get("questionText") or q_text.get("title") or ""
            a_text = item.get("answer_text") or item.get("answerText") or item.get("text") or item.get("content") or item.get("answer") or ""
            
            if q_text and a_text:
                item_text = f"Q: {q_text}\nA: {a_text}"
            else:
                item_text = a_text or str(item)
        else:
            item_text = str(item)
            
        if not item_chapter:
            detected_scores = {k: 0 for k in keyword_map.keys()}
            for ch_type, kw_list in keyword_map.items():
                for kw in kw_list:
                    if kw in item_text:
                        detected_scores[ch_type] += 1
            best_ch = max(detected_scores, key=detected_scores.get)
            if detected_scores[best_ch] > 0:
                item_chapter = best_ch
            else:
                item_chapter = "family" # Fallback
                
        if item_chapter in grouped:
            grouped[item_chapter].append(item_text)
            
    return grouped

class AutobiographyService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def _extract_personal_details(self, context: str) -> dict:
        system_prompt = """당신은 인물, 장소, 사건 등의 고유 정보를 정확하게 추출하는 데이터 분석가입니다.
제공된 인터뷰/문맥 데이터에서 다음 유형의 고유 요소를 빠짐없이 추출하여 JSON 형식으로 반환하세요.
오직 텍스트에 등장하는 사실만 추출해야 하며, 절대 지어내지 마세요.
{
  "people": ["사람 이름, 가족 관계 단위나 직책 등"],
  "places": ["장소, 지역, 건물명 등"],
  "activities": ["취미, 특기, 반복 활동 등"],
  "achievements": ["수상 내역, 자격, 성취 등"],
  "events": ["특정 사건명, 여행, 위기 순간 등"]
}
"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                response_format={ "type": "json_object" },
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Detail extraction error: {e}")
            return {"people": [], "places": [], "activities": [], "achievements": [], "events": []}

    async def generate_autobiography_memoir(self, user_id: str, user_name: str, retrieved_context: str = None, answers: list = None) -> str:
        """
        Timeline Graph와 Scene Composition을 거쳐 서사를 생성합니다.
        """
        if retrieved_context is None:
            print(f"[{user_name}] 1. 전체 문맥 검색 중...")
            # 1. 벡터 데이터베이스에서 전체 컨텍스트 검색
            if answers:
                # Circular import 방지를 위해 헬퍼 코드를 인라인 임포트하거나 직접 구현 가능
                # 여기서는 answers가 있는 경우 retrieved_context가 이미 바깥에서 제공되므로 실행되지 않겠지만, Fallback으로 안전하게 방어
                from app.api.v1.endpoints.generation import extract_context_from_answers
                retrieved_context = extract_context_from_answers(answers)
            else:
                retrieved_context = await retrieve_all_user_contexts(user_id=user_id, limit=30)
        
        if not retrieved_context:
            return "검색된 사용자 데이터가 없습니다. 자서전을 생성할 수 없습니다."


        print(f"[{user_name}] 2. Personal Detail & Timeline 추출 중...")
        # 디테일 추출 (고유명사 증폭용)
        personal_details = await self._extract_personal_details(retrieved_context)
        
        # Timeline 추출
        timeline_events = await timeline_service.reconstruct_timeline(retrieved_context)
        
        print(f"[{user_name}] 3. Scene 단위 챕터 구조화 중...")
        # Scene 구조로 재배치
        chapter_data_list = await scene_builder.build_full_story_structure(timeline_events)
        
        # 각 챕터별로 본문 생성
        full_markdown = f"제목: {user_name}의 자서전\n\n"
        
        grouped_answers = group_answers_by_chapter(answers) if answers else {}
        
        for i, chapter in enumerate(chapter_data_list):
            next_chapter = chapter_data_list[i+1] if i + 1 < len(chapter_data_list) else None
            print(f"[{user_name}] 4. 챕터 생성 중: {chapter.chapter_num}. {chapter.chapter_title}")
            
            # 이 챕터에 대한 추가적인 상세 Context 검색
            if answers:
                chapter_context = "\n\n".join(grouped_answers.get(chapter.chapter_type, []))
                # 해당 챕터에 질문이 없는 경우 전체 컨텍스트를 기본 뼈대로 사용
                if not chapter_context:
                    chapter_context = retrieved_context
            else:
                chapter_context = await retrieve_chapter_contexts(user_id, chapter.chapter_type, limit=10)
            
            # Post-check 용이성을 위해 Retry 로직 래핑 가능 (현재는 단일 패스)
            chapter_result = await self._generate_chapter_text(user_name, chapter, next_chapter, chapter_context, personal_details)
            
            # (선택) Post-check 로직: 미래 챕터에 과거 단어 너무 많으면 재시도 등...
            
            chapter_text = chapter_result.get("content", "")
            chapter_quote = chapter_result.get("quote", "")
            
            chapter.generated_text = chapter_text
            
            full_markdown += f"## {chapter.chapter_title}\n"
            full_markdown += f"<!-- MOOD: {chapter.mood} -->\n"
            full_markdown += f"{chapter_text}\n\n"
            if chapter_quote:
                full_markdown += f"<!-- QUOTE: {chapter_quote} -->\n\n"

        return full_markdown

    async def _generate_chapter_text(self, user_name: str, chapter, next_chapter, additional_context: str, personal_details: dict) -> dict:
        """
        구조화된 Scene 정보를 기반으로 챕터 텍스트와 에센셜 Quote를 생성합니다.
        """
        scenes_json = [s.model_dump() for s in chapter.scenes]
        role_info = CHAPTER_ROLES.get(chapter.chapter_type, CHAPTER_ROLES["family"])
        
        system_prompt = f"""당신은 한 사람의 생애를 깊이 있는 서사로 표현하는 베테랑 자서전 작가입니다.
주어진 Scene 구조와 관련 문맥을 살려 1개의 챕터를 작성하세요.

[챕터 역할 (Chapter Role)]
- 이 장의 역할: {role_info['role']}
- 다뤄야 할 주제: {role_info['theme']}
- 정서 톤: {role_info['tone']}
- 허용된 생애 주기: {chapter.chapter_type}에 맞는 이야기만 집중하고 다른 생애 이야기로 길게 새지 마세요.
- 피해야 할 서술: {role_info['avoid']}

[서사 구조 제약 (Narrative Arc Rule) 및 연결 문장(Transition)]
1. 단순 사건 나열 금지: 문단 구성 시 가급적 [상황/배경 → 갈등/선택 → 변화/결과 → 의미 성찰]의 흐름을 반영하세요.
2. 모든 문단을 억지 교훈으로 끝내지 마세요. 자연스러운 여운을 남기세요.
3. 이전 Scene과 다음 Scene이 물 흐르듯 이어지도록 시간 경과나 내면의 변화를 나타내는 부드러운 전환(Transition)을 사용하세요.
4. [매우 중요] 본문의 마지막 문단 끝에는 반드시 다음 챕터로 자연스럽게 넘어가는 1~2문장의 '연결 문장(Transition Sentence)'을 작성하세요.
   - 단, 마지막 챕터일 경우는 제외합니다.
   - "다음 장에서는 ~에 대해 이야기하겠다" 식의 직설적인 표현 대신, 현재 챕터의 경험이 어떻게 다음 챕터의 밑거름이 되었는지 소설처럼 부드럽게 암시하세요.

[디테일 증폭 제약 (Personal Detail Amplifier)]
- 제공된 '고유명사 리스트(Personal Details)' 중 이 챕터와 맥락이 닿는 '이름', '장소', '조직', '사건'을 **최소 2~3개 이상** 본문에 구체적으로 포함하세요.
- "친구들과 바다를 갔다" 대신 "철수와 기차를 타고 강릉 바다를 보러 갔다"처럼 사실 기반의 구체적 명사를 우선하세요. (단, 없는 사실을 새로 지어내지 말 것)

[출력 형식 제한 (JSON)]
다음 형태의 JSON을 반환해야 합니다:
{{
  "content": "마크다운 없이 작성된 순수 본문 텍스트 (단락은 \\n\\n 로 구분). 연결 문장도 이 본문 마지막에 포함되어야 함.",
  "quote": "이 챕터 본연의 감정과 핵심 메시지를 관통하는 1~2줄의 짧고 인상적인 문장 (연결 문장을 여기에 쓰지 마세요)"
}}
"""
        next_chap_str = f"- 다음 챕터 제목: {next_chapter.chapter_title}\n- 다음 챕터 주제: {CHAPTER_ROLES.get(next_chapter.chapter_type, dict()).get('theme', '')}" if next_chapter else "- 마지막 챕터입니다. (다음 챕터로 연결하는 transition 불필요, 깊은 여운으로 마무리)"


        
        user_prompt = f"""[현재 챕터 정보]
- 번호/유형: Chapter {chapter.chapter_num} ({chapter.chapter_type})
- 제목: {chapter.chapter_title}

[다음 챕터 예고 (Transition 연결용)]
{next_chap_str}

[전체 고유명사 풀 (Personal Details)]
{json.dumps(personal_details, ensure_ascii=False, indent=2)}

[이 챕터의 Scene 구조]
{json.dumps(scenes_json, ensure_ascii=False, indent=2)}

[관련 추가 문맥 데이터]
{additional_context}

위 지침을 준수하여 이 챕터의 'content'와 'quote'를 JSON으로 작성해 주세요.
"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={ "type": "json_object" }
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error generating chapter {chapter.chapter_num}: {e}")
            return {"content": "내용을 생성하는 중 오류가 발생했습니다.", "quote": ""}

autobiography_service = AutobiographyService()

