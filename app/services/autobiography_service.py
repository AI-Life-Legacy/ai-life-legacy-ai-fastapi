from openai import AsyncOpenAI
from app.core.config import settings
from app.services.vector_store import retrieve_all_user_contexts, retrieve_chapter_contexts
from app.services.timeline_service import timeline_service
from app.services.scene_builder import scene_builder
import asyncio
import os
import json
import hashlib
import base64

CHAPTER_ROLES = {
    "childhood": {
        "theme": "성장 배경, 가족 분위기, 최초의 기억 형성",
        "avoid": "정치적 이슈, 직업과 커리어의 자세한 이야기",
        "tone": "따뜻함, 호기심, 순수함",
        "role": "자서전의 시작, 가치관의 뿌리 제시"
    },
    "school": {
        "theme": "친구 관계, 정체성 탐색, 작은 도전과 배움",
        "avoid": "지나치게 무거운 고난, 가정사 중심 서술",
        "tone": "생기, 떨림, 변화",
        "role": "자아 형성과 관계의 확장"
    },
    "youth": {
        "theme": "진로 선택, 현실 진입, 첫 사회생활과 적응",
        "avoid": "노년 회고나 가족 마무리 중심 주제",
        "tone": "열정, 고민, 긴장감",
        "role": "성인으로서의 첫 발돋움과 홀로서기"
    },
    "marriage": {
        "theme": "배우자와의 만남, 관계의 확장, 책임과 안정",
        "avoid": "가족 밖의 지식적 업무 에피소드 집중",
        "tone": "사랑, 신뢰, 성숙함",
        "role": "개인에서 공동체로 확장되는 시기"
    },
    "career": {
        "theme": "사회적 역할, 직장에서의 갈등과 위기, 실패와 성취",
        "avoid": "지나치게 평온한 일상 위주 서술",
        "tone": "치열함, 결단, 성취감",
        "role": "인생의 완성기와 극복의 서사"
    },
    "hobby": {
        "theme": "일상의 균형, 자아 발견, 취미와 회복",
        "avoid": "과도한 업무 스트레스 묘사",
        "tone": "자유, 즐거움, 소소한 행복",
        "role": "삶의 숨 고르기와 개인의 내면 탐구"
    },
    "self_reflection": {
        "theme": "성찰, 가치관의 변화, 삶과 건강에 대한 태도",
        "avoid": "단순한 사건 나열",
        "tone": "차분함, 회고, 진심",
        "role": "인생의 깊이와 성숙한 생각의 표현"
    },
    "family": {
        "theme": "앞으로의 당부, 남기고 싶은 말, 인생 철학, 가족의 미래",
        "avoid": "과거 사건 중심의 긴 회상",
        "tone": "따뜻함, 감사, 담담함",
        "role": "자서전의 여운 있는 마무리"
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

    keyword_map = {
        "childhood": ["유년", "어린 시절", "태어난", "고향", "부모", "아버지", "어머니", "형제", "자매"],
        "school": ["학교", "학창", "선생님", "친구", "공부", "사춘기", "중학교", "고등학교", "초등학교"],
        "youth": ["대학", "20대", "청년", "첫 직장", "군대", "진로", "전공", "취업"],
        "marriage": ["결혼", "배우자", "남편", "아내", "연애", "신혼", "출산", "아이", "자식"],
        "career": ["직장", "회사", "업무", "성취", "도전", "실패", "동료", "상사", "사업"],
        "hobby": ["취미", "여가", "주말", "운동", "그림", "음악", "여행", "음식", "좋아하는"],
        "self_reflection": ["건강", "나이", "깨달음", "가치관", "인생", "태도", "후회", "보람", "성찰"],
        "family": ["미래", "계획", "자녀", "손주", "가족", "남기고", "당부", "꿈", "철학"]
    }

    for item in answers:
        item_text = ""
        item_chapter = None

        if isinstance(item, dict):
            # toc_id ?깆쓣 ?듯븳 留ㅼ묶 (1 -> childhood, 2 -> school ??
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
        system_prompt = """?뱀떊? ?몃Ъ, ?μ냼, ?ш굔 ?깆쓽 怨좎쑀 ?뺣낫瑜??뺥솗?섍쾶 異붿텧?섎뒗 ?곗씠??遺꾩꽍媛?낅땲??
?쒓났???명꽣酉?臾몃㎘ ?곗씠?곗뿉???ㅼ쓬 ?좏삎??怨좎쑀 ?붿냼瑜?鍮좎쭚?놁씠 異붿텧?섏뿬 JSON ?뺤떇?쇰줈 諛섑솚?섏꽭??
?ㅼ쭅 ?띿뒪?몄뿉 ?깆옣?섎뒗 ?ъ떎留?異붿텧?댁빞 ?섎ŉ, ?덈? 吏?대궡吏 留덉꽭??
{
  "people": ["?щ엺 ?대쫫, 媛議?愿怨??⑥쐞??吏곸콉 ??],
  "places": ["?μ냼, 吏?? 嫄대Ъ紐???],
  "activities": ["痍⑤?, ?밴린, 諛섎났 ?쒕룞 ??],
  "achievements": ["?섏긽 ?댁뿭, ?먭꺽, ?깆랬 ??],
  "events": ["?뱀젙 ?ш굔紐? ?ы뻾, ?꾧린 ?쒓컙 ??]
}
"""
        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_EXTRACT_MODEL,
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

    async def _generate_dalle_illustration(self, user_id: str, chapter) -> str | None:
        """
        Generate a safe memoir-style illustration for a chapter and store it on the AI server.
        """
        scene_summaries = []
        for s in chapter.scenes:
            scene_summaries.append(f"- {s.title}: {s.setting or ''} {s.resolution or ''}")
        scenes_text = "\n".join(scene_summaries)

        prompt = f"""Create a warm memoir illustration for a Korean autobiography chapter titled "{chapter.chapter_title}".

Chapter context:
{scenes_text}

Style and safety requirements:
- Make it look like an editorial watercolor / soft film-tone illustration, not a real photograph.
- Do not depict a specific identifiable real person or celebrity.
- Prefer places, objects, light, seasons, rooms, streets, desks, letters, albums, and symbolic family atmosphere.
- No readable text, no logos, no signatures, no watermarks.
- High quality, calm, nostalgic, suitable for a printed life-story PDF."""

        try:
            print(f"[Image] Generating dynamic illustration for Chapter {chapter.chapter_num}...")
            response = await self.client.images.generate(
                model=settings.OPENAI_IMAGE_MODEL,
                prompt=prompt,
                n=1,
            )
            image_data = response.data[0]

            import httpx
            from pathlib import Path
            from PIL import Image, ImageEnhance

            if getattr(image_data, "b64_json", None):
                image_bytes = base64.b64decode(image_data.b64_json)
            elif getattr(image_data, "url", None):
                async with httpx.AsyncClient() as http_client:
                    img_resp = await http_client.get(image_data.url)
                    img_resp.raise_for_status()
                    image_bytes = img_resp.content
            else:
                raise ValueError("Image generation returned no image data")

            img_dir = Path(settings.CHROMA_DB_PATH).parent / "assets" / "generated_illustrations"
            os.makedirs(img_dir, exist_ok=True)
            safe_user_id = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:12]
            img_filename = f"illustration_{safe_user_id}_{chapter.chapter_num}.jpg"
            img_path = img_dir / img_filename

            raw_path = img_dir / f"raw_{safe_user_id}_{chapter.chapter_num}.png"
            with open(raw_path, "wb") as f:
                f.write(image_bytes)

            with Image.open(raw_path) as image:
                resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                processed = image.convert("RGB").resize((1600, 1100), resample_filter)
                processed = ImageEnhance.Contrast(processed).enhance(1.04)
                processed = ImageEnhance.Sharpness(processed).enhance(1.08)
                processed.save(img_path, "JPEG", quality=88, optimize=True)

            try:
                raw_path.unlink()
            except OSError:
                pass

            local_uri = f"file:///{img_path.as_posix()}"
            print(f"[Image] Dynamic illustration saved to: {local_uri}")
            return local_uri
        except Exception as e:
            print(f"Warning: image generation failed for Chapter {chapter.chapter_num}: {e}")
        return None

    async def generate_autobiography_memoir(
        self,
        user_id: str,
        user_name: str,
        retrieved_context: str = None,
        answers: list = None,
        theme: str = "classic",
        generate_illustrations: bool = False,
        personalization: dict | None = None,
    ) -> str:
        """
        Timeline Graph? Scene Composition??嫄곗퀜 ?쒖궗瑜??앹꽦?⑸땲??
        """
        if retrieved_context is None:
            print(f"[{user_name}] 1. ?꾩껜 臾몃㎘ 寃??以?..")
            # 1. 踰≫꽣 ?곗씠?곕쿋?댁뒪?먯꽌 ?꾩껜 而⑦뀓?ㅽ듃 寃??
            if answers:
                from app.api.v1.endpoints.generation import extract_context_from_answers
                retrieved_context = extract_context_from_answers(answers)
            else:
                retrieved_context = await retrieve_all_user_contexts(user_id=user_id, limit=30)

        if not retrieved_context:
            return "寃?됰맂 ?ъ슜???곗씠?곌? ?놁뒿?덈떎. ?먯꽌?꾩쓣 ?앹꽦?????놁뒿?덈떎."


        print(f"[{user_name}] 2. Personal Detail & Timeline 異붿텧 以?..")
        # 2-1. 怨쇨굅 RAG 湲곗뼲???꾩껜 濡쒕뱶?섏뿬 ?꾩옱 ?듬? 而⑦뀓?ㅽ듃? ?듯빀
        past_full_memory = await retrieve_all_user_contexts(user_id=user_id, limit=30)

        combined_context_parts = []
        if retrieved_context:
            combined_context_parts.append(f"[?꾩옱 ?듬? 湲곕줉]\n{retrieved_context}")
        if past_full_memory:
            combined_context_parts.append(f"[怨쇨굅 RAG 湲곗뼲]\n{past_full_memory}")

        personalization = personalization or {}
        personalization_context = self._build_personalization_context(personalization)
        if personalization_context:
            combined_context_parts.insert(0, personalization_context)

        full_timeline_context = "\n\n".join(combined_context_parts)
        if not full_timeline_context:
            full_timeline_context = retrieved_context

        # 2-2. ?듯빀??留λ씫?먯꽌 怨좎쑀紐낆궗(?뷀뀒?? 異붿텧 諛???꾨씪??蹂듭썝
        personal_details = await self._extract_personal_details(full_timeline_context)
        timeline_events = await timeline_service.reconstruct_timeline(full_timeline_context)

        print(f"[{user_name}] 3. Scene ?⑥쐞 梨뺥꽣 援ъ“??以?..")
        # Scene structure is rebuilt from the reconstructed timeline.
        chapter_data_list = await scene_builder.build_full_story_structure(timeline_events)
        self._apply_personalized_chapter_titles(chapter_data_list, personalization)

        # 媛?梨뺥꽣蹂꾨줈 蹂몃Ц ?앹꽦
        generated_chapters = []

        grouped_answers = group_answers_by_chapter(answers) if answers else {}

        for i, chapter in enumerate(chapter_data_list):
            next_chapter = chapter_data_list[i+1] if i + 1 < len(chapter_data_list) else None
            print(f"[{user_name}] 4. 梨뺥꽣 ?앹꽦 以? {chapter.chapter_num}. {chapter.chapter_title}")

            # 1. ??梨뺥꽣???뱁솕??怨쇨굅 湲곕줉(RAG) 寃??
            past_rag_context = await retrieve_chapter_contexts(user_id, chapter.chapter_type, limit=10)

            # 2. ?꾩옱 ?명꽣酉??몄뀡?먯꽌 ?살? ?듬? 痍⑦빀
            current_chapter_answers = ""
            if answers:
                current_chapter_answers = "\n\n".join(grouped_answers.get(chapter.chapter_type, []))

            # 3. ?꾩옱 ?듬?怨?怨쇨굅 RAG 而⑦뀓?ㅽ듃 ?듯빀 (Hybrid)
            chapter_context_parts = []

            if current_chapter_answers:
                chapter_context_parts.append(f"[?꾩옱 ?명꽣酉??듬?]\n{current_chapter_answers}")

            if past_rag_context:
                chapter_context_parts.append(f"[怨쇨굅 湲곕줉 (RAG)]\n{past_rag_context}")

            chapter_context = "\n\n".join(chapter_context_parts)

            # 4. 諛⑹뼱 肄붾뱶: 留뚯빟 ?대떦 梨뺥꽣??????꾩옱/怨쇨굅 ?곗씠?곌? 紐⑤몢 ?녿떎硫??꾩껜 而⑦뀓?ㅽ듃(retrieved_context) ?ъ슜
            if not chapter_context and retrieved_context:
                chapter_context = retrieved_context

            # Post-check ?⑹씠?깆쓣 ?꾪빐 Retry 濡쒖쭅 ?섑븨 媛??(?꾩옱???⑥씪 ?⑥뒪)
            chapter_result = await self._generate_chapter_text(
                user_name,
                chapter,
                next_chapter,
                chapter_context,
                personal_details,
                personalization,
            )

            # (?좏깮) Post-check 濡쒖쭅: 誘몃옒 梨뺥꽣??怨쇨굅 ?⑥뼱 ?덈Т 留롮쑝硫??ъ떆????..

            chapter_text = chapter_result.get("content", "")
            chapter_quote = chapter_result.get("quote", "")

            chapter.generated_text = chapter_text

            # DALL-E ?대?吏 ?앹꽦 泥섎━
            generated_chapters.append((chapter, chapter_text, chapter_quote))

        # Chapter text depends on the previous chapter, but illustrations do
        # not. Generate only illustrations concurrently with a conservative
        # limit so the image API is not flooded.
        illustration_uris = [None] * len(generated_chapters)
        if generate_illustrations:
            illustration_semaphore = asyncio.Semaphore(2)

            async def generate_illustration(index: int, chapter):
                async with illustration_semaphore:
                    illustration_uris[index] = await self._generate_dalle_illustration(
                        user_id,
                        chapter,
                    )

            await asyncio.gather(*(
                generate_illustration(index, chapter)
                for index, (chapter, _, _) in enumerate(generated_chapters)
            ))

        full_markdown = f"# {user_name}의 자서전\n\n"
        for index, (chapter, chapter_text, chapter_quote) in enumerate(generated_chapters):
            full_markdown += f"## {chapter.chapter_title}\n"
            full_markdown += f"<!-- MOOD: {chapter.mood} -->\n"
            if illustration_uris[index]:
                full_markdown += f"<!-- IMAGE: {illustration_uris[index]} -->\n"
            full_markdown += f"{chapter_text}\n\n"
            if chapter_quote:
                full_markdown += f"<!-- QUOTE: {chapter_quote} -->\n\n"

        return full_markdown

    def _build_personalization_context(self, personalization: dict) -> str:
        if not personalization:
            return ""

        toc_plan = personalization.get("tocPlan") or []
        purposes = personalization.get("purposes") or []
        feedback = personalization.get("feedback") or {}
        lines = [
            "[Autobiography personalization]",
            f"- name: {personalization.get('name') or ''}",
            f"- age: {personalization.get('age') or ''}",
            f"- life_stage: {personalization.get('lifeStage') or ''}",
            f"- purposes: {', '.join(purposes) if isinstance(purposes, list) else purposes}",
            f"- output_style: {personalization.get('style') or ''}",
            f"- output_style_id: {personalization.get('styleId') or ''}",
        ]
        if feedback:
            feedback_tags = feedback.get("tags") or []
            lines.extend(
                [
                    "- previous_result_feedback:",
                    f"  rating: {feedback.get('rating') or ''}",
                    f"  tags: {', '.join(feedback_tags) if isinstance(feedback_tags, list) else feedback_tags}",
                    f"  comment: {feedback.get('comment') or ''}",
                    "  instruction: Improve the new autobiography by addressing this feedback without explicitly mentioning the rating.",
                ]
            )
        if toc_plan:
            lines.append("- recommended_toc:")
            lines.extend([f"  {idx + 1}. {title}" for idx, title in enumerate(toc_plan)])
        return "\n".join(lines)

    def _apply_personalized_chapter_titles(self, chapters: list, personalization: dict):
        toc_plan = personalization.get("tocPlan") or []
        if not toc_plan:
            return

        for idx, chapter in enumerate(chapters):
            if idx < len(toc_plan):
                chapter.chapter_title = toc_plan[idx]

    async def _generate_chapter_text(
        self,
        user_name: str,
        chapter,
        next_chapter,
        additional_context: str,
        personal_details: dict,
        personalization: dict | None = None,
    ) -> dict:
        """
        援ъ“?붾맂 Scene ?뺣낫瑜?湲곕컲?쇰줈 梨뺥꽣 ?띿뒪?몄? ?먯꽱??Quote瑜??앹꽦?⑸땲??
        """
        scenes_json = [s.model_dump() for s in chapter.scenes]
        role_info = CHAPTER_ROLES.get(chapter.chapter_type, CHAPTER_ROLES["family"])
        personalization_context = self._build_personalization_context(personalization or {})
        style = (personalization or {}).get("style") or "detailed"
        style_id = (personalization or {}).get("styleId") or "detailed"
        length_rule = "Write this chapter in 4 to 6 rich paragraphs."
        if style_id == "simple":
            length_rule = "Write this chapter in 2 to 3 concise paragraphs."
        elif style_id == "literary":
            length_rule = "Write this chapter like a literary essay with scene transitions and polished rhythm."
        elif style_id == "calm":
            length_rule = "Write this chapter calmly, emphasizing facts, choices, and changes without exaggeration."
        elif style_id == "warm":
            length_rule = "Write this chapter warmly, emphasizing people, relationships, gratitude, and memory."

        system_prompt = f"""?뱀떊? ???щ엺???앹븷瑜?源딆씠 ?덈뒗 ?쒖궗濡??쒗쁽?섎뒗 踰좏뀒???먯꽌???묎??낅땲??
二쇱뼱吏?Scene 援ъ“? 愿??臾몃㎘???대젮 1媛쒖쓽 梨뺥꽣瑜??묒꽦?섏꽭??

[媛쒖씤???묒꽦 湲곗?]
{personalization_context}
- 寃곌낵臾??ㅽ???吏移? {style}
- 遺꾨웾 吏移? {length_rule}
- 異붿쿇 紐⑹감? ?꾩옱 梨뺥꽣 ?쒕ぉ???곗꽑 諛섏쁺?섍퀬, 湲곗〈 怨좎젙 紐⑹감泥섎읆 蹂댁씠吏 ?딄쾶 ?묒꽦?섏꽭??

[梨뺥꽣 ??븷 (Chapter Role)]
- ???μ쓽 ??븷: {role_info['role']}
- ?ㅻ쨪????二쇱젣: {role_info['theme']}
- ?뺤꽌 ?? {role_info['tone']}
- ?덉슜???앹븷 二쇨린: {chapter.chapter_type}??留욌뒗 ?댁빞湲곕쭔 吏묒쨷?섍퀬 ?ㅻⅨ ?앹븷 ?댁빞湲곕줈 湲멸쾶 ?덉? 留덉꽭??
- ?쇳빐?????쒖닠: {role_info['avoid']}

[?쒖궗 援ъ“ ?쒖빟 (Narrative Arc Rule) 諛??곌껐 臾몄옣(Transition)]
1. ?⑥닚 ?ш굔 ?섏뿴 湲덉?: 臾몃떒 援ъ꽦 ??媛湲됱쟻 [?곹솴/諛곌꼍 ??媛덈벑/?좏깮 ??蹂??寃곌낵 ???섎? ?깆같]???먮쫫??諛섏쁺?섏꽭??
2. 紐⑤뱺 臾몃떒???듭? 援먰썕?쇰줈 ?앸궡吏 留덉꽭?? ?먯뿰?ㅻ윭???ъ슫???④린?몄슂.
3. ?댁쟾 Scene怨??ㅼ쓬 Scene??臾??먮Ⅴ???댁뼱吏?꾨줉 ?쒓컙 寃쎄낵???대㈃??蹂?붾? ?섑??대뒗 遺?쒕윭???꾪솚(Transition)???ъ슜?섏꽭??
4. [留ㅼ슦 以묒슂] 蹂몃Ц??留덉?留?臾몃떒 ?앹뿉??諛섎뱶???ㅼ쓬 梨뺥꽣濡??먯뿰?ㅻ읇寃??섏뼱媛??1~2臾몄옣??'?곌껐 臾몄옣(Transition Sentence)'???묒꽦?섏꽭??
   - ?? 留덉?留?梨뺥꽣??寃쎌슦???쒖쇅?⑸땲??
   - "?ㅼ쓬 ?μ뿉?쒕뒗 ~??????댁빞湲고븯寃좊떎" ?앹쓽 吏곸꽕?곸씤 ?쒗쁽 ??? ?꾩옱 梨뺥꽣??寃쏀뿕???대뼸寃??ㅼ쓬 梨뺥꽣??諛묎굅由꾩씠 ?섏뿀?붿? ?뚯꽕泥섎읆 遺?쒕읇寃??붿떆?섏꽭??

[?뷀뀒??利앺룺 ?쒖빟 (Personal Detail Amplifier)]
- ?쒓났??'怨좎쑀紐낆궗 由ъ뒪??Personal Details)' 以???梨뺥꽣? 留λ씫???용뒗 '?대쫫', '?μ냼', '議곗쭅', '?ш굔'??**理쒖냼 2~3媛??댁긽** 蹂몃Ц??援ъ껜?곸쑝濡??ы븿?섏꽭??
- "移쒓뎄?ㅺ낵 諛붾떎瑜?媛붾떎" ???"泥좎닔? 湲곗감瑜??怨?媛뺣쫱 諛붾떎瑜?蹂대윭 媛붾떎"泥섎읆 ?ъ떎 湲곕컲??援ъ껜??紐낆궗瑜??곗꽑?섏꽭?? (?? ?녿뒗 ?ъ떎???덈줈 吏?대궡吏 留?寃?

[異쒕젰 ?뺤떇 ?쒗븳 (JSON)]
?ㅼ쓬 ?뺥깭??JSON??諛섑솚?댁빞 ?⑸땲??
{{
  "content": "留덊겕?ㅼ슫 ?놁씠 ?묒꽦???쒖닔 蹂몃Ц ?띿뒪??(?⑤씫? \\n\\n 濡?援щ텇). ?곌껐 臾몄옣????蹂몃Ц 留덉?留됱뿉 ?ы븿?섏뼱????",
  "quote": "??梨뺥꽣 蹂몄뿰??媛먯젙怨??듭떖 硫붿떆吏瑜?愿?듯븯??1~2以꾩쓽 吏㏐퀬 ?몄긽?곸씤 臾몄옣 (?곌껐 臾몄옣???ш린???곗? 留덉꽭??"
}}
"""
        next_chap_str = f"- ?ㅼ쓬 梨뺥꽣 ?쒕ぉ: {next_chapter.chapter_title}\n- ?ㅼ쓬 梨뺥꽣 二쇱젣: {CHAPTER_ROLES.get(next_chapter.chapter_type, dict()).get('theme', '')}" if next_chapter else "- 留덉?留?梨뺥꽣?낅땲?? (?ㅼ쓬 梨뺥꽣濡??곌껐?섎뒗 transition 遺덊븘?? 源딆? ?ъ슫?쇰줈 留덈Т由?"



        user_prompt = f"""[?꾩옱 梨뺥꽣 ?뺣낫]
- 踰덊샇/?좏삎: Chapter {chapter.chapter_num} ({chapter.chapter_type})
- ?쒕ぉ: {chapter.chapter_title}

[?ㅼ쓬 梨뺥꽣 ?덇퀬 (Transition ?곌껐??]
{next_chap_str}

[?꾩껜 怨좎쑀紐낆궗 ? (Personal Details)]
{json.dumps(personal_details, ensure_ascii=False, indent=2)}

[??梨뺥꽣??Scene 援ъ“]
{json.dumps(scenes_json, ensure_ascii=False, indent=2)}

[愿??異붽? 臾몃㎘ ?곗씠??
{additional_context}

??吏移⑥쓣 以?섑븯????梨뺥꽣??'content'? 'quote'瑜?JSON?쇰줈 ?묒꽦??二쇱꽭??
"""

        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_AUTOBIOGRAPHY_MODEL,
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
            return {"content": "?댁슜???앹꽦?섎뒗 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎.", "quote": ""}

autobiography_service = AutobiographyService()

