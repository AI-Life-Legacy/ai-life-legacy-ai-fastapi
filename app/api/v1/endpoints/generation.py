from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from app.schemas.generation import (
    QuestionRequest, QuestionResponse,
    AutobiographyRequest, AutobiographyResponse
)
from app.services.openai_service import generate_follow_up_question
from app.services.autobiography_service import autobiography_service
from app.services.pdf_service import pdf_service
from app.services.vector_store import retrieve_all_user_contexts
from app.core.config import BASE_DIR, settings
from pathlib import Path
import os
import json
import hashlib
from openai import AuthenticationError, PermissionDeniedError, RateLimitError

router = APIRouter()

@router.post("/question", response_model=QuestionResponse)
async def create_follow_up_question(request: QuestionRequest):
    try:
        # 1. RAG 諛?怨쇨굅 ?댁뿭???쒖슜??瑗щ━ 吏덈Ц ?앹꽦 (user_id 濡쒖쭅 泥섎━)
        user_id = request.user_id or "unknown"
        question_text = await generate_follow_up_question(user_id, request.current_answer, request.chat_history)
        return QuestionResponse(question=question_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_context_from_answers(answers: List[Any]) -> str:
    extracted_texts = []
    for item in answers:
        if isinstance(item, str):
            extracted_texts.append(item)
        elif isinstance(item, dict):
            q_text = item.get("question_text") or item.get("questionText") or item.get("question")
            if isinstance(q_text, dict):
                q_text = q_text.get("question_text") or q_text.get("questionText") or q_text.get("title") or ""

            a_text = item.get("answer_text") or item.get("answerText") or item.get("text") or item.get("content") or item.get("answer") or ""

            if q_text and a_text:
                extracted_texts.append(f"Q: {q_text}\nA: {a_text}")
            elif a_text:
                extracted_texts.append(a_text)
            else:
                val_str = " ".join([str(v) for k, v in item.items() if isinstance(v, str)])
                if val_str:
                    extracted_texts.append(val_str)
        else:
            extracted_texts.append(str(item))

    return "\n\n".join(extracted_texts)

@router.post("/autobiography", response_model=AutobiographyResponse)
async def create_autobiography(request: AutobiographyRequest):
    try:
        user_id = request.user_id or request.userId
        user_name = request.user_name or request.userName or "사용자"

        if not user_id:
            raise HTTPException(status_code=400, detail="userId or user_id is required.")

        # Convert chapters to answers if present
        if request.chapters and not request.answers:
            flat_answers = []
            for ch in request.chapters:
                toc_id = ch.get("toc_id") or ch.get("tocId")
                questions = ch.get("questions") or []
                for q in questions:
                    flat_answers.append({
                        "toc_id": toc_id,
                        "question_id": q.get("question_id") or q.get("questionId"),
                        "question": q.get("question") or q.get("questionText") or q.get("question_text"),
                        "answer": q.get("answer") or q.get("answerText") or q.get("answer_text")
                    })
            request.answers = flat_answers

        if not request.answers:
            raise HTTPException(
                status_code=400,
                detail="?ъ슜???듬? ?곗씠??answers)媛 ?녾굅??鍮꾩뼱 ?덉뒿?덈떎. ?먯꽌?꾩쓣 ?앹꽦?????놁뒿?덈떎."
            )

        # 1. Generate Content Hash based on request answers
        serialized_answers = json.dumps(
            {
                "answers": request.answers,
                "personalization": request.personalization or {},
                "theme": request.theme,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        content_hash = hashlib.sha256(serialized_answers.encode("utf-8")).hexdigest()

        cache_dir = Path(BASE_DIR) / ".cache" / "autobiography"
        cache_file = cache_dir / f"{content_hash}.json"

        pdf_filename = f"autobiography_{user_id}_{content_hash}.pdf"
        pdf_file_path = Path(BASE_DIR) / "generated_pdfs" / pdf_filename

        # 2. Check Cache
        if not request.force and cache_file.exists() and pdf_file_path.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # 濡쒓렇 異쒕젰 ?붽뎄?ы빆:
                # - request answers count
                # - content_hash
                # - cache hit/miss
                # - output pdf path
                print(f"[LOG] request answers count: {len(request.answers)}")
                print(f"[LOG] content_hash: {content_hash}")
                print(f"[LOG] cache hit/miss: cache hit")
                print(f"[LOG] output pdf path: {pdf_file_path}")

                return AutobiographyResponse(
                    status="COMPLETED",
                    pdf_url=cache_data["pdf_url"],
                    page_count=cache_data["page_count"],
                    cached=True,
                    markdown=cache_data.get("markdown"),
                    markdown_url=cache_data.get("markdown_url")
                )
            except Exception as ce:
                print(f"Warning: Failed to read cache: {ce}. Proceeding to regenerate.")

        # 濡쒓렇 異쒕젰 ?붽뎄?ы빆 (cache miss)
        print(f"[LOG] request answers count: {len(request.answers)}")
        print(f"[LOG] content_hash: {content_hash}")
        print(f"[LOG] cache hit/miss: cache miss")

        # 3. Extract context from request answers
        retrieved_context = extract_context_from_answers(request.answers)

        # 4. Generate Markdown content using request answers (passing theme & generate_illustrations)
        md_content = await autobiography_service.generate_autobiography_memoir(
            user_id, user_name, retrieved_context=retrieved_context, answers=request.answers,
            theme=request.theme, generate_illustrations=request.generate_illustrations,
            personalization=request.personalization
        )

        # 5. Save MarkDown to storage
        storage_path = Path(BASE_DIR) / "storage" / "data"
        os.makedirs(storage_path, exist_ok=True)
        md_filename = f"autobiography_{user_id}_{content_hash}.md"
        md_file_path = storage_path / md_filename
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 6. Generate PDF from MarkDown in generated_pdfs with selected theme
        _, page_count = pdf_service.generate_premium_pdf(md_content, str(pdf_file_path), theme=request.theme)

        # 濡쒓렇 異쒕젰 ?붽뎄?ы빆 (pdf path)
        print(f"[LOG] output pdf path: {pdf_file_path}")

        # 7. Save a copy to System's Downloads folder for convenience
        try:
            downloads_path = Path.home() / "Downloads"
            if downloads_path.exists():
                import shutil
                system_pdf_path = downloads_path / pdf_filename
                shutil.copy2(pdf_file_path, system_pdf_path)
        except Exception as e:
            print(f"Warning: Failed to copy PDF to Downloads: {e}")

        # 8. Define URL
        pdf_url = f"{settings.AI_SERVER_PUBLIC_URL.rstrip('/')}/generated-pdfs/{pdf_filename}"
        markdown_url = f"{settings.AI_SERVER_PUBLIC_URL.rstrip('/')}/storage/data/{md_filename}"

        # 9. Save Cache
        try:
            cache_data = {
                "pdf_url": pdf_url,
                "page_count": page_count,
                "markdown": md_content,
                "markdown_url": markdown_url
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as ce:
            print(f"Warning: Failed to save cache: {ce}")

        return AutobiographyResponse(
            status="COMPLETED",
            pdf_url=pdf_url,
            page_count=page_count,
            cached=False,
            markdown=md_content,
            markdown_url=markdown_url
        )

    except Exception as e:
        err_msg = str(e).lower()

        # Check specific OpenAI Exception names or message substring
        if isinstance(e, AuthenticationError) or "authentication" in err_msg or "apikey" in err_msg or "api_key" in err_msg:
            print(f"[ERROR] OpenAI Authentication Error: {e}")
            raise HTTPException(status_code=401, detail=f"OpenAI Authentication Error: {str(e)}")

        elif isinstance(e, PermissionDeniedError) or "permission" in err_msg or "model_not_found" in err_msg or "access" in err_msg:
            print(f"[ERROR] OpenAI Permission Denied / Model Access Error: {e}")
            raise HTTPException(status_code=403, detail=f"OpenAI Permission/Access Error: {str(e)}")

        elif isinstance(e, RateLimitError) or "quota" in err_msg or "rate_limit" in err_msg or "rate limit" in err_msg:
            print(f"[ERROR] OpenAI Rate Limit / Quota Exceeded Error: {e}")
            raise HTTPException(status_code=429, detail=f"OpenAI Rate Limit / Quota Error: {str(e)}")

        print(f"[ERROR] Error in autobiography generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

