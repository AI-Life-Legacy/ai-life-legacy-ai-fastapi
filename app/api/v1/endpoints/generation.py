from fastapi import APIRouter, HTTPException
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
        # 1. RAG 및 과거 내역을 활용한 꼬리 질문 생성 (user_id는 임시로 고정하거나 로직에서 처리)
        # 실제로는 toc_id나 다른 정보를 통해 user_id를 가져와야 할 수도 있습니다.
        user_id = "test_user" 
        question_text = await generate_follow_up_question(user_id, request.current_answer, request.chat_history)
        return QuestionResponse(question=question_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/autobiography", response_model=AutobiographyResponse)
async def create_autobiography(request: AutobiographyRequest):
    try:
        # 1. Retrieve all user contexts from Vector DB
        retrieved_context = await retrieve_all_user_contexts(user_id=request.userId, limit=30)
        
        # 2. Generate content hash based on retrieved context (core input data)
        content_hash = hashlib.sha256(retrieved_context.encode("utf-8")).hexdigest()
        
        cache_dir = Path(BASE_DIR) / ".cache" / "autobiography"
        cache_file = cache_dir / f"{content_hash}.json"
        
        pdf_filename = f"autobiography_{request.userId}_{content_hash}.pdf"
        pdf_file_path = Path(BASE_DIR) / "generated_pdfs" / pdf_filename
        
        # 3. Check Cache
        if not request.force and cache_file.exists() and pdf_file_path.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                
                print(f"[CACHE HIT] Returning cached autobiography for user {request.userId} (hash: {content_hash})")
                return AutobiographyResponse(
                    status="COMPLETED",
                    pdf_url=cache_data["pdf_url"],
                    page_count=cache_data["page_count"],
                    cached=True
                )
            except Exception as ce:
                print(f"Warning: Failed to read cache: {ce}. Proceeding to regenerate.")
        
        print(f"[CACHE MISS] Generating new autobiography for user {request.userId} (hash: {content_hash})")
        
        # 4. Generate Markdown content using RAG service (pass retrieved_context)
        md_content = await autobiography_service.generate_autobiography_memoir(
            request.userId, request.userName, retrieved_context=retrieved_context
        )
        
        # 5. Save MarkDown to storage
        storage_path = Path(BASE_DIR) / "storage" / "data"
        os.makedirs(storage_path, exist_ok=True)
        md_filename = f"autobiography_{request.userId}_{content_hash}.md"
        md_file_path = storage_path / md_filename
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        # 6. Generate PDF from MarkDown in generated_pdfs
        _, page_count = pdf_service.generate_premium_pdf(md_content, str(pdf_file_path))
        
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
        
        # 9. Save Cache
        try:
            cache_data = {
                "pdf_url": pdf_url,
                "page_count": page_count
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as ce:
            print(f"Warning: Failed to save cache: {ce}")
            
        return AutobiographyResponse(
            status="COMPLETED",
            pdf_url=pdf_url,
            page_count=page_count,
            cached=False
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

