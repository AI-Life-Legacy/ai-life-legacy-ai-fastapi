from fastapi import APIRouter, HTTPException
from app.schemas.generation import (
    QuestionRequest, QuestionResponse, 
    AutobiographyRequest, AutobiographyResponse
)
from app.services.openai_service import generate_follow_up_question
from app.services.autobiography_service import autobiography_service
from app.services.pdf_service import pdf_service
from app.core.config import BASE_DIR
from pathlib import Path
import os

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
        # 1. Generate Markdown content using RAG service
        md_content = await autobiography_service.generate_autobiography_memoir(request.userId, request.userName)
        
        # 2. Save MarkDown to storage
        storage_path = Path(BASE_DIR) / "storage" / "data"
        os.makedirs(storage_path, exist_ok=True)
        
        md_filename = f"autobiography_{request.userId}.md"
        md_file_path = storage_path / md_filename
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
        # 3. Generate PDF from MarkDown
        pdf_filename = f"autobiography_{request.userId}.premium.pdf"
        pdf_file_path = storage_path / pdf_filename
        _, page_count = pdf_service.generate_premium_pdf(md_content, str(pdf_file_path))
        
        # 4. Save a copy to System's Downloads folder for convenience
        # (This is internal for local development/testing)
        try:
            downloads_path = Path.home() / "Downloads"
            if downloads_path.exists():
                import shutil
                system_pdf_path = downloads_path / pdf_filename
                shutil.copy2(pdf_file_path, system_pdf_path)
        except Exception as e:
            print(f"Warning: Failed to copy PDF: {e}")
        
        # 5. Return status and URL (For now, returning local path as URL if no S3)
        # In production, this would be an S3 URL.
        pdf_url = f"http://localhost:8000/storage/data/{pdf_filename}"
        
        return AutobiographyResponse(
            status="COMPLETED",
            pdf_url=pdf_url,
            page_count=page_count
        )
    except Exception as e:
        print(f"Error in autobiography generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
