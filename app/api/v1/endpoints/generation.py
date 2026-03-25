from fastapi import APIRouter, HTTPException
from app.schemas.generation import (
    QuestionRequest, QuestionResponse, 
    AutobiographyRequest, AutobiographyResponse
)
from app.services.openai_service import generate_follow_up_question
from app.services.autobiography_service import autobiography_service
from app.services.pdf_service import pdf_service
from app.core.config import settings, BASE_DIR
from pathlib import Path
import os

router = APIRouter()

@router.post("/question", response_model=QuestionResponse)
async def create_follow_up_question(request: QuestionRequest):
    try:
        question_text = await generate_follow_up_question(request.userId, request.originalQuestion, request.userAnswer)
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
        pdf_service.generate_premium_pdf(md_content, str(pdf_file_path))
        
        # 4. Save a copy to System's Downloads folder for convenience
        try:
            downloads_path = Path.home() / "Downloads"
            if downloads_path.exists():
                import shutil
                system_pdf_path = downloads_path / pdf_filename
                shutil.copy2(pdf_file_path, system_pdf_path)
                print(f"PDF copy saved to Downloads: {system_pdf_path}")
        except Exception as e:
            print(f"Warning: Failed to copy PDF to Downloads folder: {e}")
        
        return AutobiographyResponse(
            status="success",
            mdPath=str(md_file_path),
            pdfPath=str(pdf_file_path)
        )
    except Exception as e:
        print(f"Error in autobiography generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
