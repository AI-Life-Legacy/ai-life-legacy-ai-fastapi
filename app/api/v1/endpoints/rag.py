from fastapi import APIRouter, HTTPException
from app.schemas.rag import RagSyncRequest, RagSearchRequest, RagSearchResponse, RagSearchResult
from app.services.vector_store import add_document, search_context

router = APIRouter()

@router.post("/sync")
def sync_rag_data(request: RagSyncRequest):
    try:
        add_document(request.userId, request.text, request.metadata)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=RagSearchResponse)
def search_rag_context(request: RagSearchRequest):
    try:
        results = search_context(request.userId, request.query)
        
        # LangChain 반환값: List[(Document, score)]
        # 결과 리스트 초기화
        response_items = []
        for doc, score in results:
            response_items.append(RagSearchResult(text=doc.page_content, score=score))
            
        return RagSearchResponse(results=response_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
