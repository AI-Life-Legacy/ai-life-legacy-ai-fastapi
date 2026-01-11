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
        # ChromaDB 쿼리 반환값: {'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
        
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0] # 거리 (낮을수록 좋음) 또는 유사도? 기본값은 L2 거리입니다.
        
        # 응답 모델로 변환
        response_items = []
        for i, doc in enumerate(documents):
            score = distances[i] if i < len(distances) else None
            response_items.append(RagSearchResult(text=doc, score=score))
            
        return RagSearchResponse(results=response_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
