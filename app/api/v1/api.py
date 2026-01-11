from fastapi import APIRouter
from app.api.v1.endpoints import classification, generation, rag

api_router = APIRouter()

# 분류 (Classification)
api_router.include_router(classification.router, prefix="/classification", tags=["classification"])

# 생성 (Generation)
api_router.include_router(generation.router, prefix="/generation", tags=["generation"])

# 검색 증강 생성 (RAG)
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
