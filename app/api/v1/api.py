from fastapi import APIRouter
from app.api.v1.endpoints import classification, generation, rag, chat

api_router = APIRouter()

# 분류 (Classification)
api_router.include_router(classification.router, tags=["classification"])

# 생성 (Generation)
api_router.include_router(generation.router, tags=["generation"])

# 검색 증강 생성 (RAG)
api_router.include_router(rag.router, tags=["rag"])

# 아바타 채팅 (Avatar Chat)
api_router.include_router(chat.router, tags=["chat"])
