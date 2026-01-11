import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings
from typing import List, Dict

# 영구 클라이언트 (Persistent Client)
chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=settings.OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name="life_legacy",
    embedding_function=openai_ef
)

def add_document(user_id: str, text: str, metadata: Dict):
    # ID 생성을 위해 sourceId가 문자열인지 확인
    source_id = metadata.get('sourceId') or metadata.get('source_id') or 'unknown'
    doc_id = f"{user_id}_{source_id}"
    
    collection.add(
        documents=[text],
        metadatas=[{**metadata, "user_id": user_id}],
        ids=[doc_id]
    )

def search_context(user_id: str, query: str, n_results: int = 3) -> Dict:
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"user_id": user_id} 
    )
