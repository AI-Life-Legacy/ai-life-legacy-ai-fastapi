import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import settings
from typing import List, Dict, Tuple

# 임베딩 모델 설정
openai_ef = OpenAIEmbeddings(
    openai_api_key=settings.OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# 벡터 스토어 초기화
vector_store = Chroma(
    collection_name="life_legacy",
    embedding_function=openai_ef,
    persist_directory=settings.CHROMA_DB_PATH
)

# 텍스트 스플리터 설정 (RAG 품질 향상을 위해 추가)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

def add_document(user_id: str, text: str, metadata: Dict):
    """
    문서를 청크로 분할하여 벡터 스토어에 저장합니다.
    """
    # 1. 텍스트 분할
    texts = text_splitter.split_text(text)
    
    # 2. 메타데이터 구성
    source_id = metadata.get('sourceId') or metadata.get('source_id') or 'unknown'
    
    # 각 청크별 Document 객체 생성
    documents = []
    for i, chunk_text in enumerate(texts):
        # 청크별 고유 ID 생성 (선택사항, Chroma가 자동 생성하게 할 수도 있음)
        # 여기서는 자동 생성을 맡기거나, 필요하다면 메타데이터만 추가
        chunk_metadata = {
            **metadata,
            "user_id": user_id,
            "chunk_index": i,
            "source_id": source_id
        }
        documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    
    # 3. 벡터 스토어에 추가
    vector_store.add_documents(documents)

def search_context(user_id: str, query: str, n_results: int = 3) -> List[Tuple[Document, float]]:
    """
    유사도 검색을 수행합니다.
    반환값: List[(Document, score)]
    """
    # user_id 필터링을 포함하여 검색
    return vector_store.similarity_search_with_score(
        query,
        k=n_results,
        filter={"user_id": user_id}
    )
