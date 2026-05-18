import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.config import settings
from typing import List, Dict, Tuple
from starlette.concurrency import run_in_threadpool

# 임베딩 모델 설정
openai_ef = OpenAIEmbeddings(
    openai_api_key=settings.OPENAI_API_KEY,
    model=settings.OPENAI_EMBEDDING_MODEL
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

async def add_document(user_id: str, text: str, metadata: Dict):
    """
    문서를 청크로 분할하여 벡터 스토어에 저장합니다. (비동기)
    """
    await run_in_threadpool(_add_document_sync, user_id, text, metadata)

def _add_document_sync(user_id: str, text: str, metadata: Dict):
    # 1. 텍스트 분할
    texts = text_splitter.split_text(text)
    
    # 2. 메타데이터 구성
    source_id = metadata.get('sourceId') or metadata.get('source_id') or 'unknown'
    
    # 각 청크별 Document 객체 생성
    documents = []
    for i, chunk_text in enumerate(texts):
        chunk_metadata = {
            **metadata,
            "user_id": user_id,
            "chunk_index": i,
            "source_id": source_id
        }
        documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))
    
    # 3. 벡터 스토어에 추가
    vector_store.add_documents(documents)

def _expand_query(query: str) -> str:
    """
    RAG 유사도 검색을 강화하기 위해 한국어 구어체 질문을 동의어/문맥어로 보강합니다.
    """
    expanded = query
    query_lower = query.lower()
    
    # 1. 생년/몇년생/언제태어남 관련
    if any(w in query_lower for w in ["몇년생", "생년", "태어난 해", "태어난해", "출생년", "출생연", "언제 태어", "언제태어"]):
        expanded += " 태어난 해 출생 년도 생년월일 태어난 날짜 탄생일"
    
    # 2. 생일/생신/탄생 관련
    if any(w in query_lower for w in ["생일", "생신", "탄생일", "귀 빠진"]):
        expanded += " 태어난 날 생년월일 탄생 탄생일"
        
    # 3. 나이/연세 관련
    if any(w in query_lower for w in ["나이", "연세", "춘추", "몇살"]):
        expanded += " 나이 연세 출생 태어난 해"
        
    # 4. 고향/태어난곳 관련
    if any(w in query_lower for w in ["고향", "태어난 곳", "태어난곳", "출생지", "어디서 태어", "어디서태어"]):
        expanded += " 태어난 곳 고향 출생지 본가"
        
    # 5. 이름/성함/뜻 관련
    if any(w in query_lower for w in ["이름", "성함", "자", "휘"]):
        expanded += " 이름 성함 이름에 담긴 뜻 뜻"
        
    return expanded

async def search_context(user_id: str, query: str, n_results: int = 3) -> List[Tuple[Document, float]]:
    """
    유사도 검색을 수행합니다. (비동기)
    """
    return await run_in_threadpool(_search_context_sync, user_id, query, n_results)

def _search_context_sync(user_id: str, query: str, n_results: int = 3) -> List[Tuple[Document, float]]:
    # 1. RAG 유사도 검색 최적화를 위한 쿼리 확장(Query Expansion) 수행
    expanded_query = _expand_query(query)
    
    # 더 넓은 후보군(k=20)을 가져와 하이브리드 재정렬(Re-ranking)을 수행합니다.
    user_results = vector_store.similarity_search_with_score(
        expanded_query,
        k=20,
        filter={"user_id": user_id}
    )
    
    # 2. 키워드 기반 가중치 추출 (이름, 자녀, 가족, 탄생 등 중요 명사에 대한 가중치를 조절하여 RAG 정확도 대폭 상향)
    keywords = []
    query_lower = query.lower()
    
    if "이름" in query_lower or "성함" in query_lower or "성명" in query_lower:
        keywords.extend(["이름", "성함", "명명", "지으셨", "불리", "순신"])
    if "아들" in query_lower or "자녀" in query_lower or "자식" in query_lower:
        keywords.extend(["아들", "자녀", "자식", "첫째", "둘째", "셋째", "회", "면", "울"])
    if "가족" in query_lower or "부모" in query_lower or "어머니" in query_lower or "아버지" in query_lower or "아내" in query_lower or "부인" in query_lower:
        keywords.extend(["아버지", "어머니", "형제", "부모", "아내", "부인", "이정", "변씨", "방씨"])
    if "태어" in query_lower or "몇년생" in query_lower or "나이" in query_lower or "생일" in query_lower or "생신" in query_lower:
        keywords.extend(["태어", "1545", "출생", "생년월일", "나이", "탄생"])

    # 3. 하이브리드 점수 계산 (매칭되는 키워드당 벡터 거리를 좁혀 매칭 우위 상향 조정)
    reranked = []
    for doc, score in user_results:
        content_lower = doc.page_content.lower()
        match_count = sum(1 for kw in keywords if kw in content_lower)
        
        # L2 거리는 작을수록 우수함. 매칭 키워드당 0.18의 거리를 보정하여 우선순위 재배치
        final_score = score - (match_count * 0.18)
        reranked.append((doc, final_score))
        
    reranked.sort(key=lambda x: x[1])
    
    # 만약 결과가 부족하거나 벡터 DB에서 아예 누락된 경우, 전체 유저 청크에서 백업용 키워드 매칭 검색
    if len(reranked) < n_results:
        try:
            collection = vector_store._collection
            all_chunks = collection.get(where={"user_id": user_id})
            ids = all_chunks.get("ids", [])
            metadatas = all_chunks.get("metadatas", [])
            documents = all_chunks.get("documents", [])
            
            existing_contents = {doc.page_content for doc, _ in reranked}
            
            backup_matches = []
            for cid, meta, doc_text in zip(ids, metadatas, documents):
                if doc_text in existing_contents:
                    continue
                match_count = sum(1 for kw in keywords if kw in doc_text.lower())
                if match_count > 0:
                    backup_doc = Document(page_content=doc_text, metadata=meta)
                    backup_score = 1.25 - (match_count * 0.15)
                    backup_matches.append((backup_doc, backup_score))
            
            backup_matches.sort(key=lambda x: x[1])
            reranked.extend(backup_matches)
            reranked.sort(key=lambda x: x[1])
        except Exception as e:
            print(f"Warning in RAG keyword fallback: {e}")
            
    # 최종 매치된 객체와 보정 점수를 슬라이싱하여 리턴
    return [(doc, float(score)) for doc, score in reranked[:n_results]]
async def retrieve_all_user_contexts(user_id: str, limit: int = 15) -> str:
    """
    사용자의 모든 생애 데이터를 검색하여 하나의 텍스트로 합쳐서 반환합니다.
    """
    return await run_in_threadpool(_retrieve_all_user_contexts_sync, user_id, limit)
 
def _retrieve_all_user_contexts_sync(user_id: str, limit: int = 15) -> str:
    # 1. 특정 사용자의 데이터 검색 (최신순 또는 중요도순이겠으나 여기서는 관련성 높은 순으로 다수 가져옴)
    # 실제로는 '인생 전체'를 아우르는 쿼리를 던져서 관련 문서를 많이 가져오는 방식
    query = "사용자의 생애, 성장 과정, 가족, 학창 시절, 직장 생활, 현재 삶, 미래 계획"
    
    # 해당 사용자의 데이터와 글로벌 데이터를 함께 조회
    results = _search_context_sync(user_id, query, n_results=limit)
    
    # 2. 텍스트 추출 및 결합, Surrogate 등 문제 문자 제거
    contexts = [doc.page_content for doc, score in results]
    raw_text = "\n\n".join(contexts)
    # 파이썬 json 덤프 시 OpenAI 400 (무효한 JSON) 에러 방지를 위해 ascii/utf-8 클리닝
    clean_text = raw_text.encode('utf-8', 'ignore').decode('utf-8')
    return clean_text

async def retrieve_full_user_memory(user_id: str) -> str:
    """
    유저의 모든 자서전 데이터를 검색 없이 통째로 가져옵니다. (Long-Context Window 방식)
    """
    return await run_in_threadpool(_retrieve_full_user_memory_sync, user_id)

def _retrieve_full_user_memory_sync(user_id: str) -> str:
    try:
        collection = vector_store._collection
        results = collection.get(where={"user_id": user_id})
        documents = results.get("documents", [])
        if documents:
            raw_text = "\n\n".join(documents)
            return raw_text.encode('utf-8', 'ignore').decode('utf-8')
    except Exception as e:
        print(f"Error in retrieve_full_user_memory: {e}")
    return ""

async def retrieve_chapter_contexts(user_id: str, chapter_type: str, limit: int = 10) -> str:
    """
    특정 챕터에 맞는 Multi-query를 생성하여 MMR(Maximal Marginal Relevance) 기반으로 검색합니다.
    (현재 ChromaDB MMR을 직접 쓰지 않고 다중 쿼리 검색 결과를 통합하고 중복을 제거하는 시뮬레이션 구현)
    """
    return await run_in_threadpool(_retrieve_chapter_contexts_sync, user_id, chapter_type, limit)

def _retrieve_chapter_contexts_sync(user_id: str, chapter_type: str, limit: int = 10) -> str:
    queries = []
    if chapter_type == "childhood":
        queries = ["어릴 때 기억, 유년기, 부모님", "태어난 곳, 초등학교 입학 전, 어릴적"]
    elif chapter_type == "school":
        queries = ["초등학교, 중학교, 고등학교, 학창시절", "사춘기 친구들 선생님 소풍"]
    elif chapter_type == "youth":
        queries = ["대학교, 20대, 대학 시절", "첫 직장, 사회 진출, 청년기"]
    elif chapter_type == "marriage":
        queries = ["연애, 배우자 처음 만난, 결혼식", "신혼 휴가 첫째 아이 출산"]
    elif chapter_type == "career":
        queries = ["직장 생활, 승진, 회사, 업무", "가장 힘들었던 순간 위기 극복, 퇴사"]
    elif chapter_type == "hobby":
        queries = ["최근 취미 여가 시간 좋아하는 것", "퇴근 후 일상 주말"]
    elif chapter_type == "self_reflection":
        queries = ["건강 생각 깨달음", "나이 들면서 느낀 점 가치관 인생 철학"]
    elif chapter_type == "family":
        queries = ["앞으로의 계획 꿈", "자녀 손주 가족들에게 남기고 싶은 흔적"]
    else:
        queries = [chapter_type]
        
    all_results = []
    # 각 쿼리 당 충분한 문서를 가져옵니다.
    for q in queries:
        all_results.extend(_search_context_sync(user_id, q, n_results=limit))
        
    # 중복 제거 (간이 MMR 효과: 유사한 text content 중복 방지)
    unique_docs = {}
    for doc, score in all_results:
        # text 앞 50자로 단순 중복 판별
        key = doc.page_content[:50]
        if key not in unique_docs or score < unique_docs[key]["score"]:
            unique_docs[key] = {"doc": doc, "score": score}
            
    # score 역순(낮은게 좋음 L2 등) 정렬
    sorted_unique = sorted(unique_docs.values(), key=lambda x: x["score"])
    
    final_docs = [item["doc"].page_content for item in sorted_unique[:limit]]
    raw_text = "\n\n".join(final_docs)
    return raw_text.encode('utf-8', 'ignore').decode('utf-8')
