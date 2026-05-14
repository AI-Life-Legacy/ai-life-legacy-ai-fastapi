# AI 서버 RAG 및 검색(Search) 구조 분석

본 문서는 FastAPI AI 서버에서 사용자의 과거 자서전 기록이나 답변을 검색하여 활용하는 **RAG(Retrieval-Augmented Generation)** 파이프라인과 관련 API 구조를 정리한 기술 문서입니다.

---

## 1. 검색 관련 API 및 기능

1.  **명시적 검색 API** (`POST /api/v1/rag/search`)
    *   주요 역할: 특정 사용자(`userId`)의 데이터 베이스 안에서 텍스트 쿼리(`query`)에 매칭되는 가장 관련성 높은 문서를 직접 조회하여 점수(Score)와 함께 반환합니다.
2.  **아바타 채팅 내부 컨텍스트 검색** (in `generate_avatar_response`)
    *   주요 역할: 사용자가 채팅 메시지를 보내면, AI가 답변하기 직전에 해당 메시지를 쿼리로 사용하여 Vector Store를 검색(Top 3)합니다.
3.  **자서전 챕터별 데이터 조회** (in `retrieve_chapter_contexts`)
    *   주요 역할: 자서전을 렌더링할 때 각 챕터 주제에 맞는 Multi-query를 백그라운드에서 실행하고, 결과의 중복을 제거하여 챕터 본문 작성에 필요한 방대한 재료를 확보합니다.

---

## 2. 데이터 소스 및 저장소 구조

*   **원본 데이터 (문서)**: `POST /api/v1/rag/sync` 등을 통해 백엔드에서 전달되는 사용자 답변 및 기록 텍스트입니다.
*   **Vector Store**: **ChromaDB** (LangChain `Chroma` 래퍼 사용). 외부 관계형 DB 연동 없이 파일 시스템 기반(`storage/chroma_db`)으로 로컬에서 자체 영구 저장(persist)됩니다.
*   **Embedding 모델**: `OpenAIEmbeddings`를 사용하며, 모델은 **`text-embedding-3-small`** (기본값)을 사용하여 비용 대비 높은 품질의 벡터 변환을 수행합니다.
*   **Text Splitter (청크 분할)**: `RecursiveCharacterTextSplitter`를 사용하여 문서를 쪼갭니다.
    *   `chunk_size`: 1000
    *   `chunk_overlap`: 200

---

## 3. RAG 처리 단계 (Processing Algorithm)

### 1단계: 쿼리(Query) 구성 및 임베딩
*   **단일 쿼리**: 채팅이나 검색의 경우 사용자의 입력 문자열(Message)을 그대로 쿼리로 사용합니다.
*   **다중 쿼리 (Multi-query)**: 자서전 챕터 생성 시 "유년기", "학교 생활" 등 챕터에 맞춰 내부적으로 2~4개의 확장 쿼리를 동시에 실행합니다 (예: "초등학교, 중학교, 고등학교, 학창시절", "사춘기 친구들 선생님 소풍").

### 2단계: 벡터 유사도 검색 (Similarity Search)
*   **유저 필터링**: `filter={"user_id": user_id}`를 통해 해당 사용자의 데이터만 독립적으로 검색합니다.
*   **글로벌 데이터 결합**: `{"user_id": "__GLOBAL__"}`로 지정된 전역 데이터(공통 페르소나 지식 등)도 함께 조회합니다.
*   ChromaDB가 쿼리 임베딩과 문서 임베딩의 거리(L2 Distance 기준)를 비교하여 점수를 매깁니다. (낮은 점수일수록 유사함)

### 3단계: 정렬 및 Top-K 추출, 중복 제거
*   조회된 문서들을 Distance 오름차순으로 정렬 후 상위 `n_results` (아바타 채팅 시 3개)를 추출합니다.
*   **간이 MMR (Maximal Marginal Relevance) 적용**: 자서전 챕터 조회 시에는 텍스트의 앞 50글자를 키(Key)로 삼아 내용이 겹치는 중복 검색 결과를 제거하는 고도화 로직을 수행합니다.

### 4단계: 컨텍스트(Context) 병합 및 Prompt 주입
*   추출된 `Document`들의 `page_content`를 추출하여 불릿 포인트(`- `) 형태로 연결합니다.
*   자서전 생성의 경우 UTF-8 파싱 오류 방지를 위해 Surrogate 문자를 클리닝(`encode-decode`)합니다.
*   병합된 거대한 문자열이 OpenAI 요청 시 `{context}` 변수에 주입됩니다.

---

## 4. `context_used` 반환 기준 (아바타 채팅)

`POST /chat` 엔드포인트 응답에는 `context_used`라는 Boolean 필드가 포함됩니다. 다음과 같은 기준에 따라 결정됩니다.

*   **`True`가 되는 경우**:
    *   `user_id`를 기반으로 RAG 검색을 수행하여, 1개 이상의 관련 텍스트 문서를 찾아 `context_text` 문자열의 길이가 0 이상으로 구성된 경우.
*   **`False`가 되는 경우**:
    *   `user_id`가 `"anonymous"`, `"unknown"`, 또는 비어있어서 검색 로직 자체를 Skip한 경우.
    *   VectorDB 검색을 시도했으나 쿼리와 유사도가 높은 문서가 단 1건도 매칭되지 않은 경우.
    *   ChromaDB 접속이나 검색 과정에서 Exception(예외)이 발생한 경우 (이 경우 에러를 스킵하고 빈 컨텍스트로 진행).

---

## 5. RAG 적용의 한계 및 주의사항

1.  **근거 없는 답변(Hallucination) 방지책**
    *   RAG의 고질적 문제인 환각 현상을 억제하기 위해, OpenAI 프롬프트상에 **"외부 지식 절대 사용 금지", "제공된 과거 기록에 없는 경우 절대로 지어내지 말 것"** 이라는 강력한 통제 룰을 삽입했습니다.
2.  **Context가 텅 비었을 때의 방어 로직 (Fallback)**
    *   검색된 기록이 아무것도 없을 때 Prompt에 빈 칸을 넣으면 AI가 혼란을 겪습니다. 이를 방지하기 위해 로직 단에서 **"제공된 과거 기억이나 자서전 기록이 없습니다. 일상적인 대화 어조로 성심껏 응답하세요."**라는 문자열을 강제로 `{context}` 영역에 채워 넣어, 일상 대화 모드로 매끄럽게 전환되도록 방어합니다.
3.  **Role Persona vs Context의 우선순위**
    *   **사실 관계(Fact)**는 무조건 Context를 최우선하며 임의 변형이 불가합니다.
    *   **어조(Tone)**와 **반응 방식(Attitude)**은 철저하게 Role Persona(아버지, 어머니, 큐레이터 등)의 지시를 따릅니다.
    *   결과적으로 AI는 "자서전의 팩트(Context)를 재료로 삼아, 부여받은 배역(Persona)의 성격대로 연기"하게 됩니다.
