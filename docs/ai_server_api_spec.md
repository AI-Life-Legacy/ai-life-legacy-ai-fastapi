# AI Server API Specification

본 문서는 실제 구현된 FastAPI 코드를 기준으로 작성된 AI 서버 API 명세서입니다. 프록시(Proxy) 등에서 단순히 더미(Dummy) 응답을 반환하는 코드는 제외하거나 명시하고, 실제 AI 연산 및 파일 생성 로직이 동작하는 구현체를 중점적으로 정리했습니다.

## 1. API 요약 표

| Method | Endpoint | Request Body | Response | 주요 처리 로직 | OpenAI 호출 여부 | 파일 생성 여부 | 호출하는 서비스/함수 |
|---|---|---|---|---|---|---|---|
| POST | `/api/v1/case` | `UserCaseRequest` | `UserCaseResponse` | 사용자 자기소개 기반 자서전 작성 유형 분류 | O | X | `classify_user_case` |
| POST | `/api/v1/generation/question` | `QuestionRequest` | `QuestionResponse` | 과거 답변과 RAG 컨텍스트 기반의 꼬리 질문 생성 | O | X | `generate_follow_up_question` |
| POST | `/api/v1/generation/autobiography` | `AutobiographyRequest` | `AutobiographyResponse` | 답변 내용 통합, 마크다운 생성, PDF 렌더링 및 해시 기반 캐싱 처리 | O | O (MD, PDF, JSON 캐시) | `autobiography_service.generate_autobiography_memoir`, `pdf_service.generate_premium_pdf` |
| POST | `/api/v1/chat/chat` <br> (Proxy: `/chat`) | `ChatRequest` | `ChatResponse` | 작성자/뷰어 모드 및 페르소나 롤(Role) 기반 아바타 채팅 응답 | O | X | `generate_avatar_response` |
| POST | `/api/v1/rag/search` | `RagSearchRequest` | `RagSearchResponse` | 벡터 스토어에서 질문(Query)에 대한 유사도 기반 컨텍스트 검색 | O (Embedding) | X | `search_context` |
| POST | `/combine` <br> (Proxy) | `CombineRequest` | `{"combined_text": "..."}` | 두 개의 텍스트 데이터를 하나로 병합 (현재 단순 문자열 결합 로직) | X | X | - |
| GET | `/generated-pdfs/{filename}` | - | PDF File | 정적 파일(`generated_pdfs` 디렉토리) 서빙 | X | X | `StaticFiles` 라우팅 |

---

## 2. API 상세 명세

### 1. 자서전 유형 분류 (Case Classification)
- **Endpoint**: `/api/v1/case`
- **Request Body**
  ```json
  {
    "introText": "어릴 적 시골에서 자라서 농사에 관심이 많습니다..."
  }
  ```
- **Response**
  ```json
  {
    "case": "case1",
    "reasoning": "어린 시절 농촌 경험이 중심이므로 기본 자서전 유형에 적합"
  }
  ```

### 2. 꼬리 질문 생성 (Follow-up Question)
- **Endpoint**: `/api/v1/generation/question`
- **Request Body**
  ```json
  {
    "toc_id": 1,
    "current_answer": "그때는 정말 눈이 많이 왔었어요.",
    "chat_history": [
      {"role": "ai", "content": "어린 시절 가장 기억에 남는 날은 언제인가요?"},
      {"role": "user", "content": "그때는 정말 눈이 많이 왔었어요."}
    ]
  }
  ```
- **Response**
  ```json
  {
    "question": "그렇게 눈이 많이 온 날, 가족들과 어떤 놀이를 하며 시간을 보내셨나요?"
  }
  ```

### 3. 자서전 생성 (Autobiography Generation)
- **Endpoint**: `/api/v1/generation/autobiography`
- **Request Body** (`answers` 또는 `chapters` 배열 필수)
  ```json
  {
    "user_id": "user123",
    "user_name": "홍길동",
    "answers": [
      {"question": "어릴 적 기억", "answer": "동네 친구들과 매일 놀았음..."}
    ],
    "force": false
  }
  ```
- **주요 처리 로직**: 
  1. `answers` 데이터를 파싱하고 추출하여 SHA-256 Content Hash 생성.
  2. `force=false`이고 캐시 데이터가 있으면 곧바로 캐시된 URL 반환 (캐시 히트 로직 적용).
  3. 캐시 미스 시 OpenAI를 통해 회고록 마크다운 생성.
  4. 생성된 MD 파일을 로컬에 저장하고 PDF로 렌더링하여 `generated_pdfs` 폴더에 저장.
  5. PDF 파일 사본을 시스템 Downloads 폴더에 복사 시도.
- **반환 구조 (pdf_url / page_count)**
  ```json
  {
    "status": "COMPLETED",
    "pdf_url": "http://localhost:8000/generated-pdfs/autobiography_user123_hash값.pdf",
    "page_count": 12,
    "cached": false
  }
  ```

### 4. 아바타 채팅 (Avatar Chat)
- **Endpoint**: `/api/v1/chat/chat` (프록시: `/chat`)
- **Request Body**
  ```json
  {
    "user_id": "user123",
    "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "role_id": "curator",
    "role": "friendly assistant",
    "message": "안녕, 오늘 날씨가 참 좋아",
    "mode": "writer",
    "viewer_id": "viewer456"
  }
  ```
- **반환 구조 (answer / session_id / context_used)**
  ```json
  {
    "answer": "안녕하세요! 맑은 날씨에 기분이 좋으시겠어요. 오늘은 어떤 추억을 이야기해 볼까요?",
    "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "role_id": "curator",
    "context_used": false
  }
  ```

### 5. RAG 컨텍스트 검색 (RAG Search)
- **Endpoint**: `/api/v1/rag/search`
- **Request Body**
  ```json
  {
    "userId": "user123",
    "query": "어릴 적 눈 왔던 날"
  }
  ```
- **Response**
  ```json
  {
    "results": [
      {
        "text": "그때는 정말 눈이 많이 왔었어요...",
        "score": 0.89
      }
    ]
  }
  ```

### 6. 답변 결합 (Combine)
- **Endpoint**: `/combine`
- **Request Body**
  ```json
  {
    "question1": "질문1",
    "data1": "답변1",
    "question2": "질문2",
    "data2": "답변2"
  }
  ```
- **Response** (현재 OpenAI 없이 단순 문자열 포맷팅 처리)
  ```json
  {
    "combined_text": "답변1\n\n답변2"
  }
  ```

### 7. 생성된 PDF 정적 파일 경로 (Generated PDFs)
- **Endpoint**: GET `/generated-pdfs/{filename}`
- **설명**: FastAPI의 `StaticFiles`를 사용하여 `BASE_DIR/generated_pdfs` 디렉토리를 마운트. 자서전 생성 결과물인 PDF에 브라우저를 통해 직접 접근하거나 다운로드하는 용도로 활용됩니다.

---

## 3. 공통 고려사항

### 에러 응답 구조
AI 서버는 예외 발생 시 FastAPI `HTTPException`을 발생시키며, 특히 OpenAI 호출 과정의 인증 및 권한 예외를 다음과 같이 세분화하여 처리합니다.
```json
{
  "detail": "OpenAI Authentication Error: ..."
}
```
- **400 Bad Request**: 필수 파라미터 누락 (예: `userId` 누락, `answers` 없음, `message` 빈 문자열 등)
- **401 Unauthorized**: OpenAI API 인증 실패 (`AuthenticationError` 또는 `api_key` 관련 오류)
- **403 Forbidden**: OpenAI 모델 접근 권한 부족 (`PermissionDeniedError`, `model_not_found` 등)
- **429 Too Many Requests**: OpenAI API 호출 할당량 초과 및 속도 제한 (`RateLimitError`, `quota` 등)
- **500 Internal Server Error**: 기타 코드 내부 로직 및 처리 실패 에러

### Timeout 및 캐시 고려사항
1. **Timeout (타임아웃)**: 
   - 자서전 마크다운 생성이나 꼬리 질문 생성은 프롬프트의 복잡성에 따라 OpenAI API의 응답 지연 시간이 길어질 수 있습니다.
   - NestJS 백엔드 측에서는 Axios/Fetch 등 HTTP 클라이언트에서 넉넉한 Timeout(예: 30초~60초 이상)을 설정해야 합니다.
2. **캐시 전략 (자서전 생성)**:
   - 자서전 생성 요청 시 `answers` 객체를 직렬화하여 **SHA-256 해시값**을 추출합니다.
   - 이 해시값을 기준으로 `.cache/autobiography/` 경로에 메타데이터(pdf_url, page_count)를 저장하고, `generated_pdfs/`에 PDF를 저장합니다.
   - 데이터 변경이 없을 경우 OpenAI를 재호출하지 않고 파일 시스템의 캐시된 PDF를 즉시 반환하여 생성 속도를 획기적으로 높였습니다. (단, `force=true` 요청 시 기존 캐시를 무시하고 강제로 재생성 수행)
