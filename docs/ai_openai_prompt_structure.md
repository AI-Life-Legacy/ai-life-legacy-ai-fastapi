# AI 서버 OpenAI 호출 및 프롬프트 구조 분석

본 문서는 FastAPI AI 서버 내에서 OpenAI API가 어떻게 구성되고 호출되는지, 그리고 각 기능별 프롬프트(Prompt)와 에러 및 비용 방지 정책이 어떻게 설계되었는지 정리한 기술 문서입니다.

---

## 1. OpenAI 기본 설정

*   **API Key 환경변수**: `.env` 파일의 `OPENAI_API_KEY` (`app/core/config.py`에서 관리)
*   **사용 모델명**:
    *   `gpt-4o-mini`: 빠른 응답이 필요한 케이스 분류, 꼬리 질문 생성, 정보 추출, 아바타 채팅 등 (저비용/고효율)
    *   `gpt-4o`: 깊이 있는 서사와 복잡한 제약 조건(JSON 강제, 고유명사 증폭, 챕터 연결)을 만족해야 하는 자서전 본문 생성 (고비용/고품질)
    *   `text-embedding-3-small`: RAG 컨텍스트 벡터화 모델 (`OPENAI_EMBEDDING_MODEL` 기본값)
    *   `tts-1`: 아바타 음성 생성
*   **Temperature**:
    *   `0.0` (정확성): 자서전 케이스 분류 (`classify_user_case`)
    *   `0.1` (사실 기반): 고유명사 추출 (`_extract_personal_details`)
    *   `0.7` (창의성/자연스러움): 꼬리 질문 생성, 자서전 본문 작성, 아바타 채팅
*   **Max Tokens & Timeout**: 명시적인 제한 없이 `openai-python` 라이브러리의 Default 설정을 따름 (모델의 컨텍스트 윈도우 최대 활용).
*   **Client 초기화 위치**:
    *   `app/services/openai_service.py` (전역 `AsyncOpenAI` 인스턴스)
    *   `app/services/autobiography_service.py` (클래스 내부 `self.client = AsyncOpenAI(...)`)

---

## 2. 기능별 OpenAI 호출 구조

| 기능 | 호출 함수/파일 | 사용 모델 | 입력 Prompt (요약) | 출력 형식 | 비용 높은 호출 여부 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **자서전 케이스 분류** | `classify_user_case`<br>(openai_service) | `gpt-4o-mini` | 자기소개 텍스트 기반으로 6가지 케이스(학력/결혼/자녀유무) 중 1개 선택 | 단일 텍스트<br>("case1") | 낮음 |
| **꼬리 질문 생성** | `generate_follow_up_question`<br>(openai_service) | `gpt-4o-mini` | 대화 이력, 최근 답변, RAG 컨텍스트를 주입하여 파생 질문 생성 | 단일 텍스트 | 낮음 |
| **고유명사 추출** | `_extract_personal_details`<br>(autobiography_service) | `gpt-4o-mini` | RAG 문맥에서 사람, 장소, 활동, 업적 등 5개 카테고리 추출 | JSON Object | 보통 |
| **자서전 챕터 작성** | `_generate_chapter_text`<br>(autobiography_service) | `gpt-4o` | Scene 구조, 고유명사 리스트, 이전/다음 챕터 제목을 주입하여 소설식 서사 및 인용구 생성 | JSON Object | **높음** |
| **아바타 채팅** | `generate_avatar_response`<br>(openai_service) | `gpt-4o-mini` | 페르소나 지시사항 + RAG 문맥 + 사용자 질문 | 단일 텍스트 | 낮음 |
| **아바타 보이스** | `generate_voice_response`<br>(openai_service) | `tts-1` | 텍스트 기반 보이스 합성 (role_id에 따라 음성 모델 분기) | Audio Bytes | 보통 |

---

## 3. 프롬프트(Prompt) 세부 구조

`app/prompts/templates.py` 및 생성 서비스 파일 내에 정의된 프롬프트들의 구조입니다. 긴 문장 대신 핵심 Instruction만 요약합니다.

### 1) Case 분류 프롬프트 (`CASE_CLASSIFICATION_USER`)
*   **구조**: 사용자 데이터 + 6가지 Case 정의 (대졸 여부/기혼 여부/자녀 여부).
*   **통제**: 일치하는 번호를 오직 "case1" 형태로만 응답하도록 통제. 부족할 시 기본값 "case1" 반환 지시.

### 2) Question 생성 프롬프트 (`QUESTION_GENERATION_USER`)
*   **구조**: 기존 질문 + 답변 내역 + 과거 문맥(RAG).
*   **통제**: 따옴표나 추가 설명 없이 흥미로운 2차 질문(꼬리 질문) 1개만 작성하도록 제약.

### 3) Combine 프롬프트 (`AUTOBIOGRAPHY_COMBINATION_USER`)
*   **구조**: 질문 2개와 각각의 답변 2개 + 과거 문맥.
*   **통제**: 단답형의 답변을 문학적이고 부드러운 형태로 엮어내기 위한 **Few-Shot 예시(예시 질문/답변 -> 원하는 스타일의 답변)** 제공.

### 4) Autobiography 본문 프롬프트 (Service 내 하드코딩)
*   **구조**:
    *   **역할 지시**: 챕터별 Role, Theme, Tone, Avoid(피해야 할 내용) 적용.
    *   **서사 구조**: 단순 나열 금지. [상황 -> 갈등 -> 변화 -> 성찰] 구조 강제.
    *   **Transition 지시**: 다음 챕터 제목을 알려주고, 본문 마지막에 물 흐르듯 넘어가는 "연결 문장" 1~2개 추가 지시.
    *   **고유명사 증폭**: 추출해둔 `personal_details` JSON을 주고, 이름/장소를 무조건 본문에 2~3개 이상 포함하도록 강제.
    *   **출력 제약**: `{"content": "...", "quote": "..."}` 의 JSON 형태 반환 강제.

### 5) Avatar Role Persona 프롬프트 (`AVATAR_SYSTEM_{ROLE}`)
*   **구조**: 페르소나에 따른 성격, 말투(존댓말/반말 여부), 예시 텍스트 제공.
    *   `CURATOR`: 3인칭 존댓말(하십시오/해요체), 사려 깊은 안내자.
    *   `FATHER` / `MOTHER`: 1인칭 부모 시점, 다정한 반말(해라체/해체), 무조건적 지지와 위로.
    *   `BROTHER` / `SISTER`: 활기차거나 묵묵한 반말, 든든한 형제/자매 포지션.
    *   `SELF`: 1인칭 존댓말(해요체), 진지한 자아 성찰 모드.

### 6) Hallucination(환각) 방지 프롬프트 (가장 엄격함)
*   **적용 위치**: 아바타 채팅 공통 시스템 프롬프트 (`AVATAR_CHAT_PROMPT` 및 개별 페르소나)
*   **주요 지시문**:
    1.  **외부 지식 사용 금지**: 학습된 일반 지식(예: 아이들은 보통 축구를 좋아함) 사용 절대 금지.
    2.  **모르는 것에 대한 답변**: RAG 컨텍스트에 없는 내용을 질문받으면, 지어내거나 추측하지 말고 **반드시 "기록에는 정확히 남아 있지 않아요"라고만 답변할 것**.

---

## 4. 에러 및 예외 처리 로직 (Error Handling)

AI 서버 엔드포인트(`generation.py`, `chat.py`)는 `try-except`로 예외를 잡아 다음과 같이 처리합니다.

1.  **AuthenticationError (인증 오류)**
    *   원인: API Key 만료 또는 누락.
    *   반환: **HTTP 401** `OpenAI Authentication Error`
2.  **PermissionDeniedError / Model_Not_Found**
    *   원인: 해당 모델(`gpt-4o` 등)에 대한 권한이 없거나 찾을 수 없음.
    *   반환: **HTTP 403** `OpenAI Permission/Access Error`
3.  **RateLimitError / Quota Exceeded**
    *   원인: 단시간 내 너무 많은 요청, 또는 결제 한도 초과.
    *   반환: **HTTP 429** `OpenAI Rate Limit / Quota Error`
4.  **Timeout / Internal Error**
    *   원인: 응답 지연(OpenAI 서버 폭주) 또는 기타 예외.
    *   반환: **HTTP 500** Internal Server Error.
5.  **빈 응답 또는 짧은 입력(Fallback 여부)**
    *   `classify_user_case`: 입력이 5자 미만으로 빈약할 경우 OpenAI를 호출하지 않고 **디폴트 `case1` JSON을 즉시 반환(Fallback)**.
    *   `generate_avatar_response`: `user_id`가 `anonymous`이거나 검색된 문맥이 없을 경우, 에러를 내지 않고 기본 지시문("과거 기록이 없으니 일상적 대화로 응답하라")으로 Fallback.

---

## 5. 비용 및 중복 방지 전략 (Caching)

가장 비용이 높고 렌더링에 수십 초가 소요되는 **자서전 생성(`autobiography`) API**의 경우, 막강한 중복 방지 캐싱 레이어를 내장하고 있습니다.

*   **백엔드 의존도 최소화**: 백엔드에서 자체 캐싱을 하지 않더라도, AI 서버 자체가 전달받은 `request.answers` 배열 전체를 직렬화한 뒤 **SHA-256 해시값(Content Hash)**을 생성합니다.
*   **AI 서버 내부 File Caching**: 
    *   `.cache/autobiography/{hash}.json` 파일의 존재 여부를 검사합니다.
    *   만약 해시값이 동일하다면 (즉, 사용자 답변이 토씨 하나 바뀌지 않았다면), **비싼 `gpt-4o` 연산과 PDF 렌더링 과정을 완전히 생략(Bypass)**합니다.
    *   기존에 만들어둔 로컬 PDF의 정적 라우팅 경로(`pdf_url`)와 `page_count`만 즉시 반환하여, 불필요한 OpenAI 토큰 소모를 100% 방어하는 훌륭한 구조를 갖추고 있습니다. (`force=true` 파라미터가 들어올 때만 캐시를 무시하고 강제 재생성함)
