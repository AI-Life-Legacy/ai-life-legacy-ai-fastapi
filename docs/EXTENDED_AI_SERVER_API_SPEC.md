# EXTENDED_AI_SERVER_API_SPEC.md
(AI 자서전 Life Legacy - FastAPI AI 서버 기능 확장 명세서)

본 문서는 프론트엔드의 다채로운 아바타 페르소나 및 정교한 인터뷰 흐름을 위해 AI 서버(FastAPI)에서 보완해야 할 API 스펙입니다.

---

## 1. 아바타 채팅 (페르소나 및 세션 유지)
관람자가 선택한 역할(목소리)에 맞게 응답을 생성하고 대화 문맥을 유지합니다.

* **Endpoint**: `POST /chat`
* **Request Body**:
    ```json
    {
      "user_id": "string",     // 작성자 ID
      "session_id": "string",  // 대화 세션 ID (문맥 유지용)
      "role_id": "string",     // 페르소나 ID (curator, father, mother 등)
      "message": "string"      // 사용자 입력 메시지
    }
    ```
* **Response**:
    ```json
    {
      "answer": "string",
      "session_id": "string",
      "context_used": boolean  // RAG 참조 여부
    }
    ```

---

## 2. 꼬리 질문 생성 (대화 내역 인지)
단일 답변 분석이 아닌, 인터뷰 문맥을 파악하여 자연스러운 추가 질문을 던집니다.

* **Endpoint**: `POST /question`
* **Request Body**:
    ```json
    {
      "toc_id": number,
      "current_answer": "string",
      "chat_history": [
        { "role": "ai", "content": "..." },
        { "role": "user", "content": "..." }
      ]
    }
    ```

---

## 3. 자서전 생성 및 PDF 결과 처리
생성 프로세스 최적화 및 결과물(S3 업로드 등) 반환 로직을 명확히 합니다.

* **Endpoint**: `POST /autobiography`
* **Response (수정)**:
    파일 바이너리를 직접 반환하는 대신, 업로드된 결과물 정보를 반환합니다.
    ```json
    {
      "status": "COMPLETED",
      "pdf_url": "https://s3...",
      "page_count": number
    }
    ```

---

## 4. [신규] 아바타 TTS 음성 생성 (Voice)
아바타의 답변을 텍스트가 아닌 실제 목소리로 변환하여 풍부한 경험을 제공합니다.

* **Endpoint**: `POST /chat/voice`
* **Description**: 텍스트를 입력받아 페르소나에 맞는 오디오 데이터를 반환합니다.
* **Request Body**: `{ "text": "string", "role_id": "string" }`
* **Response**: `audio/mpeg` (Binary Stream)
