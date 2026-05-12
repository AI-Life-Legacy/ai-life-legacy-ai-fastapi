import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_chat():
    print("\n--- Testing POST /chat ---")
    payload = {
        "user_id": "test_user_123",
        "session_id": "session_abc",
        "role_id": "father",
        "message": "아버지, 어린 시절 가장 행복했던 기억이 뭐예요?"
    }
    response = requests.post(f"{BASE_URL}/chat/chat", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_voice():
    print("\n--- Testing POST /chat/voice ---")
    payload = {
        "text": "안녕하세요, 아들의 행복한 기억을 들려줄게요.",
        "role_id": "father"
    }
    response = requests.post(f"{BASE_URL}/chat/chat/voice", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    if response.status_code == 200:
        with open("test_voice.mp3", "wb") as f:
            f.write(response.content)
        print("Audio saved to test_voice.mp3")

def test_question():
    print("\n--- Testing POST /question ---")
    payload = {
        "toc_id": 1,
        "current_answer": "저는 어릴 때 나무 기차를 가지고 노는 걸 좋아했어요.",
        "chat_history": [
            {"role": "ai", "content": "어릴 때 어떤 장난감을 좋아했니?"},
            {"role": "user", "content": "저는 어릴 때 나무 기차를 가지고 노는 걸 좋아했어요."}
        ]
    }
    response = requests.post(f"{BASE_URL}/generation/question", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_autobiography():
    print("\n--- Testing POST /autobiography ---")
    payload = {
        "userId": "test_user_123",
        "userName": "홍길동"
    }
    response = requests.post(f"{BASE_URL}/generation/autobiography", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    # Note: Ensure the server is running before executing this script
    try:
        test_chat()
        test_voice()
        test_question()
        test_autobiography()
    except Exception as e:
        print(f"Error: {e}")
