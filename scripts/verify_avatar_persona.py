import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.openai_service import generate_avatar_response

async def verify_persona_and_truth():
    test_user_id = "verification_user"
    
    # 테스트 케이스 1: 화자 고정 및 페르소나 확인
    print("Test 1: Persona and Pronoun Check")
    q1 = "당신은 누구신가요?"
    r1 = await generate_avatar_response(test_user_id, q1, "irrelevant")
    print(f"Q: {q1}\nA: {r1}\n")

    # 테스트 케이스 2: 진실성 확인 (음악 vs 운동)
    print("Test 2: Truthfulness Check (Music vs Sports)")
    q2 = "어렸을 때 어떤 운동을 좋아하셨어요?"
    r2 = await generate_avatar_response(test_user_id, q2, "irrelevant")
    print(f"Q: {q2}\nA: {r2}\n")
    
    # 테스트 케이스 3: 자서전 내용 확인 (탄생)
    print("Test 3: Fact Checking (Birth)")
    q3 = "언제 태어나셨나요?"
    r3 = await generate_avatar_response(test_user_id, q3, "irrelevant")
    print(f"Q: {q3}\nA: {r3}\n")

    # 검증 로직
    is_father = "아버지" in r1 or "아빠" in r1 or "나" in r1
    truthful = "운동" not in r2 or "기억이 나지 않는다" in r2 or "음악" in r2
    fact_correct = "1969" in r3 and "서울" in r3

    if is_father and truthful and fact_correct:
        print("Final Verification: SUCCESS")
    else:
        print(f"Final Verification: FAILED (Persona: {is_father}, Truthful: {truthful}, Fact: {fact_correct})")

if __name__ == "__main__":
    asyncio.run(verify_persona_and_truth())
