import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.openai_service import generate_avatar_response

async def test_global_search():
    # 임의의 사용자 ID로 테스트 (이 사용자는 아직 데이터가 하나도 없는 상태)
    test_user_id = "test_user_123"
    test_message = "언제 어디서 태어났는지 알려줘"
    
    print(f"Testing global search for user: {test_user_id}")
    print(f"Message: {test_message}")
    
    response = await generate_avatar_response(test_user_id, test_message, "상냥한 손자")
    
    print("\n--- AI Response ---")
    print(response)
    print("-------------------\n")
    
    if "1969" in response and "서울" in response:
        print("Success: AI correctly retrieved global avatar knowledge!")
    else:
        print("Warning: AI response might not be based on the learned data.")

if __name__ == "__main__":
    asyncio.run(test_global_search())
