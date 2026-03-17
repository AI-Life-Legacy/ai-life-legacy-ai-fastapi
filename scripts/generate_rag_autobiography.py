import asyncio
import os
from pathlib import Path
import sys

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Load environments
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.services.autobiography_service import autobiography_service

async def main():
    # 테스트를 위해 __GLOBAL__ 사용자 데이터 사용 (avatar_sample.txt가 저장된 곳)
    user_id = "__GLOBAL__"
    user_name = "홍길동"
    
    print(f"Generating RAG-based autobiography for {user_name} ({user_id})...")
    
    try:
        memoir_text = await autobiography_service.generate_autobiography_memoir(user_id, user_name)
        
        # 결과 저장
        output_dir = PROJECT_ROOT / "storage" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "autobiography_output_rag.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(memoir_text)
            
        print(f"\nSuccess! Autobiography generated and saved to {output_path}")
        print("-" * 50)
        print(memoir_text[:500] + "...") # 앞부분만 살짝 출력
        print("-" * 50)
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
