import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.vector_store import add_document
from app.core.config import settings

async def train_avatar():
    data_path = Path("storage/data/avatar_sample.txt")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 글로벌 지식으로 저장 (userId를 "__GLOBAL__"로 설정)
    print(f"Loading avatar data from {data_path}...")
    await add_document(
        user_id="__GLOBAL__",
        text=content,
        metadata={"source": "avatar_sample", "type": "global_knowledge"}
    )
    print("Success: Avatar knowledge injected into vector store.")

if __name__ == "__main__":
    asyncio.run(train_avatar())
