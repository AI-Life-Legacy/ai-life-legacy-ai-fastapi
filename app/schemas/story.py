from pydantic import BaseModel, Field
from typing import List, Optional

class LifeEvent(BaseModel):
    id: str = Field(..., description="고유 이벤트 ID")
    estimated_age: Optional[int] = Field(None, description="추정 나이")
    estimated_year: Optional[int] = Field(None, description="추정 연도")
    location: str = Field("", description="사건이 일어난 장소")
    people: List[str] = Field(default_factory=list, description="관련된 인물 목록")
    event_summary: str = Field(..., description="사건의 핵심 요약")
    emotion: str = Field("", description="사건 당시 주된 감정 (예: 기쁨, 슬픔, 긴장, 뿌듯함)")
    event_type: str = Field(..., description="사건 분류 (family, school, career, crisis, hobby, romance, self_reflection 등)")
    life_stage: str = Field("present", description="생애 주기 카테고리 (childhood, youth, university, early_career, career_crisis, present, future 중 1개)")
    # 인과 관계 매핑을 위한 리스트
    leads_to: List[str] = Field(default_factory=list, description="이 사건이 원인이 되어 발생한 다른 LifeEvent ID 목록")

class Scene(BaseModel):
    title: str = Field(..., description="Scene의 제목")
    setting: str = Field("", description="Scene의 주된 시간적/공간적 배경")
    characters: List[str] = Field(default_factory=list, description="등장인물")
    conflict: str = Field("", description="내적 갈등 또는 외적 시련")
    turning_point: str = Field("", description="가장 극적인 전환점 또는 깨달음의 순간")
    resolution: str = Field("", description="갈등의 해소 또는 결과")
    reflection: str = Field("", description="현재 시점에서의 성찰 (나레이션용)")
    linked_events: List[LifeEvent] = Field(default_factory=list, description="이 Scene에 포함된 2개 이상의 관련된 LifeEvent들")
    emotion_intensity: int = Field(1, description="감정의 강도 (1~10)")
    spread_hint: str = Field("", description="PDF 레이아웃을 위한 힌트 (예: needs_quote_after)")

class ChapterData(BaseModel):
    chapter_num: int
    chapter_title: str
    chapter_type: str = Field(..., description="유년기, 학창시절 등 8개 분류 중 하나")
    mood: str = Field("family", description="임베딩 기반으로 감지된 분위기 (childhood, youth, career 등 8개 중 하나)")
    scenes: List[Scene] = Field(default_factory=list, description="이 챕터에 포함된 Scene 목록")
    generated_text: str = Field("", description="LLM을 통해 최종 생성된 챕터 본문 (Markdown)")
