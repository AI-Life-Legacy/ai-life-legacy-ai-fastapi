import re
import os
from pathlib import Path
from jinja2 import Template
from weasyprint import HTML, CSS
from PIL import Image, ImageStat
from app.core.config import settings

# Project root setup (using settings if available)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Default Artifact Dir (should be configurable or dynamic, but using the one from the script for now)
ARTIFACT_DIR = Path(r"C:\Users\rmswn\.gemini\antigravity\brain\15aee333-3881-490a-804e-34d2290adc99")

# Thematic assets (assigned by mood)
MOOD_ASSETS = {
    "childhood": str(ARTIFACT_DIR / "memoir_childhood_warm_yard_1773758498739.png"),
    "youth": str(ARTIFACT_DIR / "memoir_youth_city_bicycle_1773758588102.png"),
    "career": str(ARTIFACT_DIR / "memoir_nostalgia_1_1773714327864.png"), # Fallback
    "marriage": str(ARTIFACT_DIR / "memoir_nostalgia_2_1773714346454.png"),
    "crisis": str(ARTIFACT_DIR / "memoir_nostalgia_3_1773714360412.png"),
    "family": str(ARTIFACT_DIR / "memoir_nostalgia_4_1773714379542.png"),
    "hobby": str(ARTIFACT_DIR / "memoir_nostalgia_1_1773714327864.png"),
    "future": str(ARTIFACT_DIR / "memoir_nostalgia_2_1773714346454.png"),
}

class PdfService:
    def detect_mood(self, title, content):
        content = title + " " + content
        if any(k in content for k in ["유년", "어린 시절", "태어난", "고향", "부모님"]): return "childhood"
        if any(k in content for k in ["학창", "청소년", "도시", "사춘기", "대학"]): return "youth"
        if any(k in content for k in ["졸업", "사회", "회사", "업무", "성취", "도전"]): return "career"
        if any(k in content for k in ["결혼", "동반자", "아내", "여행", "제주도"]): return "marriage"
        if any(k in content for k in ["위기", "힘들었", "야근", "구조조정", "실직"]): return "crisis"
        if any(k in content for k in ["현재", "아이들", "가족", "식사", "보람"]): return "family"
        if any(k in content for k in ["취미", "그림", "운동", "요리", "건강"]): return "hobby"
        if any(k in content for k in ["미래", "계획", "봉사", "철학", "나에게"]): return "future"
        return "family"

    def get_image_luminance(self, image_path: str):
        """Calculates the average luminance (brightness) of an image."""
        try:
            # Resolve path safely (handle file:/// prefix)
            clean_path = image_path.replace("file:///", "")
            local_path = clean_path.replace("/", os.sep)
            
            # Check if the file actually exists
            if not os.path.exists(local_path):
                return 128 # Middle fallback
            
            img = Image.open(local_path).convert('L')
            stat = ImageStat.Stat(img)
            return stat.mean[0] # Average brightness (0-255)
        except Exception as e:
            print(f"Warning: Luminance analysis failed for {image_path}: {e}")
            return 128

    def get_title_size(self, title: str):
        """Returns a CSS font-size based on the title length."""
        length = len(title)
        if length < 12: return "42pt"
        if length < 20: return "34pt"
        if length < 30: return "28pt"
        return "24pt"

    def estimate_lines(self, text: str) -> int:
        """
        글자 수와 기본 폭을 기반으로 텍스트가 차지할 줄 수를 추정합니다.
        1줄 = 약 38글자 기준
        """
        length = len(text)
        return max(1, length // 38) + 1

    def parse_markdown_content(self, content: str):
        # Title extraction
        title_match = re.search(r"^제목:\s*(.*)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "자서전"

        # Chapter extraction
        chapters = []
        sections = re.split(r"^##\s+", content, flags=re.MULTILINE)
        
        for i, section in enumerate(sections[1:]):
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            chapter_title = lines[0].strip()
            body = '\n'.join(lines[1:]).strip()
            
            # Mood extraction from generated comment
            mood_match = re.search(r"<!-- MOOD:\s*(.*?)\s*-->", body)
            mood = mood_match.group(1).strip() if mood_match else self.detect_mood(chapter_title, body)

            # Quote extraction from generated comment
            quote_match = re.search(r"<!-- QUOTE:\s*(.*?)\s*-->", body)
            chapter_quote = quote_match.group(1).strip() if quote_match else ""
            
            # Clean up tags
            body = re.sub(r"<!-- MOOD:.*?-->", "", body).strip()
            body = re.sub(r"<!-- QUOTE:.*?-->", "", body).strip()
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
            
            primary_image = "file:///" + MOOD_ASSETS.get(mood, MOOD_ASSETS["family"]).replace("\\", "/")
            
            chapters.append({
                "chapter_title": chapter_title,
                "paragraphs": paragraphs,
                "images": [primary_image],
                "mood": mood,
                "quote": chapter_quote
            })

        return {"title": title, "chapters": chapters}

    def paginate_to_spreads(self, data):
        """
        Deterministic Rhythm Pagination Engine.
        """
        spreads = []
        
        for chapter_idx, chapter in enumerate(data['chapters']):
            chapter_num = chapter_idx + 1
            paras = list(chapter['paragraphs'])
            chapter_templates = []
            
            intro_quote = paras[0][:80] + "..." if paras and len(paras[0]) > 80 else (paras[0] if paras else "")

            # --- Scene 1: Poster Opener (T1) ---
            text_for_opener = []
            if paras:
                text_for_opener = [paras.pop(0)]

            opener_img = chapter['images'][0] if chapter['images'] else ""
            has_bg_image = False
            if opener_img:
                clean_p = opener_img.replace("file:///", "").replace("/", os.sep)
                if os.path.exists(clean_p):
                    has_bg_image = True

            brightness = self.get_image_luminance(opener_img) if has_bg_image else 255
            title_theme = "dark" if brightness > 100 else "light"
            title_size = self.get_title_size(chapter['chapter_title'])
            
            spreads.append({
                "type": "T1",
                "chapter_num": chapter_num,
                "chapter_title": chapter['chapter_title'],
                "title_size": title_size,
                "images": chapter['images'],
                "mood": chapter['mood'],
                "title_theme": title_theme,
                "has_bg_image": has_bg_image,
                "intro_quote": intro_quote,
                "left_content": {"paragraphs": []}, 
                "right_content": {"paragraphs": text_for_opener}
            })
            chapter_templates.append("T1")

            if has_bg_image and paras:
                # --- Scene 2: The Visual Anchor (T7) ---
                text_chunk_right = []
                current_lines = 0
                max_lines = 28 # 이미지 옆은 여백을 위해 줄 수 타이트하게 제한
                
                while paras and current_lines < max_lines:
                    p_lines = self.estimate_lines(paras[0])
                    if current_lines + p_lines <= max_lines:
                        text_chunk_right.append(paras.pop(0))
                        current_lines += p_lines
                    else:
                        break
                
                spreads.append({
                    "type": "T7",
                    "chapter_num": chapter_num,
                    "chapter_title": chapter['chapter_title'],
                    "images": chapter['images'],
                    "left_content": {},
                    "right_content": {"paragraphs": text_chunk_right}
                })
                chapter_templates.append("T7")

            # --- Step 3: Fill remaining content (Deterministic Text Layout) ---
            while paras:
                max_lines_total = 64
                collected_paras = []
                total_lines = 0
                
                while paras and total_lines < max_lines_total:
                    p_lines = self.estimate_lines(paras[0])
                    if total_lines + p_lines <= max_lines_total:
                        collected_paras.append(paras.pop(0))
                        total_lines += p_lines
                    else:
                        break
                
                if not collected_paras:
                    break

                # Determistic 분배 (무조건 최소 좌측 1문단 보장)
                if len(collected_paras) == 1:
                    # 1단락이면 무조건 왼쪽에 몰아넣어 여백의 미 살림
                    text_chunk_left = collected_paras
                    text_chunk_right = []
                    selected_type = "T5"
                else:
                    mid_line = total_lines / 2
                    accum = 0
                    split_idx = 1
                    for idx, p in enumerate(collected_paras):
                        accum += self.estimate_lines(p)
                        if accum >= mid_line and idx > 0:
                            split_idx = idx + 1
                            break
                    
                    if split_idx >= len(collected_paras): split_idx = len(collected_paras) - 1
                    if split_idx < 1: split_idx = 1
                    
                    text_chunk_left = collected_paras[:split_idx]
                    text_chunk_right = collected_paras[split_idx:]
                    selected_type = "T2" if len(collected_paras) >= 4 else "T5"

                spreads.append({
                    "type": selected_type,
                    "chapter_num": chapter_num,
                    "chapter_title": chapter['chapter_title'],
                    "images": chapter['images'],
                    "left_content": {"paragraphs": text_chunk_left},
                    "right_content": {"paragraphs": text_chunk_right}
                })
                chapter_templates.append(selected_type)

                # 감정 쉼표 삽입 룰: 글이 연속 2번 꽉찼거나, T2/T5 연속이면 휴식 부여
                if paras and len(chapter_templates) >= 3 and chapter_templates[-1] in ["T2", "T5"] and chapter_templates[-2] in ["T2", "T5"]:
                    
                    # 추출된 Quote가 있으면 우선 사용, 없으면 본문 첫 문단에서 차용
                    if chapter.get('quote'):
                        quote_text = chapter['quote']
                        chapter['quote'] = "" # 소비함
                    else:
                        quote_text = paras.pop(0)[:150] + "..." if paras and len(paras[0]) > 150 else (paras.pop(0) if paras else "기억은 오래도록 머뭅니다.")
                        
                    spreads.append({
                        "type": "T8",
                        "chapter_num": chapter_num,
                        "chapter_title": chapter['chapter_title'],
                        "images": chapter['images'],
                        "quote_text": quote_text,
                        "left_content": {},
                        "right_content": {}
                    })
                    chapter_templates.append("T8")

            # --- Step 4: Flush remaining Quote at the end of Chapter ---
            if chapter.get('quote'):
                spreads.append({
                    "type": "T8",
                    "chapter_num": chapter_num,
                    "chapter_title": chapter['chapter_title'],
                    "images": chapter['images'],
                    "quote_text": chapter['quote'],
                    "left_content": {},
                    "right_content": {}
                })
                chapter_templates.append("T8")
                chapter['quote'] = ""

            print(f"LAYOUT: Chapter {chapter_num} ({chapter.get('mood','')}) -> {' -> '.join(chapter_templates)}")

        return {"title": data.get('title', '자서전'), "spreads": spreads}

    def generate_premium_pdf(self, markdown_content: str, output_path: str):
        raw_data = self.parse_markdown_content(markdown_content)
        spread_data = self.paginate_to_spreads(raw_data)

        # spread-based Template (304x225mm)
        html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Myeongjo:wght@400;700&display=swap');

        @page {
            size: 304mm 225mm;
            margin: 0;
        }

        body {
            font-family: 'Nanum Myeongjo', serif;
            font-size: 10.5pt;
            line-height: 1.6;
            color: #222;
            margin: 0;
            padding: 0;
        }

        .spread {
            width: 304mm;
            height: 225mm;
            display: flex;
            background-color: white;
            position: relative;
            overflow: hidden;
            break-after: page;
        }

        /* Gutter (Gradients at center) */
        .spread::after {
            content: "";
            position: absolute;
            left: 152mm;
            top: 0;
            bottom: 0;
            width: 12mm;
            transform: translateX(-50%);
            background: linear-gradient(to right, rgba(0,0,0,0.01), rgba(0,0,0,0.06) 50%, rgba(0,0,0,0.01));
            z-index: 100;
            pointer-events: none;
        }

        .page {
            width: 152mm;
            height: 225mm;
            box-sizing: border-box;
            position: relative;
            display: flex;
            flex-direction: column;
            padding: 20mm 15mm 20mm 25mm; 
        }

        .page.left {
            padding: 22mm 25mm 22mm 18mm; /* Inside(Right) 25mm, Outside(Left) 18mm */
        }
        .page.right {
            padding: 22mm 18mm 22mm 25mm; /* Inside(Left) 25mm, Outside(Right) 18mm */
        }

        /* --- Header / Footer --- */
        .header {
            font-size: 8.5pt;
            color: #aaa;
            margin-bottom: 8mm;
            display: flex;
            justify-content: space-between;
            border-bottom: 0.3pt solid #eee;
            padding-bottom: 2mm;
        }
        .footer {
            position: absolute;
            bottom: 12mm;
            font-size: 9pt;
            color: #888;
        }
        .page.left .footer { left: 18mm; }
        .page.right .footer { right: 18mm; }

        /* --- Typography --- */
        p {
            margin: 0 0 1.2em 0;
            text-indent: 1em;
            text-align: justify;
            word-break: keep-all;
        }
        p:first-of-type { text-indent: 0; }

        .lead-para {
            font-size: 1.15em;
            font-weight: 500;
            line-height: 1.7;
            color: #000;
            margin-bottom: 2em;
        }

        /* --- Template 1: Chapter Opener --- */
        .t1-opener {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            background-color: #fafafa;
            width: 304mm;
        }
        
        .t1-background {
            position: absolute;
            inset: 0;
            background-size: cover;
            background-position: center;
            z-index: 1;
        }
        
        .t1-overlay {
            position: absolute;
            inset: 0;
            z-index: 2;
        }
        
        .t1-opener.no-image {
            background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
        }
        .t1-opener.no-image .t1-background { display: none; }
        .t1-opener.no-image .t1-overlay {
            background: linear-gradient(to right, rgba(0,0,0,0.03) 0%, transparent 152mm, rgba(0,0,0,0.01) 100%);
        }

        .t1-typography {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding: 30mm 15mm;
            position: relative;
            z-index: 3;
            max-width: 122mm;
        }

        .t1-opener.theme-light { color: white; }
        .t1-opener.theme-light .t1-overlay {
            background: linear-gradient(to top, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.3) 45%, transparent 100%);
        }
        
        .t1-opener.theme-dark { color: #111; }
        .t1-opener.theme-dark.with-image .t1-overlay {
            background: linear-gradient(to top, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0.2) 60%, transparent 100%);
            backdrop-filter: blur(3px);
        }

        .chapter-num {
            font-size: 14pt;
            letter-spacing: 12px;
            margin-bottom: 5mm;
            text-transform: uppercase;
            font-weight: 300;
            opacity: 0.9;
        }
        .title-accent {
            width: 15mm;
            height: 2pt;
            background: currentColor;
            margin-bottom: 8mm;
        }
        .chapter-title {
            font-weight: 800;
            line-height: 1.3;
            word-break: keep-all; 
        }
        .chapter-sub {
            font-size: 10pt;
            letter-spacing: 6px;
            margin-top: 12mm;
            opacity: 0.7;
            text-transform: uppercase;
        }

        .image-full {
             width: calc(100% + 43mm);
            height: calc(100% + 44mm);
            margin: -22mm -25mm -22mm -18mm;
            object-fit: cover;
        }
        
        .t7-image {
            width: calc(100% + 40mm);
            height: calc(100% + 44mm);
            margin: -22mm -25mm -22mm -18mm;
            object-fit: cover;
        }

        .image-inline {
            width: 100%;
            height: auto;
            max-height: 80mm;
            object-fit: cover;
            margin: 10mm 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        .caption {
            font-size: 8.5pt;
            font-style: italic;
            color: #999;
            text-align: center;
            margin-top: 3mm;
        }

        .t8-quote-box {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 30mm 15mm;
            background-color: #fdfdfd;
        }
        .t8-quote-text {
            font-size: 20pt;
            color: #111;
            line-height: 1.8;
            text-align: center;
            font-weight: 700;
            position: relative;
            font-style: italic;
            word-break: keep-all;
        }
        .t8-quote-text::before {
            content: "“";
            font-size: 80pt;
            color: #eee;
            position: absolute;
            top: -30mm;
            left: 50%;
            transform: translateX(-50%);
        }
    </style>
</head>
<body>
    {% for spread in spreads %}
    <div class="spread">
        <div class="page left">
            <div class="header">
                <span>{{ title }}</span>
                <span></span>
            </div>
            <div class="content" style="flex: 1; position: relative;">
                {% if spread.type == 'T1' %}
                    <div class="t1-opener theme-{{ spread.title_theme }} {% if spread.has_bg_image %}with-image{% else %}no-image{% endif %}">
                        <div class="t1-background" {% if spread.has_bg_image %}style="background-image: url('{{ spread.images[0] }}');"{% endif %}></div>
                        <div class="t1-overlay"></div>
                        <div class="t1-typography">
                            <div class="chapter-num">Chapter {{ spread.chapter_num }}</div>
                            <div class="title-accent"></div>
                            <div class="chapter-title" style="font-size: {{ spread.title_size }};">{{ spread.chapter_title }}</div>
                            <div class="chapter-sub">Record of a Beautiful Life</div>
                        </div>
                    </div>
                {% elif spread.type == 'T3' %}
                    <img src="{{ spread.images[0] }}" class="image-full" alt="Memory">
                {% elif spread.type == 'T7' %}
                    <img src="{{ spread.images[0] }}" class="t7-image">
                {% elif spread.type == 'T8' %}
                    <div class="t8-quote-box">
                        <div class="t8-quote-text">{{ spread.quote_text }}</div>
                    </div>
                {% else %}
                    {% if spread.left_content.paragraphs %}
                        {% for p in spread.left_content.paragraphs %}
                            <p class="{% if loop.first and spread.type == 'T1' %}lead-para{% endif %}">{{ p }}</p>
                        {% endfor %}
                    {% endif %}
                {% endif %}
            </div>
            <div class="footer">{{ loop.index * 2 - 1 }}</div>
        </div>

        <div class="page right">
            <div class="header">
                <span></span>
                <span>{{ spread.chapter_title }}</span>
            </div>
            <div class="content" style="flex: 1;">
                {% if spread.right_content.paragraphs %}
                    {% for p in spread.right_content.paragraphs %}
                        <p class="{% if loop.first and spread.type == 'T1' %}lead-para{% endif %}">{{ p }}</p>
                    {% endfor %}
                {% endif %}
                {% if spread.type == 'T2' and loop.index % 3 == 0 %}
                    <div style="margin-top: 15mm;">
                        <img src="{{ spread.images[0] }}" class="image-inline">
                        <div class="caption">기억의 창 너머로 마주한 소중한 시간들.</div>
                    </div>
                {% endif %}
            </div>
            <div class="footer">{{ loop.index * 2 }}</div>
        </div>
    </div>
    {% endfor %}
    <div class="spread">
        <div class="page left" style="background-color: #fafafa;"></div>
        <div class="page right" style="background-color: #fafafa; display: flex; align-items: center; justify-content: center; text-align: center;">
            <div style="color: #bbb; font-style: italic;">기록된 삶은 잊히지 않는 역사가 됩니다.<br><br>— THE END —</div>
        </div>
    </div>
</body>
</html>
"""
        template = Template(html_template)
        rendered_html = template.render(
            title=spread_data['title'], 
            spreads=spread_data['spreads']
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Render PDF
        HTML(string=rendered_html, base_url=".").write_pdf(target=output_path)
        return output_path

pdf_service = PdfService()
