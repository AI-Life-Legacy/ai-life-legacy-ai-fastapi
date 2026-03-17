import re
import os
from pathlib import Path
from jinja2 import Template
from weasyprint import HTML, CSS

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(r"C:\Users\rmswn\.gemini\antigravity\brain\cab725da-ae9e-4419-8986-24a87a05bd80")

# Emotional assets (generated previously)
IMAGE_ASSETS = [
    str(ARTIFACT_DIR / "memoir_nostalgia_1_1773714327864.png"),
    str(ARTIFACT_DIR / "memoir_nostalgia_2_1773714346454.png"),
    str(ARTIFACT_DIR / "memoir_nostalgia_3_1773714360412.png"),
    str(ARTIFACT_DIR / "memoir_nostalgia_4_1773714379542.png"),
]

def parse_markdown_autobiography(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Title extraction
    title_match = re.search(r"^제목:\s*(.*)$", content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "자서전"

    # Chapter extraction (## Headers)
    chapters = []
    sections = re.split(r"^##\s+", content, flags=re.MULTILINE)
    
    for i, section in enumerate(sections[1:]):
        lines = section.strip().split('\n')
        if not lines:
            continue
        
        chapter_title = lines[0].strip()
        body = '\n'.join(lines[1:]).strip()
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
        
        # Assign an image to each chapter (cycle through assets)
        image_path = IMAGE_ASSETS[i % len(IMAGE_ASSETS)]
        
        chapters.append({
            "chapter_title": chapter_title,
            "paragraphs": paragraphs,
            "image_path": "file:///" + image_path.replace("\\", "/")
        })

    return {"title": title, "chapters": chapters}

def generate_pdf():
    # Load autobiography data
    md_path = PROJECT_ROOT / "storage" / "data" / "autobiography_output_rag.md"
    if not md_path.exists():
        print(f"Error: {md_path} not found.")
        return

    data = parse_markdown_autobiography(md_path)

    # HTML Template for Landscape 2-page Spread
    html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <style>
        @page {
            size: A4 landscape;
            margin: 0;
            @bottom-right {
                content: counter(page);
                font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
                font-size: 10pt;
                color: #888;
                margin-right: 20mm;
                margin-bottom: 10mm;
            }
        }

        body {
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #fdfdfd;
        }

        /* Container for a single A4 Landscape page (containing 2 virtual pages) */
        .spread {
            width: 297mm;
            height: 210mm;
            display: flex;
            page-break-after: always;
            overflow: hidden;
        }

        /* Left side: Image or Chapter Visual */
        .page-left {
            width: 50%;
            height: 100%;
            position: relative;
            background-color: #f5f5f5;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            border-right: 1px solid #eee;
        }

        .page-left img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: sepia(20%) brightness(95%);
        }

        .chapter-overlay {
            position: absolute;
            bottom: 20mm;
            left: 20mm;
            color: white;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
            font-size: 24pt;
            font-weight: bold;
        }

        /* Right side: Text content */
        .page-right {
            width: 50%;
            height: 100%;
            padding: 25mm 20mm;
            box-sizing: border-box;
            background-color: white;
            display: flex;
            flex-direction: column;
        }

        .chapter-title {
            font-size: 24pt;
            font-weight: bold;
            color: #222;
            margin-bottom: 15mm;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5mm;
        }

        .content {
            font-size: 12.5pt;
            line-height: 1.8;
            color: #333;
            text-align: justify;
            flex-grow: 1;
        }

        .paragraph {
            text-indent: 1em;
            margin-bottom: 6mm;
        }

        /* Cover & Intro Styles */
        .full-page-cover {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: #2c3e50;
            color: white;
            text-align: center;
        }

        .cover-title {
            font-size: 48pt;
            margin-bottom: 10mm;
        }
        
        .cover-author {
            font-size: 20pt;
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <!-- Virtual Spread: Cover -->
    <div class="spread">
        <div class="page-left" style="background-color: #2c3e50;">
            <div style="color: white; font-style: italic; font-size: 18pt; opacity: 0.7;">인생의 소중한 기록</div>
        </div>
        <div class="page-right" style="background-color: #2c3e50; color: white; justify-content: center; align-items: center; border: none;">
            <div style="font-size: 42pt; font-weight: bold; margin-bottom: 20px;">{{ title }}</div>
            <div style="font-size: 18pt; opacity: 0.9;">저자: {{ title.split('의')[0] }}</div>
        </div>
    </div>

    <!-- Chapter Spreads -->
    {% for chapter in chapters %}
    <div class="spread">
        <div class="page-left">
            <img src="{{ chapter.image_path }}" alt="Nostalgic Scene">
            <div class="chapter-overlay">{{ loop.index }}. {{ chapter.chapter_title }}</div>
        </div>
        <div class="page-right">
            <div class="chapter-title">{{ chapter.chapter_title }}</div>
            <div class="content">
                {% for para in chapter.paragraphs %}
                <p class="paragraph">{{ para }}</p>
                {% endfor %}
            </div>
        </div>
    </div>
    {% endfor %}

    <!-- Final Page -->
    <div class="spread">
        <div class="page-left" style="background-color: #f9f9f9;">
            <div style="color: #999; font-size: 14pt;">마침</div>
        </div>
        <div class="page-right" style="justify-content: center; align-items: center;">
            <p style="font-style: italic; color: #666; font-size: 16pt; text-align: center;">기억은 사라지지만,<br>기록은 영원히 남습니다.</p>
        </div>
    </div>
</body>
</html>
"""

    template = Template(html_template)
    rendered_html = template.render(title=data['title'], chapters=data['chapters'])

    output_path = PROJECT_ROOT / "storage" / "data" / "autobiography_book.pdf"
    
    # Generate PDF (Allowing remote/local files for images)
    HTML(string=rendered_html, base_url=".").write_pdf(target=output_path)
    
    print(f"Book-style PDF successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_pdf()
