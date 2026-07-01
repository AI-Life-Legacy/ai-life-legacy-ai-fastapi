import os
import re
from pathlib import Path

from app.core.config import settings
from jinja2 import Template
from PIL import Image, ImageDraw, ImageStat
from weasyprint import HTML


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = Path(settings.CHROMA_DB_PATH).parent / "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

MOOD_ASSETS = {
    "childhood": str(ASSETS_DIR / "childhood.png"),
    "youth": str(ASSETS_DIR / "youth.png"),
    "career": str(ASSETS_DIR / "career.png"),
    "marriage": str(ASSETS_DIR / "marriage.png"),
    "crisis": str(ASSETS_DIR / "crisis.png"),
    "family": str(ASSETS_DIR / "family.png"),
    "hobby": str(ASSETS_DIR / "hobby.png"),
    "future": str(ASSETS_DIR / "future.png"),
}


class PdfService:
    DEFAULT_ASSET_COLORS = {
        "childhood": ("#f8d7a4", "#6b8f71", "#fef6e4"),
        "youth": ("#b7d9f7", "#456990", "#f4fbff"),
        "career": ("#d9e2ec", "#334e68", "#f8fafc"),
        "marriage": ("#ffd6dc", "#9f5f80", "#fff5f7"),
        "crisis": ("#cbd5e1", "#475569", "#f1f5f9"),
        "family": ("#f7c59f", "#7f5539", "#fff8f1"),
        "hobby": ("#c8e6c9", "#3d7a45", "#f4fff5"),
        "future": ("#d8ccff", "#5b4b8a", "#faf7ff"),
    }

    THEME_STYLES = {
        "classic": {
            "font_family": "'Malgun Gothic', 'Noto Sans KR', serif",
            "bg_color": "#fbfaf7",
            "text_color": "#24211f",
            "muted_color": "#7b7169",
            "accent_color": "#8b1a1a",
            "card_bg": "#ffffff",
        },
        "modern": {
            "font_family": "'Malgun Gothic', 'Noto Sans KR', sans-serif",
            "bg_color": "#f8fafc",
            "text_color": "#172033",
            "muted_color": "#64748b",
            "accent_color": "#0d9488",
            "card_bg": "#ffffff",
        },
        "warm": {
            "font_family": "'Malgun Gothic', 'Noto Sans KR', serif",
            "bg_color": "#fff8f1",
            "text_color": "#35261f",
            "muted_color": "#8a6d5c",
            "accent_color": "#9a5a2f",
            "card_bg": "#fffdf9",
        },
    }

    def ensure_default_assets(self):
        for mood, path in MOOD_ASSETS.items():
            image_path = Path(path)
            if image_path.exists():
                continue

            bg, accent, light = self.DEFAULT_ASSET_COLORS.get(mood, self.DEFAULT_ASSET_COLORS["family"])
            img = Image.new("RGB", (1400, 1000), bg)
            draw = ImageDraw.Draw(img, "RGBA")
            draw.rectangle((0, 0, 1400, 1000), fill=bg)
            draw.ellipse((-180, -120, 560, 620), fill=f"{light}CC")
            draw.ellipse((760, 260, 1580, 1180), fill=f"{accent}55")
            draw.polygon([(0, 1000), (1400, 760), (1400, 1000)], fill=f"{accent}66")
            draw.line((90, 820, 1260, 620), fill=f"{accent}AA", width=10)
            draw.line((120, 870, 980, 700), fill="#ffffff88", width=5)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(image_path)

    def to_file_uri(self, image_path: str) -> str:
        if not image_path:
            return ""
        if image_path.startswith("file:///"):
            return image_path
        return Path(image_path).resolve().as_uri()

    def file_uri_to_path(self, image_uri: str) -> Path:
        if image_uri.startswith("file:///"):
            return Path(image_uri.replace("file:///", ""))
        return Path(image_uri)

    def image_exists(self, image_uri: str) -> bool:
        try:
            return self.file_uri_to_path(image_uri).exists()
        except Exception:
            return False

    def get_image_luminance(self, image_uri: str) -> float:
        try:
            path = self.file_uri_to_path(image_uri)
            if not path.exists():
                return 128
            img = Image.open(path).convert("L")
            return ImageStat.Stat(img).mean[0]
        except Exception as exc:
            print(f"Warning: Luminance analysis failed for {image_uri}: {exc}")
            return 128

    def detect_mood(self, title: str, content: str) -> str:
        text = f"{title} {content}"
        mood_keywords = {
            "childhood": ["어린", "유년", "고향", "부모", "성장", "태어난"],
            "youth": ["학교", "친구", "청춘", "대학", "학생", "진로"],
            "career": ["직장", "회사", "일", "업무", "성취", "실패"],
            "marriage": ["결혼", "배우자", "가족", "아이", "자녀"],
            "crisis": ["위기", "힘들", "고비", "상처", "변화"],
            "hobby": ["취미", "여행", "운동", "그림", "음악"],
            "future": ["앞으로", "미래", "계획", "꿈", "후손"],
        }
        for mood, keywords in mood_keywords.items():
            if any(keyword in text for keyword in keywords):
                return mood
        return "family"

    def get_title_size(self, title: str) -> str:
        length = len(title)
        if length < 12:
            return "36pt"
        if length < 20:
            return "30pt"
        if length < 30:
            return "25pt"
        return "21pt"

    def estimate_lines(self, text: str) -> int:
        return max(1, len(text) // 42) + 1

    def parse_markdown_content(self, content: str):
        title_match = re.search(r"^(?:제목|Title)\s*:\s*(.*)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "자서전"
        sections = re.split(r"^##\s+", content, flags=re.MULTILINE)
        chapters = []

        for section in sections[1:]:
            lines = section.strip().split("\n")
            if not lines:
                continue

            chapter_title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            mood_match = re.search(r"<!-- MOOD:\s*(.*?)\s*-->", body)
            quote_match = re.search(r"<!-- QUOTE:\s*(.*?)\s*-->", body)
            image_match = re.search(r"<!-- IMAGE:\s*(.*?)\s*-->", body)

            mood = mood_match.group(1).strip() if mood_match else self.detect_mood(chapter_title, body)
            quote = quote_match.group(1).strip() if quote_match else ""
            image_uri = self.to_file_uri(image_match.group(1).strip()) if image_match else self.to_file_uri(MOOD_ASSETS.get(mood, MOOD_ASSETS["family"]))

            body = re.sub(r"<!-- MOOD:.*?-->", "", body).strip()
            body = re.sub(r"<!-- QUOTE:.*?-->", "", body).strip()
            body = re.sub(r"<!-- IMAGE:.*?-->", "", body).strip()
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

            chapters.append(
                {
                    "chapter_title": chapter_title,
                    "paragraphs": paragraphs,
                    "image": image_uri,
                    "has_image": self.image_exists(image_uri),
                    "title_size": self.get_title_size(chapter_title),
                    "title_theme": "dark" if self.get_image_luminance(image_uri) > 115 else "light",
                    "mood": mood,
                    "quote": quote,
                }
            )

        return {"title": title, "chapters": chapters}

    def paginate_to_spreads(self, data):
        spreads = []
        for chapter_idx, chapter in enumerate(data["chapters"]):
            chapter_num = chapter_idx + 1
            paragraphs = list(chapter["paragraphs"])
            first_para = paragraphs.pop(0) if paragraphs else ""

            spreads.append(
                {
                    "type": "opener",
                    "chapter_num": chapter_num,
                    "chapter_title": chapter["chapter_title"],
                    "title_size": chapter["title_size"],
                    "title_theme": chapter["title_theme"],
                    "image": chapter["image"],
                    "has_image": chapter["has_image"],
                    "right_paragraphs": [first_para] if first_para else [],
                    "quote": chapter["quote"],
                }
            )

            while paragraphs:
                collected = []
                lines = 0
                while paragraphs and lines < 56:
                    next_lines = self.estimate_lines(paragraphs[0])
                    if collected and lines + next_lines > 56:
                        break
                    collected.append(paragraphs.pop(0))
                    lines += next_lines

                midpoint = max(1, len(collected) // 2)
                spreads.append(
                    {
                        "type": "text",
                        "chapter_num": chapter_num,
                        "chapter_title": chapter["chapter_title"],
                        "image": chapter["image"],
                        "has_image": chapter["has_image"],
                        "left_paragraphs": collected[:midpoint],
                        "right_paragraphs": collected[midpoint:],
                    }
                )

            if chapter["quote"]:
                spreads.append(
                    {
                        "type": "quote",
                        "chapter_num": chapter_num,
                        "chapter_title": chapter["chapter_title"],
                        "image": chapter["image"],
                        "has_image": chapter["has_image"],
                        "quote": chapter["quote"],
                    }
                )

        return {"title": data["title"], "spreads": spreads}

    def generate_premium_pdf(self, markdown_content: str, output_path: str, theme: str = "classic"):
        self.ensure_default_assets()
        raw_data = self.parse_markdown_content(markdown_content)
        spread_data = self.paginate_to_spreads(raw_data)
        style = self.THEME_STYLES.get(theme, self.THEME_STYLES["classic"])

        html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <style>
    @page { size: 304mm 225mm; margin: 0; }
    body {
      margin: 0;
      font-family: {{ style.font_family }};
      color: {{ style.text_color }};
      background: {{ style.bg_color }};
    }
    .spread {
      width: 304mm;
      height: 225mm;
      display: flex;
      page-break-after: always;
      position: relative;
      overflow: hidden;
      background: {{ style.bg_color }};
    }
    .spread::after {
      content: "";
      position: absolute;
      top: 0;
      bottom: 0;
      left: 152mm;
      width: 10mm;
      transform: translateX(-50%);
      background: linear-gradient(to right, rgba(0,0,0,0.02), rgba(0,0,0,0.08), rgba(0,0,0,0.02));
      z-index: 30;
    }
    .page {
      width: 152mm;
      height: 225mm;
      box-sizing: border-box;
      position: relative;
      overflow: hidden;
      background: {{ style.bg_color }};
    }
    .page.left { padding: 22mm 25mm 20mm 18mm; }
    .page.right { padding: 22mm 18mm 20mm 25mm; }
    .header {
      height: 9mm;
      font-size: 8pt;
      color: {{ style.muted_color }};
      border-bottom: 0.3pt solid rgba(0,0,0,0.12);
      margin-bottom: 9mm;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    .footer {
      position: absolute;
      bottom: 10mm;
      color: {{ style.muted_color }};
      font-size: 8pt;
    }
    .left .footer { left: 18mm; }
    .right .footer { right: 18mm; }
    p {
      font-size: 10.8pt;
      line-height: 1.72;
      margin: 0 0 1.15em 0;
      text-align: justify;
      word-break: keep-all;
    }
    .lead p:first-child {
      font-size: 12pt;
      line-height: 1.82;
      font-weight: 600;
    }
    .opener {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      left: 0;
      background: {{ style.card_bg }};
    }
    .opener-image {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .opener-overlay {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      left: 0;
    }
    .theme-light .opener-overlay {
      background: linear-gradient(to top, rgba(0,0,0,0.82), rgba(0,0,0,0.34), rgba(0,0,0,0.08));
    }
    .theme-dark .opener-overlay {
      background: linear-gradient(to top, rgba(255,255,255,0.82), rgba(255,255,255,0.32), rgba(255,255,255,0.06));
    }
    .opener-copy {
      position: absolute;
      left: 24mm;
      bottom: 30mm;
      width: 108mm;
      z-index: 2;
    }
    .chapter-kicker {
      color: {{ style.accent_color }};
      font-size: 12pt;
      font-weight: 700;
      letter-spacing: 6px;
      margin-bottom: 6mm;
      text-transform: uppercase;
    }
    .chapter-title {
      line-height: 1.25;
      font-weight: 800;
      word-break: keep-all;
    }
    .theme-light .chapter-title,
    .theme-light .chapter-subtitle { color: white; }
    .chapter-rule {
      width: 22mm;
      height: 2pt;
      background: {{ style.accent_color }};
      margin: 0 0 7mm 0;
    }
    .chapter-subtitle {
      margin-top: 8mm;
      font-size: 9pt;
      letter-spacing: 3px;
      color: {{ style.muted_color }};
    }
    .chapter-image {
      width: calc(100% + 43mm);
      height: 62mm;
      object-fit: cover;
      display: block;
      margin: 0 -18mm 9mm -25mm;
      border-bottom: 3pt solid {{ style.accent_color }};
    }
    .quote-box {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      left: 0;
      background: {{ style.card_bg }};
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 30mm;
      box-sizing: border-box;
    }
    .quote-image {
      position: absolute;
      top: 0;
      right: 0;
      bottom: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      opacity: 0.14;
    }
    .quote-text {
      position: relative;
      z-index: 2;
      font-size: 20pt;
      line-height: 1.7;
      text-align: center;
      font-weight: 800;
      color: {{ style.text_color }};
      word-break: keep-all;
    }
    .ending {
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      font-size: 14pt;
      color: {{ style.muted_color }};
      background: {{ style.card_bg }};
    }
  </style>
</head>
<body>
{% for spread in spreads %}
  <div class="spread">
    <div class="page left">
      {% if spread.type == "opener" %}
        <div class="opener theme-{{ spread.title_theme }}">
          {% if spread.has_image %}
            <img class="opener-image" src="{{ spread.image }}" alt="">
          {% endif %}
          <div class="opener-overlay"></div>
          <div class="opener-copy">
            <div class="chapter-kicker">Chapter {{ spread.chapter_num }}</div>
            <div class="chapter-rule"></div>
            <div class="chapter-title" style="font-size: {{ spread.title_size }};">{{ spread.chapter_title }}</div>
            <div class="chapter-subtitle">Life Legacy Autobiography</div>
          </div>
        </div>
      {% elif spread.type == "quote" %}
        <div class="quote-box">
          {% if spread.has_image %}<img class="quote-image" src="{{ spread.image }}" alt="">{% endif %}
          <div class="quote-text">{{ spread.quote }}</div>
        </div>
      {% else %}
        <div class="header"><span>{{ title }}</span><span></span></div>
        {% if spread.has_image %}<img class="chapter-image" src="{{ spread.image }}" alt="">{% endif %}
        {% for p in spread.left_paragraphs %}
          <p>{{ p }}</p>
        {% endfor %}
        <div class="footer">{{ loop.index * 2 - 1 }}</div>
      {% endif %}
    </div>
    <div class="page right">
      <div class="header"><span></span><span>{{ spread.chapter_title }}</span></div>
      <div class="{% if spread.type == 'opener' %}lead{% endif %}">
        {% for p in spread.right_paragraphs %}
          <p>{{ p }}</p>
        {% endfor %}
      </div>
      <div class="footer">{{ loop.index * 2 }}</div>
    </div>
  </div>
{% endfor %}
  <div class="spread">
    <div class="page left ending"></div>
    <div class="page right ending">
      <div>기록은 삶을 오래 남기는 또 하나의 방식입니다.<br><br>THE END</div>
    </div>
  </div>
</body>
</html>
"""
        rendered_html = Template(html_template).render(
            title=spread_data["title"],
            spreads=spread_data["spreads"],
            style=style,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        html = HTML(string=rendered_html, base_url=str(PROJECT_ROOT))
        html.write_pdf(target=output_path)
        page_count = len(html.render().pages)
        return output_path, page_count


pdf_service = PdfService()
