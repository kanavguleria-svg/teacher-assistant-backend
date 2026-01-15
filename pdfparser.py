import zipfile
import os
import json
import re
import time
import hashlib
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
import openai


# =========================
# ENV + CLIENT
# =========================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# RATE LIMIT BACKOFF
# =========================

def call_with_backoff(func, *args, **kwargs):
    for attempt in range(6):
        try:
            return func(*args, **kwargs)
        except openai.RateLimitError as e:
            wait = min(2 ** attempt, 60)
            print(f"[RATE LIMIT] Sleeping {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Rate limit retries exhausted")


# =========================
# JSON SAFETY
# =========================

def safe_json_loads(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM output")
    return json.loads(match.group())


# =========================
# CACHING (CRITICAL)
# =========================

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def load_cache(key: str):
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None

def save_cache(key: str, data: dict):
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(data))


# =========================
# CHUNKING
# =========================

def chunk_text_by_words(text: str, max_words: int = 3000) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


# =========================
# LLM – MAP STEP
# =========================

def extract_topics_from_chunk(chunk_text: str) -> dict:
    prompt = f"""
        Extract topics and exam-relevant points from the text below.

        Rules:
        - Only use information present
        - 7-12 points per topic
        - JSON only

        Format:
        {{
        "topics": [
            {{
            "topic": "...",
            "main_points": ["...", "..."]
            }}
        ]
        }}

    Text:
    \"\"\"{chunk_text}\"\"\"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You extract structured textbook facts."},
            {"role": "user", "content": prompt},
        ],
    )

    return safe_json_loads(response.choices[0].message.content)


# =========================
# MERGE STEP (NO LLM)
# =========================

def merge_topics(chunk_results: list[dict]) -> dict:
    merged = defaultdict(set)

    for result in chunk_results:
        for topic in result.get("topics", []):
            name = topic["topic"].strip().lower()
            for point in topic["main_points"]:
                merged[name].add(point.strip())

    return {
        "topics": [
            {
                "topic": name.title(),
                "main_points": sorted(points)
            }
            for name, points in merged.items()
        ]
    }


# =========================
# LLM – REDUCE STEP
# =========================

def synthesize_final_chapter(merged_topics: dict) -> dict:
    prompt = f"""
        You are an expert NCERT textbook analyst.

        Given the following chapter text, do the following:
        1. Identify major topics in the chapter.
        2. For each topic, give me exam-relevant main points.
        3. Keep language simple and factual.
        4. Do NOT hallucinate content not present in the text.

        Return the response strictly in JSON with this format:

        {{
        "chapter_summary": "<5-10 sentence summary>",
        "topics": [
            {{
            "topic": "<topic name>",
            "main_points": ["point 1", "point 2", "point 3", "point 4", "point 5", "point 6"]
            }}
        ]
        }}


    Extracted topics:
    \"\"\"{json.dumps(merged_topics)}\"\"\"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You finalize educational summaries."},
            {"role": "user", "content": prompt},
        ],
    )

    return safe_json_loads(response.choices[0].message.content)


# =========================
# PDF PIPELINE
# =========================

def extract_pdf(pdf_path: Path) -> dict:
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        if page.extract_text():
            full_text += page.extract_text() + "\n"

    if not full_text.strip():
        raise ValueError("No text extracted")

    chunks = chunk_text_by_words(full_text)
    print(f"Chunks: {len(chunks)}")

    chunk_results = []

    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}")

        key = cache_key(chunk)
        cached = load_cache(key)

        if cached:
            chunk_results.append(cached)
            continue

        result = call_with_backoff(extract_topics_from_chunk, chunk)
        save_cache(key, result)
        chunk_results.append(result)

        time.sleep(3)  # soft throttle

    merged = merge_topics(chunk_results)
    return synthesize_final_chapter(merged)


# =========================
# ZIP PIPELINE
# =========================

def parse_chapters_from_local_zip(zip_path: str, standard: int, subject: str):
    base = Path(f"./e-book/{standard}/{subject}")
    base.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        pdfs = sorted(
            [p for p in archive.infolist() if p.filename.lower().endswith(".pdf")],
            key=lambda x: x.filename
        )

        for pdf in pdfs:
            print(f"\n--- Processing {pdf.filename} ---")
            extracted = archive.extract(pdf, path=base)
            result = extract_pdf(Path(extracted))
            print(json.dumps(result, indent=2))


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    parse_chapters_from_local_zip(
        zip_path="./maths.zip",
        standard=10,
        subject="Maths",
    )
