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
import django
from django.db import transaction

# Ensure Django environment is configured so we can import models and save results
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from database.content import Resource, Chapter, Topic


# =========================
# ENV + CLIENT
# =========================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("OpenAI API Key Loaded:", bool(os.getenv("OPENAI_API_KEY")))
# =========================
# RATE LIMIT BACKOFF
# =========================

def call_with_backoff(func, *args, **kwargs):
    print("Calling with backoff...")
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
    print("Reaching safe_json_loads...")
    try:
        print("Rendering safe JSON loads...")
        text = re.sub(r"```json|```", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM output")
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        print("Exception in safe_json_loads:", e)
        raise ValueError(f"JSON decode error: {e}")

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
    # Use doubled braces for literal JSON example inside an f-string so
    # Python doesn't try to interpret them as formatting fields.
    prompt = f"""
        Extract topics, education-relevant points and exam-relevant (knowledge-relevant) questions from the text below.

        Rules:
        - Only use information present
        - 4-7 correct points per topic, no fluff
        - JSON only

        Format:
        {{
        "topics": [
            {{
            "topic": "...",
            "main_points": ["...", "..."],
            "simplified_explanation": ["...", "..."],
            "exercise_questions": [{{"question": "...", "level": "Easy/Medium/Hard"}}]
            }}
        ]
        }}

    Text:
    \"\"\"{chunk_text}\"\"\"
    """
    print("Extracting topics from chunk...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": "You extract structured textbook facts."},
            {"role": "user", "content": prompt},
        ],
    )
    print(response)
    return safe_json_loads(response.choices[0].message.content)


# =========================
# MERGE STEP (NO LLM)
# =========================

def merge_topics(chunk_results: list[dict]) -> dict:
    # merged maps topic name -> dict with sets for each field to deduplicate
    merged = defaultdict(lambda: {
        "main_points": set(),
        "simplified_explanation": set(),
        "exercise_questions": set(),
    })

    for result in chunk_results:
        for topic in result.get("topics", []):
            name = topic.get("topic") or "Untitled"

            # main points
            for point in topic.get("main_points", []):
                if point is None:
                    continue
                merged[name]["main_points"].add(str(point).strip())

            # simplified explanations
            for point in topic.get("simplified_explanation", []):
                if point is None:
                    continue
                merged[name]["simplified_explanation"].add(str(point).strip())

            # exercise questions — normalize to tuples (question, level)
            for q in topic.get("exercise_questions", []):
                if isinstance(q, dict):
                    question_text = str(q.get("question", "")).strip()
                    level = str(q.get("level", "")).strip()
                    merged[name]["exercise_questions"].add((question_text, level))
                elif isinstance(q, (list, tuple)) and len(q) >= 2:
                    merged[name]["exercise_questions"].add((str(q[0]).strip(), str(q[1]).strip()))
                else:
                    # fallback: treat the whole item as the question with unknown level
                    merged[name]["exercise_questions"].add((str(q).strip(), ""))

    topics_out = []
    for name, buckets in merged.items():
        main_points = sorted(buckets["main_points"]) if buckets["main_points"] else []
        simplified = sorted(buckets["simplified_explanation"]) if buckets["simplified_explanation"] else []

        # convert exercise question tuples back to list of dicts
        exqs = []
        for q_text, q_level in sorted(buckets["exercise_questions"], key=lambda x: x[0]):
            exqs.append({"question": q_text, "level": q_level})

        topics_out.append({
            "topic": name.title(),
            "main_points": main_points,
            "simplified_explanation": simplified,
            "exercise_questions": exqs,
        })

    # optional debug
    for t in topics_out:
        print("Merged topic:", t["topic"], "points:", len(t["main_points"]), "exercises:", len(t["exercise_questions"]))

    return {"topics": topics_out}


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
        4. Generate a simplified explanation for each main point. Add analogies that can be understood by a 10-year-old.
        5. Extract the exercise questions at the end of the chapter and mark them as Easy, Medium, or Hard. Yoi have to assign difficulty level on your own
        6. From the Qs generate 6 additional practice questions (2 Easy, 2 Medium, 2 Hard) on topics and mark them as Easy, Medium, or Hard.
        7. Do NOT hallucinate content not present in the text.

        Return the response strictly in JSON with this format:

        {{
        "chapter_summary": "<5-10 sentence summary>",
        "topics": [
            {{
            "topic": "<topic name>",
            "main_points": ["point 1", "point 2", "point 3", "point 4", "point 5", "point 6"],
            "simplified_explanation": ["point 1", "point 2", "point 3", "point 4", "point 5", "point 6"],
            "exercise_questions": [{"question": "...", "level": "Easy/Medium/Hard"}]
            }}
        ]
        }}


    Extracted topics:
    \"\"\"{json.dumps(merged_topics)}\"\"\"
    """

    response = generate_AI_response(prompt=prompt)
    
    return safe_json_loads(response.choices[0].message.content)


def generate_AI_response(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You finalize educational summaries."},
            {"role": "user", "content": prompt},
        ],
    )
    
    return response

# =========================
# PDF PIPELINE
# =========================

def extract_pdf(pdf_path: Path, standard: int, subject: str) -> dict:
    print("Here we are extracting PDF:", pdf_path)
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
        print(f"Cache {'hit' if cached else 'miss'} for chunk {i}")
        # print("The cached value is:", cached)
        if cached:
            chunk_results.append(cached)
            continue

        result = call_with_backoff(extract_topics_from_chunk, chunk)
        save_cache(key, result)
        chunk_results.append(result)

        time.sleep(3)  # soft throttle

    merged = merge_topics(chunk_results)
    print("The merged value is:", merged)
    result = synthesize_final_chapter(merged)
    # Persist results to DB here — extract_pdf is the single source of truth
    ai_data = result

    # compute sha256 of file for idempotency
    try:
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
    except Exception as e:
        print(f"Error reading file for hash: {e}")
        file_hash = None

    persisted = {}
    try:
        chapter_name = Path(pdf_path).stem
        with transaction.atomic():
            resource = None
            if file_hash:
                resource, _created = Resource.objects.get_or_create(
                    file_hash=file_hash,
                    defaults={
                        "standard": standard,
                        "subject": subject,
                        "file_name": Path(pdf_path).name,
                        "source_path": str(pdf_path),
                    }
                )
            else:
                # fallback: create a resource without hash (less ideal)
                resource = Resource.objects.create(
                    standard=standard,
                    subject=subject,
                    file_name=Path(pdf_path).name,
                    source_path=str(pdf_path),
                    file_hash="",
                )

            chapter, _c = Chapter.objects.update_or_create(
                resource=resource,
                chapter_name=chapter_name,
                defaults={
                    "standard": standard,
                    "subject": subject,
                    "full_text": full_text,
                    "ai_summary": ai_data.get("chapter_summary") if isinstance(ai_data, dict) else None,
                }
            )

            # replace topics
            Topic.objects.filter(chapter=chapter).delete()
            topics = ai_data.get("topics", []) if isinstance(ai_data, dict) else []
            created_topics = []
            for idx, t in enumerate(topics):
                main_points = t.get("main_points", [])
                if main_points and isinstance(main_points[0], list):
                    content = "\n".join([str(x) for x in main_points[0]])
                else:
                    content = "\n".join([str(x) for x in main_points])

                top = Topic.objects.create(
                    chapter=chapter,
                    topic_name=t.get("topic") or f"Topic {idx+1}",
                    topic_content=content,
                    order=idx
                )
                created_topics.append(str(top.id))

            persisted = {
                "resource_id": str(resource.id) if resource else None,
                "chapter_id": str(chapter.id),
                "topics": created_topics,
            }
    except Exception as e:
        print(f"Error saving to DB for {pdf_path}: {e}")

    # Return both AI result and the extracted full text plus persisted ids
    return {"ai": result, "full_text": full_text, "persisted": persisted}


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
            result_bundle = extract_pdf(Path(extracted), standard=standard, subject=subject)
            print(json.dumps(result_bundle, indent=2))
            


# =========================
# ENTRY POINT
# =========================

# if __name__ == "__main__":
#     parse_chapters_from_local_zip(
#         zip_path="./maths.zip",
#         standard=10,
#         subject="Maths",
#     )
