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

from database.content import Resource, Chapter, Topic, Question


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
            "exercise_questions": [{{"question": "...", "level": "Easy/Medium/Hard", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "correct_answer": "...", "explanation": "..."}}]

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
    return safe_json_loads(response.choices[0].message.content)


# =========================
# MERGE STEP (NO LLM)
# =========================

def merge_topics(chunk_results: list[dict]) -> dict:
    # merged maps topic name -> dict with sets for each field to deduplicate
    merged = defaultdict(lambda: {
        "main_points": set(),
        "simplified_explanation": set(),
        "questions": set(),
    })
    print("Merging topics from chunk results... ", chunk_results)
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
            for q in topic.get("questions", []):
                print(q)
                if isinstance(q, dict):
                    question_text = str(q.get("question", "")).strip()
                    level = str(q.get("level", "")).strip()
                    correct_answer = str(q.get("correct_answer", "")).strip()
                    explanation = str(q.get("explanation", "")).strip()
                    options = q.get("options", {})
                    if q.get("options") is None or q.get("options") == {}:
                        qs_type = "Theory"
                    else:
                        qs_type = "MCQ"

                    merged[name]["questions"].add((question_text, level, correct_answer, explanation, options, qs_type))

                elif isinstance(q, (list, tuple)) and len(q) >= 2:
                    merged[name]["questions"].add((str(q[0]).strip(), str(q[1]).strip()))
                else:
                    # fallback: treat the whole item as the question with unknown level
                    merged[name]["questions"].add((str(q).strip(), ""))

            print("The qs are: ", merged[name]["questions"])

    topics_out = []
    for name, buckets in merged.items():
        main_points = sorted(buckets["main_points"]) if buckets["main_points"] else []
        simplified = sorted(buckets["simplified_explanation"]) if buckets["simplified_explanation"] else []

        # convert exercise question tuples back to list of dicts
        exqs = []
        for q_text, q_level, q_ca, q_explanation, q_options, q_type in sorted(buckets["questions"], key=lambda x: x[0]):
            exqs.append({
                "question": q_text,
                "level": q_level,
                "correct_answer": q_ca,
                "explanation": q_explanation,
                "options": q_options,
                "question_type": q_type
            })

        topics_out.append({
            "topic": name.title(),
            "main_points": main_points,
            "simplified_explanation": simplified,
            "exercise_questions": exqs,
        })

    # # optional debug
    # for t in topics_out:
    #     print("Merged topic:", t["topic"], "points:", len(t["main_points"]), "exercises:", len(t["exercise_questions"]))

    return {"topics": topics_out}


# =========================
# LLM – REDUCE STEP
# =========================

def synthesize_final_chapter(merged_topics: dict) -> dict:

    # Build the prompt with a static template (no f-string interpolation of
    # the JSON example) and concatenate the actual JSON dump afterwards.
    prompt_template = """
        You are an expert NCERT textbook analyst.

        Given the following chapter text, do the following:
        1. Identify major topics in the chapter.
        2. For each topic, give me exam-relevant main points.
        3. Keep language simple and factual.
        4. Generate a simplified explanation for each main point. Add analogies that can be understood by a 10-year-old.
          5. Extract the exercise questions at the end of the chapter and mark them as Easy, Medium, or Hard. You have to assign difficulty level on your own.
          6. Additionally, generate 20 extra multiple-choice questions (total across the chapter): 6 Easy, 8 Medium, and 6 Hard. IMPORTANT: these generated questions must be attached to the relevant topic — i.e., for each topic in the "topics" list, include the generated practice questions inside that topic's "questions" (or "exercise_questions") array. Distribute the generated questions as evenly as possible across the topics; if perfect evenness isn't possible, keep distribution balanced and explain the distribution in a short comment field on the chapter-level output (optional).
              - Each generated question must be in MCQ format with exactly 4 options labeled A/B/C/D, a single correct option key (A/B/C/D) in "correct_answer", and a short explanation.
          7. Include both the extracted (from text) and the additionally generated questions inside each topic's "questions" array (do NOT put generated questions in a separate top-level list).
        8. DO NOT hallucinate content not present in the text.

        Return the response strictly in JSON with this format:

        {
        "chapter_summary": "<5-10 sentence summary>",
        "topics": [
            {
                "topic": "<topic name>",
                "main_points": ["point 1", "point 2", "point 3", "point 4", "point 5", "point 6"],
                "simplified_explanation": ["point 1", "point 2", "point 3", "point 4", "point 5", "point 6"],
                "questions": [{"question": "...", "level": "Easy/Medium/Hard", "options": {}, "correct_answer": "...", "explanation": "..."}]
            }
        ]
        }

    Extracted topics below:
    """

    # Concatenate the JSON dump separately to avoid f-string brace parsing
    try:
        merged_json = json.dumps(merged_topics)
    except Exception:
        # fallback to str() to ensure we always send something to the LLM
        merged_json = str(merged_topics)

    prompt = prompt_template + "\n" + '"""' + merged_json + '"""'

    print("Synthesizing final chapter...")
    response = generate_AI_response(prompt=prompt)
    print(response)
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
    result = synthesize_final_chapter(merged)
    print("The result is:", result)
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
                simplified = t.get("simplified_explanation", [])
                if main_points and isinstance(main_points[0], list):
                    content = "\n".join([str(x) for x in main_points[0]])
                else:
                    content = "\n".join([str(x) for x in main_points])

                if simplified and isinstance(simplified, list):
                    content += "\n\nSimplified Explanation:\n"
                    if isinstance(simplified[0], list):
                        content += "\n".join([str(x) for x in simplified[0]])
                    else:
                        content += "\n".join([str(x) for x in simplified])

                top = Topic.objects.create(
                    chapter=chapter,
                    topic_name=t.get("topic") or f"Topic {idx+1}",
                    topic_content=content,
                    simplified_content="\n".join([str(x) for x in simplified]) if simplified and isinstance(simplified, list) else None,
                    order=idx
                )
                created_topics.append(str(top.id))

                # Persist exercise questions (if any) into the Question table.
                # The reducer may place questions under "exercise_questions" or "questions" —
                # accept both keys for compatibility.
                for q in (t.get("exercise_questions") or t.get("questions") or []):
                    print(q)
                    try:
                        if isinstance(q, dict):
                            q_text = str(q.get("question") or q.get("question_text") or "").strip()
                            q_level_raw = str(q.get("level", "")).strip()
                        elif isinstance(q, (list, tuple)) and len(q) >= 2:
                            q_text = str(q[0]).strip()
                            q_level_raw = str(q[1]).strip()
                        else:
                            q_text = str(q).strip()
                            q_level_raw = ""

                        ql = q_level_raw.lower()
                        if "easy" in ql:
                            q_level = Question.EASY
                        elif "hard" in ql:
                            q_level = Question.HARD
                        else:
                            q_level = Question.MEDIUM
                        
                        qca = str(q.get("correct_answer") or "").strip() if isinstance(q, dict) else ""
                        qe = str(q.get("explanation") or "").strip() if isinstance(q, dict) else ""
                        qo = q.get("options") if isinstance(q, dict) else {}
                        if qo is None:
                            qo = {}

                        Question.objects.create(
                            chapter=chapter,
                            topic=top,
                            question_level=q_level,
                            question_text=q_text,
                            options=qo,
                            correct_answer=qca,
                            explanation=qe,
                        )
                    except Exception as _qe:
                        print(f"Failed to save question for topic {top.id}: {_qe}")

            persisted = {
                "resource_id": str(resource.id) if resource else None,
                "chapter_id": str(chapter.id),
                "topics": created_topics,
            }
    except Exception as e:
        print(f"Error saving to DB for {pdf_path}: {e}")

    # Return both AI result and the extracted full text plus persisted ids
    return {"ai": result, "full_text": full_text, "persisted": persisted}

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