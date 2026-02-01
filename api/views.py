from database.content import Chapter, Topic, Question, Chat
from django.core.cache import cache
import json

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
import os
import tempfile
from pdfparser import parse_chapters_from_local_zip, extract_pdf, generate_AI_response
import time
import random
from openai import OpenAI


def _split_topic_content(raw: str) -> list:
    """Convert stored topic content into a list of strings.

    Handles both actual newlines and escaped '\\n' sequences, trims empty
    lines and whitespace.
    """
    if not raw:
        return []
    # If the content contains literal backslash-n sequences, convert them
    normalized = raw.replace('\\n', '\n')
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    return lines

@require_http_methods(["GET"])
def home(request):
    """Basic home endpoint."""
    return JsonResponse({
        'message': 'Welcome to Vidya Setu API',
        'status': 'success'
    })


@require_http_methods(["GET"])
def health(request):
    """Health check endpoint."""
    return JsonResponse({
        'status': 'healthy',
        'service': 'teacher-assistant-backend'
    })

@csrf_exempt  # allow Postman or non-browser clients to POST without CSRF token
@require_http_methods(["POST"])
def upload_and_parse_pdf(request):

    uploaded_file = request.FILES.get("file")
    subject = request.POST.get("subject")
    standard = request.POST.get("standard")

    print("the uploaded file is", uploaded_file)
    if not uploaded_file:
        return JsonResponse({"error": "Please upload a file"}, status=400)

    if not subject or not standard:
        return JsonResponse({"error": "Subject and standard are required"}, status=400)

    try:
        standard = int(standard)
    except (ValueError, TypeError):
        return JsonResponse({"error": "The standard should be an int value"}, status=400)

    filename = uploaded_file.name.lower()
    print("the filename is", filename)
    result = None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        print("the temp path is", temp_path)
        with open(temp_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        if filename.endswith(".zip"):
            # If you want to enable actual processing, uncomment and handle exceptions
            try:
                result = parse_chapters_from_local_zip(
                    zip_path=temp_path,
                    standard=standard,
                    subject=subject,
                )
            except Exception as e:
                return JsonResponse({"error": f"Failed to process ZIP: {str(e)}"}, status=500)

        elif filename.endswith(".pdf"):
            print("Here" + " we are about to extract PDF")
            try:
                result = extract_pdf(
                    pdf_path=temp_path,
                    standard=standard,
                    subject=subject,
                )
                # print(result)

            except Exception as e:
                return JsonResponse({"error": f"Failed to process PDF: {str(e)}"}, status=500)
            # return JsonResponse({"message": "PDF received (parsing disabled in code)"}, status=200)

        else:
            return JsonResponse({"error": "Only PDF or ZIP files are supported"}, status=400)

    # If processing logic sets `result`, return it; otherwise return a generic success
    return JsonResponse(
        {
            "message" : "Params successfully iterated",
            "subject" : subject,
            "standard" : standard
        },
        status = 200,
        json_dumps_params={
            "indent":2 
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def get_chapter_details(request):
    # Try to parse JSON body first (clients send application/json).
    # If parsing fails or field missing, fall back to form data.
    chapter_id = None
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    chapter_id = payload.get("chapter_id") or request.POST.get("chapter_id")

    if not chapter_id:
        return JsonResponse({"error": "Please provide chapter_id"}, status=400)
    
    try:
        chapter = Chapter.objects.get(id=chapter_id)
        # Return topics with their original content and any simplified content
        topics = Topic.objects.filter(chapter=chapter).order_by("order", "created_at")

        topics_data = [
            {
                "id": str(topic.id),
                "topic_name": topic.topic_name,
                "content_summary": _split_topic_content(getattr(topic, "topic_content", None)),
                "simplified_content": [topic.simplified_content] or [],
            }
            for topic in topics
        ]

        chapter_data = {
            "id": str(chapter.id),
            "chapter_name": chapter.chapter_name,
            "chapter_number": chapter.chapter_number,
            "standard": chapter.standard,
            "subject": chapter.subject,
            # "full_text": chapter.full_text,
            "topics": topics_data,
        }

        return JsonResponse({"chapter": chapter_data}, status=200, json_dumps_params={"indent": 2})
    
    except Chapter.DoesNotExist:
        return JsonResponse({"error": "Chapter not found"}, status=404)
    
@csrf_exempt
@require_http_methods(["GET"])
def get_all_chapters(request):
    chapters = Chapter.objects.all().order_by("standard", "subject", "chapter_number", "created_at")

    chapters_data = [
        {
            "id": str(chapter.id),
            "chapter_name": chapter.chapter_name,
            "chapter_number": chapter.chapter_number,
            "standard": chapter.standard,
            "subject": chapter.subject,
        }
        for chapter in chapters
    ]

    return JsonResponse({"chapters": chapters_data}, status=200, json_dumps_params={"indent": 2})

@csrf_exempt
@require_http_methods(["POST"])
def make_topic_easier(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    chapter_id = payload.get("chapter_id") or request.POST.get("chapter_id")
    topic_id = payload.get("topic_id") or request.POST.get("topic_id")

    if not topic_id:
        return JsonResponse({"error": "Please provide topic_id"}, status=400)

    try:
        topic = Topic.objects.get(id=topic_id)
        # original_content = getattr(topic, "topic_content", "")
        # Build a prompt for the AI to simplify and provide analogies
        original_content = getattr(topic, "topic_content", "")
        if not topic.simplified_content:
            prompt = f"""
            Simplify the following content so that a 5-year-old can understand it.
            - Use very simple words and short sentences.
            - Provide 2 short analogies that relate the concept to everyday objects or experiences a young child knows.
            - Return the simplified explanation followed by the analogies. Separate each bullet/line with a newline.

            Content:
            """ + original_content

            try:
                resp = generate_AI_response(prompt=prompt)
                simplified_content = resp.choices[0].message.content.strip()
            except Exception as e:
                return JsonResponse({"error": f"AI simplification failed: {str(e)}"}, status=500)

            # Update the existing Topic by saving the simplified content into the
            # `simplified_content` JSON field (list of strings). Do not create a new
            # Topic row — keep the OG topic but attach the simplified version.

            simplified_lines = _split_topic_content(simplified_content)
            topic.simplified_content = simplified_lines
            topic.save()

        return JsonResponse(
            {
                "message": "Topic content simplified successfully",
                "original_topic_id": str(topic.id),
                "simplified_content": topic.simplified_content or [],
            },
            status=200,
            json_dumps_params={"indent": 2},
        )

    except Topic.DoesNotExist:
        return JsonResponse({"error": "Topic not found"}, status=404)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_all_chapters_and_topics(request):
    """Utility function to delete all chapters and topics from the database."""
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    chapter_id = payload.get("chapter_id")

    if not chapter_id:
        return JsonResponse({"error": "Please provide chapter_id"}, status=400)
    
    try:
        chapter = Chapter.objects.get(id=chapter_id)
        topics_deleted, _ = Topic.objects.filter(chapter=chapter).delete()
        chapter.delete()

        return JsonResponse(
            {
                "message": "Chapter and its topics deleted successfully",
                "chapter_id": str(chapter_id),
                "topics_deleted_count": topics_deleted,
            },
            status=200,
            json_dumps_params={"indent": 2},
        )
    except Chapter.DoesNotExist:
        return JsonResponse({"error": "Chapter not found"}, status=404) 
    
@csrf_exempt
@require_http_methods(["GET"])
def get_topic_details(request):
    # Try to parse JSON body first (clients send application/json).
    # If parsing fails or field missing, fall back to form data.
    topic_id = None
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    topic_id = payload.get("topic_id") or request.POST.get("topic_id")

    if not topic_id:
        return JsonResponse({"error": "Please provide topic_id"}, status=400)
    
    try:
        topic = Topic.objects.get(id=topic_id)

        topic_data = {
            "id": str(topic.id),
            "topic_name": topic.topic_name,
            "content_summary": _split_topic_content(getattr(topic, "topic_content", None)),
            "simplified_content": [topic.simplified_content] or [],
        }

        return JsonResponse({"topic": topic_data}, status=200, json_dumps_params={"indent": 2})
    
    except Topic.DoesNotExist:
        return JsonResponse({"error": "Topic not found"}, status=404)
    

@csrf_exempt
@require_http_methods(["POST","GET"])
def generate_test_for_chapter(request):
    # Accept JSON body, but also support query params for GET requests.
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    topic_id = payload.get("topic_id") or request.GET.get("topic_id") or request.POST.get("topic_id")
    chapter_id = payload.get("chapter_id") or request.GET.get("chapter_id") or request.POST.get("chapter_id")
    # Require at least one of topic_id OR chapter_id (not both)
    if not (topic_id or chapter_id):
        return JsonResponse({"error": "Please provide topic_id or chapter_id"}, status=400)

    if topic_id:
        questions_qs = Question.objects.filter(topic_id=topic_id)
    else:
        questions_qs = Question.objects.filter(chapter_id=chapter_id)

    # Categorize by level
    print("Total questions found:", questions_qs)
    easy_qs = list(questions_qs.filter(question_level=Question.EASY))
    medium_qs = list(questions_qs.filter(question_level=Question.MEDIUM))
    hard_qs = list(questions_qs.filter(question_level=Question.HARD))

    # Desired distribution (can be tuned)
    desired = {Question.EASY: 4, Question.MEDIUM: 4, Question.HARD: 2}

    def _sample_pool(pool, count):
        if not pool:
            return []
        if len(pool) <= count:
            return pool.copy()
        return random.sample(pool, count)

    selected = []
    selected.extend(_sample_pool(easy_qs, desired[Question.EASY]))
    selected.extend(_sample_pool(medium_qs, desired[Question.MEDIUM]))
    selected.extend(_sample_pool(hard_qs, desired[Question.HARD]))

    # If we don't have enough questions to meet distribution, fill from remaining
    remaining = [q for q in questions_qs if q not in selected]
    while len(selected) < sum(desired.values()) and remaining:
        selected.append(remaining.pop(0))

    # Shuffle final order
    random.shuffle(selected)

    questions_out = []
    for q in selected:
        questions_out.append({
            "id": str(q.id),
            "topic": q.topic.topic_name if q.topic else None,
            "level": q.get_question_level_display(),
            "question_text": q.question_text,
            "options": q.options or {},
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
        })

    return JsonResponse({"test": questions_out, "count": len(questions_out)}, status=200, json_dumps_params={"indent": 2})


@csrf_exempt
@require_http_methods(["POST"])
def generate_chapter_note(request):
    """
    Request JSON: { "chapter_id": "<uuid>" }

    Returns JSON containing:
      - chapter metadata
      - topics: list of { id, topic_name, summary } (summary prefers simplified_content and is cached per-topic)
      - faq: recent unique Q/A pairs from Chat (scoped to chapter/topics in chapter)
      - questions: list of 10 sampled questions (3 easy,3 medium,4 hard) from Question table for the chapter

    This is intended to be used by the frontend to render a chapter note / PDF.
    """
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    chapter_id = payload.get("chapter_id")
    if not chapter_id:
        return JsonResponse({"error": "Please provide chapter_id"}, status=400)

    try:
        chapter = Chapter.objects.get(id=chapter_id)
    except Chapter.DoesNotExist:
        return JsonResponse({"error": "Chapter not found"}, status=404)

    # Topics: prefer simplified_content, fallback to topic_content; cache per-topic to minimize DB hits
    topics_qs = Topic.objects.filter(chapter=chapter).order_by("order", "created_at")
    topics_out = []
    for topic in topics_qs:
        cache_key = f"topic_summary:{topic.id}"
        summary = cache.get(cache_key)
        if not summary:
            simplified = getattr(topic, "simplified_content", None)
            if simplified:
                if isinstance(simplified, (list, tuple)):
                    summary = "\n".join([str(x).strip() for x in simplified if x])
                else:
                    summary = str(simplified)
            else:
                summary = (topic.topic_content or "") if getattr(topic, "topic_content", None) else ""
            # cache for 6 hours
            cache.set(cache_key, summary, timeout=6 * 3600)

        topics_out.append({
            "id": str(topic.id),
            "topic_name": topic.topic_name,
            "summary": summary,
        })

    # FAQs: gather recent chats that are tied to this chapter or topics within it. Deduplicate by question_text.
    chats_qs = Chat.objects.filter(Q(chapter=chapter) | Q(topic__chapter=chapter)).order_by("-created_at")
    faq_out = []
    seen_qs = set()
    for c in chats_qs:
        qtxt = (c.question_text or "").strip()
        if not qtxt:
            continue
        if qtxt in seen_qs:
            continue
        seen_qs.add(qtxt)
        faq_out.append({
            "question": qtxt,
            "answer": (c.answer_text or "").strip(),
            "chat_id": str(c.id),
            "created_at": c.created_at.isoformat() if getattr(c, 'created_at', None) else None,
        })
        if len(faq_out) >= 10:
            break

    # Questions: sample 3 easy, 3 medium, 4 hard (fall back to available pool)
    questions_qs = Question.objects.filter(chapter=chapter)
    easy_qs = list(questions_qs.filter(question_level=Question.EASY))
    medium_qs = list(questions_qs.filter(question_level=Question.MEDIUM))
    hard_qs = list(questions_qs.filter(question_level=Question.HARD))

    desired = {Question.EASY: 3, Question.MEDIUM: 3, Question.HARD: 4}

    def _sample_pool(pool, count):
        if not pool:
            return []
        if len(pool) <= count:
            return pool.copy()
        return random.sample(pool, count)

    selected = []
    selected.extend(_sample_pool(easy_qs, desired[Question.EASY]))
    selected.extend(_sample_pool(medium_qs, desired[Question.MEDIUM]))
    selected.extend(_sample_pool(hard_qs, desired[Question.HARD]))

    remaining = [q for q in questions_qs if q not in selected]
    while len(selected) < sum(desired.values()) and remaining:
        selected.append(remaining.pop(0))

    # Ensure stable ordering: by level then randomize within same level
    random.shuffle(selected)

    questions_out = []
    for q in selected:
        questions_out.append({
            "id": str(q.id),
            "topic": q.topic.topic_name if q.topic else None,
            "level": q.get_question_level_display(),
            "question_text": q.question_text,
            "options": q.options or {},
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
        })

    chapter_payload = {
        "id": str(chapter.id),
        "chapter_name": chapter.chapter_name,
        "chapter_number": chapter.chapter_number,
        "standard": chapter.standard,
        "subject": chapter.subject,
    }

    return JsonResponse(
        {
            "chapter": chapter_payload,
            "topics": topics_out,
            "faq": faq_out,
            "questions": questions_out,
        },
        status=200,
        json_dumps_params={"indent": 2},
    )

@csrf_exempt
@require_http_methods(["POST"])
def ask_topic_question(request):
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    topic_id = payload.get("topic_id")
    chapter_id = payload.get("chapter_id")
    question_text = payload.get("question_text")

    if not topic_id or not question_text:
        return JsonResponse({"error": "topic_id and question_text are required"}, status=400)

    # fetch topic (and optional chapter)
    try:
        topic = Topic.objects.get(id=topic_id)
    except Topic.DoesNotExist:
        return JsonResponse({"error": "topic not found"}, status=404)

    chapter = None
    if chapter_id:
        try:
            chapter = Chapter.objects.get(id=chapter_id)
        except Chapter.DoesNotExist:
            return JsonResponse({"error": "chapter not found"}, status=404)

    # cache key per topic
    cache_key = f"topic_summary:{topic_id}"
    summary = cache.get(cache_key)
    if not summary:
        # prefer simplified_content (stored as list) else topic_content (text)
        simplified = getattr(topic, "simplified_content", None)
        if simplified:
            if isinstance(simplified, (list, tuple)):
                summary = "\n".join([str(x).strip() for x in simplified if x])
            else:
                summary = str(simplified)
        else:
            summary = (topic.topic_content or "") if getattr(topic, "topic_content", None) else ""
        # store for 24 hours
        cache.set(cache_key, summary, timeout=24 * 3600)

    # build prompt safely (avoid unescaped braces in f-strings)
    prompt = (
        "You are a helpful tutor. Use the following topic summary as context to answer the question.\n\n"
        "Topic summary:\n"
        + summary
        + "\n\nQuestion:\n"
        + question_text
        + "\n\nAnswer concisely and clearly. If the answer requires steps, present them numbered. "
        "If the question cannot be answered from the summary, say so and provide a short suggestion what to read next."
    )

    try:
        resp = generate_AI_response(prompt=prompt)
        # extract text from helper response defensively
        answer_text = None
        try:
            # handle OpenAI-like response objects
            answer_text = resp.choices[0].message.content.strip()
        except Exception:
            try:
                answer_text = str(resp).strip()
            except Exception:
                answer_text = ""
    except Exception as e:
        return JsonResponse({"error": f"AI call failed: {str(e)}"}, status=500)

    # persist chat entry
    try:
        user = request.user if getattr(request, "user", None) and request.user.is_authenticated else None
        chat = Chat.objects.create(
            user=user,
            topic=topic,
            chapter=chapter,
            question_text=question_text,
            answer_text=answer_text,
        )
    except Exception as e:
        return JsonResponse({"error": f"failed to save chat: {str(e)}"}, status=500)

    return JsonResponse(
        {
            "chat_id": str(chat.id),
            "answer": answer_text,
        },
        status=200,
    )