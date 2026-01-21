from database.content import Chapter, Topic

from django.http import JsonResponse
import json
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
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

    chapter_id = payload.get("chapter_id")
    topic_id = payload.get("topic_id")
    if not topic_id:
        return JsonResponse({"error": "Please provide topic_id"}, status=400)

    try:
        topic = Topic.objects.get(id=topic_id)
        original_content = getattr(topic, "topic_content", "")
        # Build a prompt for the AI to simplify and provide analogies
        
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
                "simplified_content": simplified_lines,
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