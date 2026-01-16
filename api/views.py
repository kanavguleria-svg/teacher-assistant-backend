from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import os
import tempfile
from pdfparser import parse_chapters_from_local_zip, extract_pdf

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
    result = None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, uploaded_file.name)

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
            try:
                result = extract_pdf(
                    pdf_path=temp_path,
                    standard=standard,
                    subject=subject,
                )
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

    