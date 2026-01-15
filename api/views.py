from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
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

@require_http_methods(["POST"])
def upload_and_parse_pdf(request):
    uploaded_file = request.Files.get()
    subject = request.get("subject")
    standard = request.get("standard")

    if not uploaded_file:
        return JsonResponse({"error": "Please upload a file"}, 503)
    
    if not subject or not standard:
        return JsonResponse({"error": "Subject and standard are required"}, 503)
    
    try:
        standard = int(standard)
    except ValueError:
        return {"error":"The standard should be an int value"}
    
    filename = uploaded_file.lower()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        with open(temp_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        if filename.endswith(".zip"):
            # try:
            #     result = parse_chapters_from_local_zip(
            #         zip_path=temp_path,
            #         standard=standard,
            #         subject=subject,
            #     )
            # except Exception as e:
            #     return JsonResponse(
            #         {"error": f"Failed to process ZIP: {str(e)}"},
            #         status=500
            #     )
            return JsonResponse({"message": "Correct for pdf"})

        elif filename.endswith(".pdf"):
            return JsonResponse({"message": "Correct for pdf"})
            # try:
            #     result = extract_pdf(
            #         pdf_path=temp_path,
            #         standard=standard,
            #         subject=subject,
            #     )
            # except Exception as e:
            #     return JsonResponse(
            #         {"error": f"Failed to process PDF: {str(e)}"},
            #         status=500
            #     )

        else:
            return JsonResponse(
                {"error": "Only PDF or ZIP files are supported"},
                status=400
            )
        
    return JsonResponse(
        {
            "message": "File processed successfully",
            "subject": subject,
            "standard": standard,
            "data": result,
        },
        status=200,
        json_dumps_params={"indent": 2},
    )
