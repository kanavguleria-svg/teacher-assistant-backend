# Vidya Setu Backend

Django backend project for Vidya Setu application.

## Setup Instructions

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**:
   ```bash
   python manage.py migrate
   ```

5. **Create a superuser** (optional):
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

The server will be available at `http://127.0.0.1:8000/`

## Available URLs

- Admin panel: `http://127.0.0.1:8000/admin/`
- API home: `http://127.0.0.1:8000/api/`
- Health check: `http://127.0.0.1:8000/api/health/`

## Project Structure

```
teacher-assistant-backend/
├── config/          # Main project configuration
│   ├── settings.py  # Django settings
│   ├── urls.py      # Main URL configuration
│   ├── wsgi.py      # WSGI configuration
│   └── asgi.py      # ASGI configuration
├── api/             # API app
│   ├── urls.py      # API URL routes
│   └── views.py     # API views
├── manage.py        # Django management script
└── requirements.txt # Python dependencies
````

# Teacher Assistant Backend — API Contract

This document describes the public API endpoints for the Teacher Assistant backend (Django). It covers request/response shapes, headers, content types, and sample requests.

Base URL (dev)
- http://127.0.0.1:8000/api/

Authentication
- Current dev setup: no enforced auth (many endpoints are `@csrf_exempt`).
- Production: add token-based auth (recommended). Do NOT store API keys in repo; use environment variables.

Environment variables (important)
- `OPENAI_API_KEY` — required for AI operations (set in environment).
- `DJANGO_SETTINGS_MODULE` — standard Django settings env var.

Common notes
- File uploads must use multipart/form-data (Postman → Body → form-data). The file field name is `file`.
- JSON endpoints expect valid JSON in the request body (Content-Type: application/json).
- Topic simplified content is stored in `Topic.simplified_content` as JSON (list of strings).
- Questions are persisted to the `Question` table with fields: `question_text`, `options` (JSON), `correct_answer`, `explanation`, `question_level` (EASY/MEDIUM/HARD), `topic_id`, `chapter_id`.

Endpoints

1) POST /api/upload_and_parse_pdf
- Description: Upload a PDF or ZIP. Saves file and enqueues/executes parsing (idempotent by file SHA). Returns persisted ids or job id.
- Headers:
  - For file: Content-Type: multipart/form-data
- Body (form-data):
  - file: (File) .pdf or .zip
  - standard: (string/int)
  - subject: (string)
- Success (202/200):
  - {
      "message": "accepted" | "processed",
      "resource_id": "<uuid>",
      "chapter_id": "<uuid>",
      "topics": ["<topic_id>", ...]   // if processed
    }
- Errors:
  - 400: invalid file / missing params
  - 500: server/processing errors

Example (curl - file):
curl -X POST "http://127.0.0.1:8000/api/upload_and_parse_pdf" \
  -F "file=@/path/to/book.pdf" \
  -F "standard=10" \
  -F "subject=Math"

2) POST /api/get_chapter_details
- Description: Fetch chapter details including topics. Returns both original and simplified content (simplified stored in `simplified_content`).
- Headers:
  - Content-Type: application/json
- Body (JSON):
  - { "chapter_id": "<uuid>" }
- Response (200):
  - {
      "chapter": {
        "id": "<uuid>",
        "chapter_name": "...",
        "standard": 10,
        "subject": "Math",
        "ai_summary": "<string>"
      },
      "topics": [
        {
          "id": "<uuid>",
          "topic_name": "Real Numbers",
          "content_summary": ["line1", "line2", ...],   // original content split into list
          "simplified_content": ["line1", "line2", ...], // from Topic.simplified_content (JSON list) or []
          "topic_type": "OG"
        },
        ...
      ],
      "simplified_topics": [ /* if using separate list; current design stores simplified_content on same Topic */ ]
    }
- Errors:
  - 400: missing chapter_id
  - 404: chapter not found

Example:
curl -X POST "http://127.0.0.1:8000/api/get_chapter_details" \
  -H "Content-Type: application/json" \
  -d '{"chapter_id":"ae632f04-7954-453b-91ee-e5bafcaa70e9"}'

3) POST /api/make_topic_easier
- Description: Create a simplified explanation for a topic using the LLM and save to `Topic.simplified_content` (JSON list). Does not create a duplicate Topic (unless configured otherwise).
- Headers:
  - Content-Type: application/json
- Body:
  - { "topic_id": "<uuid>" }
- Response (200):
  - {
      "message": "Topic simplified successfully",
      "topic_id": "<uuid>",
      "simplified_content": ["short sentence 1", "Analogy 1: ...", ...]
    }
- Errors:
  - 400: missing topic_id
  - 404: topic not found
  - 500: LLM/service error (rate limits, etc.)

Example:
curl -X POST "http://127.0.0.1:8000/api/make_topic_easier" \
  -H "Content-Type: application/json" \
  -d '{"topic_id":"b93a6390-8dfd-479c-b2e1-7ab36a1516d0"}'

4) GET /api/get_all_chapters
- Description: List basic chapter metadata.
- Response:
  - { "chapters": [{ "id": "<uuid>", "chapter_name":"...", "chapter_number":1, "standard":10, "subject":"Math" }, ...] }

Example:
curl "http://127.0.0.1:8000/api/get_all_chapters"

5) GET/POST /api/generate_test_for_chapter
- Description: Generate a test from questions in a topic or entire chapter. Accepts topic_id or chapter_id.
- Inputs:
  - Query param or JSON body:
    - topic_id: "<uuid>" OR chapter_id: "<uuid>"
- Behavior:
  - Collects questions from DB and samples by level (default distribution: Easy/Medium/Hard).
  - Returns selected questions. NOTE: if endpoint should be student-facing, remove `correct_answer` and `explanation` from response.
- Response (200):
  - {
      "questions": [
        {
          "id":"<uuid>",
          "topic_id":"<uuid>",
          "question_text":"What is ...?",
          "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
          "level": "Easy",
          "correct_answer": "A",           // remove for student usage
          "explanation": "Because..."
        }, ...
      ],
      "count": 10
    }
- Errors:
  - 400: missing both topic_id and chapter_id
  - 404: no questions found

Example:
curl "http://127.0.0.1:8000/api/generate_test_for_chapter?chapter_id=<uuid>"

6) Job/Processing status (optional)
- If you implement UploadJob/Celery, add:
  - GET /api/job_status?job_id=<uuid>
  - Response: { "job_id": "<uuid>", "status": "PENDING|PROCESSING|COMPLETE|FAILED", "error": "...", "resource_id": "...", "chapter_id": "..." }

Data model summary (relevant fields)
- Resource: id, file_hash, standard, subject, file_name, source_path, created_at
- Chapter: id, resource (FK), chapter_name, standard, subject, full_text, ai_summary, created_at
- Topic: id, chapter (FK), topic_name, topic_content (text), simplified_content (JSONField), order, topic_type, created_at
- Question: id, chapter (FK), topic (FK), question_level (enum), question_text, options (JSONField), correct_answer, explanation, created_at

Error handling & LLM considerations
- LLM calls are retried with backoff. Watch for rate limits (OpenAI 429). Use exponential backoff and jitter.
- The code previously had f-string issues when prompt templates included literal JSON braces — prompts now avoid unescaped braces or build strings via concatenation. If you see "Invalid format specifier" errors, that indicates an unescaped brace inside an f-string; search for `f"""` prompts with `{` braces.
- Idempotency: extract_pdf uses file SHA-256 to avoid duplicate resource creation; pass `force=True` to reprocess.

Operational notes
- Replace debug `print()` calls with structured `logging`.
- Use Celery + Redis for background processing in production. For dev, a ThreadPoolExecutor is acceptable.
- Run migrations after model changes:
  - python manage.py makemigrations database
  - python manage.py migrate

Testing
- Unit-test `merge_topics`, `synthesize_final_chapter` (mock LLM), and `extract_pdf` pipeline (mock LLM + small sample PDFs).
- Integration tests: endpoints and DB persistence.

Contact / Next steps
- If you want, generate OpenAPI/Swagger from views or migrate endpoints to Django REST Framework for cleaner serializations and validation.
- I can produce an OpenAPI YAML for these endpoints if needed.

## API Endpoints (Quick Reference)

Base URL (dev): `http://127.0.0.1:8000/api/`  
All JSON endpoints expect `Content-Type: application/json` unless noted.

1) POST /api/upload_and_parse_pdf
- Description: Upload a PDF or ZIP and process it (multipart/form-data).
- Body (form-data):
  - `file` (File) - .pdf or .zip
  - `standard` (string/int)
  - `subject` (string)
- Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/upload_and_parse_pdf" \
  -F "file=@/path/to/book.pdf" \
  -F "standard=10" \
  -F "subject=Math"
```

2) POST /api/get_chapter_details
- Description: Get chapter metadata + topics (original + simplified content).
- Body (JSON): `{ "chapter_id": "<uuid>" }`
- Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/get_chapter_details" \
  -H "Content-Type: application/json" \
  -d '{"chapter_id":"ae632f04-7954-453b-91ee-e5bafcaa70e9"}'
```

3) POST /api/make_topic_easier
- Description: Return existing simplified content if present; if `force:true` or none exists, call AI, save `Topic.simplified_content` and return it.
- Body (JSON): `{ "topic_id":"<uuid>", "force": true|false }`
- Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/make_topic_easier" \
  -H "Content-Type: application/json" \
  -d '{"topic_id":"<topic-uuid>"}'
```

4) GET /api/get_all_chapters
- Description: List basic chapter metadata.
- Example:
```bash
curl "http://127.0.0.1:8000/api/get_all_chapters"
```

5) GET/POST /api/generate_test_for_chapter
- Description: Generate a test from questions. Provide either `topic_id` or `chapter_id`.
- Example:
```bash
curl "http://127.0.0.1:8000/api/generate_test_for_chapter?chapter_id=<uuid>"
```
or
```bash
curl -X POST "http://127.0.0.1:8000/api/generate_test_for_chapter" \
  -H "Content-Type: application/json" \
  -d '{"topic_id":"<uuid>"}'
```

6) POST /api/ask_topic_question
- Description: Ask a question scoped to a topic (uses cached topic summary as context). Saves the Q/A to `Chat`.
- Body (JSON): `{ "topic_id":"<uuid>", "chapter_id":"<uuid>", "question_text":"..." }`
- Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/ask_topic_question" \
  -H "Content-Type: application/json" \
  -d '{"topic_id":"<uuid>", "question_text":"What is ...?"}'
```

7) POST /api/generate_chapter_note
- Description: Generate structured chapter note: Topics (summaries), FAQs (from Chat), and 10 sample Questions (3 easy,3 medium,4 hard).
- Body (JSON): `{ "chapter_id":"<uuid>" }`
- Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/generate_chapter_note" \
  -H "Content-Type: application/json" \
  -d '{"chapter_id":"<uuid>"}'
```

8) Authentication / User management
- POST /api/login — form-data or JSON: `{ "username":"..", "password":".." }`
- POST /api/register — restricted: create user (only superuser/school-admins via API)
- Invitation flow: POST /api/create_invite (admin) and POST /api/accept_invite (invitee)

Notes, behaviors & best practices
- File uploads: use `multipart/form-data` (file must be in `request.FILES`).
- Topic simplification: first call returns cached `simplified_content` if present; pass `"force": true` to regenerate.
- Caching: topic summaries are cached to reduce DB reads; cache invalidation occurs on updates/deletes (signals).
- Questions persisted: generated questions are saved in the `Question` table. Q/A interactions saved in `Chat`.
- For student-facing endpoints, remove `correct_answer` and `explanation` from responses.
- Environment: set `OPENAI_API_KEY` in production; do not commit keys.

If you want, I can:
- Add full OpenAPI (Swagger) spec for these endpoints,
- Add example response payloads for each endpoint,
- Or generate curl + Postman collections for testing.
