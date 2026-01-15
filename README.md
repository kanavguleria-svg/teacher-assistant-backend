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
```
