# Flask to Django Conversion Notes

This project has been successfully converted from Flask to Django.

## Project Structure

- **Django Project**: `careapp/` - Contains settings, URLs, and WSGI configuration
- **Django App**: `prediction/` - Contains views, URLs, and app configuration
- **Templates**: `templates/` - HTML templates (updated to use Django template syntax)
- **Static Files**: `static/` - CSS, JavaScript, images

## Key Changes Made

1. **Flask routes → Django views**: All Flask routes converted to Django function-based views
2. **URL routing**: Flask `@app.route()` converted to Django `urlpatterns` in `prediction/urls.py`
3. **Template syntax**: 
   - `url_for('static', ...)` → `{% static '...' %}`
   - `request.endpoint` → `request.resolver_match.url_name`
   - Flask form handling → Django `request.POST`
4. **Request handling**: Flask `request.form` → Django `request.POST`
5. **JSON responses**: Flask `jsonify()` → Django `JsonResponse()`

## Running the Application

### Development Server
```bash
python manage.py runserver
```

The application will be available at `http://localhost:8000`

### Migrations
```bash
python manage.py migrate
```

## Original Flask File

The original Flask `app.py` file is still present but is no longer used. The Django application uses:
- `prediction/views.py` - Contains all view functions
- `prediction/urls.py` - Contains URL routing
- `careapp/settings.py` - Django settings

## Dependencies

Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

