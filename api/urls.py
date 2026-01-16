"""
URL configuration for api app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='api-home'),
    path('health', views.health, name='api-health'),
    path('upload_and_parse_pdf', views.upload_and_parse_pdf, name = "api-parse")
]
