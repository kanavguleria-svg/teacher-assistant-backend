"""
URL configuration for api app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='api-home'),
    path('health', views.health, name='api-health'),
    path('upload_and_parse_pdf', views.upload_and_parse_pdf, name = "api-parse"),
    path('get_chapter_details', views.get_chapter_details, name = "api-get-chapter-details"),
    path('make_topic_easier', views.make_topic_easier, name = "api-make-topic-easier"),
    path('delete_all_chapters_and_topics', views.delete_all_chapters_and_topics, name = "api-delete-all-chapters-and-topics"),
    path('get_topic_details', views.get_topic_details, name = "api-get-topic-details"),
    path('get_all_chapters', views.get_all_chapters, name = "api-get-all-chapters"),
    path('generate_test_for_chapter', views.generate_test_for_chapter, name = "api-generate_test_for_chapter"),
    path('ask_topic_question', views.ask_topic_question, name = "api-ask-topic-question"),
    path('generate_chapter_note', views.generate_chapter_note, name = "api-generate-chapter-note"),
    # path('synthesize_final_chapter', views.synthesize_final_chapter, name = "api-synthesize-final-chapter"),
]