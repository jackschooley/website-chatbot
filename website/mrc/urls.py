from django.urls import path

from . import views

app_name = "mrc"
urlpatterns = [
    path("", views.homepage_view, name = "homepage"),
    path("questions", views.question_index_view, name = "questions_page"),
    path("questions/<topic>", views.mrc_view, name = "detail_page"),
    path("questions/<topic>/submitted", views.submitted_view, name = "submitted_page"),
    path("contact", views.contact_view, name = "contact_page")
    ]