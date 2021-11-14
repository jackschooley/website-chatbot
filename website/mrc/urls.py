from django.urls import path

from . import views

app_name = "mrc"
urlpatterns = [
    path("", views.TopicListView.as_view(), name = "questions_page"),
    path("<topic>", views.mrc_view, name = "detail_page"),
    path("<topic>/submitted", views.submitted_view, name = "submitted_page"),
    ]