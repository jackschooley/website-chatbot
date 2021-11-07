from django.urls import path

from . import views

app_name = "mrc"
urlpatterns = [
    path("", views.IndexView.as_view(), name = "homepage"),
    path("contact", views.contact_view, name = "contact_page"),
    path("detail", views.mrc_view, name = "mrc_page"),
    path("submitted", views.submitted_view, name = "submitted_page")
    ]