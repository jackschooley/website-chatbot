from django.urls import path

from . import views

app_name = "mrc"
urlpatterns = [
    path("", views.IndexView.as_view(), name = "homepage"),
    path("detail", views.mrc_view, name = "mrc_page")
    ]