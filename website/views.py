from django.shortcuts import render

def homepage_view(request):
    return render(request, "home.html")

def contact_view(request):
    return render(request, "contact.html")