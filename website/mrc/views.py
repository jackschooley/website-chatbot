from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views import generic

from .models import MRCResultsForm

#this should eventually be the index for question topics
class IndexView(generic.ListView):
    template_name = "mrc/home.html"
    
    def get_queryset(self):
        return None

#this will be where the mrc model runs
def mrc_view(request):
    if request.method == "POST":
        form = MRCResultsForm(request.POST)
        if form.is_valid():
            #process the form results
            form.save()
            return HttpResponseRedirect("/submitted/")
    else:
        form = MRCResultsForm()
    return render(request, "mrc/detail.html", {"form": form})