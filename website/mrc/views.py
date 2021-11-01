from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.utils import timezone
from django.views import generic
import torch

from .models import MRCResultsForm
from .mrc_pipeline import mrc_pipeline

#this should eventually be the index for question topics
class IndexView(generic.ListView):
    template_name = "mrc/home.html"
    
    def get_queryset(self):
        return None

#this will be where the mrc model runs
def mrc_view(request):
    model = torch.load("model.pth")
    model.load_state_dict(torch.load("model_weights.pth"))
    model = model.cpu() #this should be fixed at some point, the weights are stored on gpu
    
    context = "Actually I'm not a reply guy."
    
    if request.method == "POST":
        form = MRCResultsForm(request.POST)
        if form.is_valid():
            #process the form results
            question = form.save(commit = False)
            question.topic = "all"
            question.date_time = timezone.now()
            question.answer_text = mrc_pipeline(question.question_text, context, model)
            question.save()
            return HttpResponseRedirect("/submitted/")
    else:
        form = MRCResultsForm()
    return render(request, "mrc/detail.html", {"form": form})

def submitted_view(request):
    return render(request, "mrc/submitted.html")