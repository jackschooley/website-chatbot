import torch
import transformers
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.utils import timezone
from django.views import generic
from .ml.model import MRCModel
from .models import MRCResultsForm
from .mrc_pipeline import mrc_pipeline

#this should eventually be the index for question topics
class IndexView(generic.ListView):
    template_name = "mrc/home.html"
    
    def get_queryset(self):
        return None

#this will be where the mrc model runs
def mrc_view(request):
    distilbert_config = transformers.DistilBertConfig(n_layers = 3, n_heads = 6,
                                                      dim = 384, hidden_dim = 1536)
    model = MRCModel(distilbert_config)
    model.load_state_dict(torch.load("mrc/ml/model_weights.pth"))
    
    context = "Actually I'm not a reply guy."
    
    if request.method == "POST":
        form = MRCResultsForm(request.POST)
        if form.is_valid():
            question = form.save(commit = False)
            question.topic = "test"
            question.date_time = timezone.now()
            question.answer_text = mrc_pipeline(question.question_text, context, model)
            question.save()
            return HttpResponseRedirect("submitted")
    else:
        form = MRCResultsForm()
    return render(request, "mrc/detail.html", {"form": form})

def submitted_view(request):
    return render(request, "mrc/submitted.html")