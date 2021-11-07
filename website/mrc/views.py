import torch
import transformers
from django.shortcuts import redirect, render
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
    #reset session variables
    request.session["question"] = None
    request.session["answer"] = None
    
    distilbert_config = transformers.DistilBertConfig(n_layers = 3, n_heads = 6,
                                                      dim = 384, hidden_dim = 1536)
    model = MRCModel(distilbert_config)
    model.load_state_dict(torch.load("mrc/ml/model_weights.pth"))
    
    context = "Actually I'm not a reply guy."
    
    if request.method == "POST":
        form = MRCResultsForm(request.POST)
        if form.is_valid():
            
            #question answering
            qa = form.save(commit = False)
            question = qa.question_text
            answer = mrc_pipeline(question, context, model)
            
            #update session variables for passing to next view
            request.session["question"] = question
            request.session["answer"] = answer
            
            #update model
            qa.topic = "test"
            qa.date_time = timezone.now()
            qa.answer_text = answer
            qa.save()
            
            return redirect("mrc:submitted_page")
    else:
        form = MRCResultsForm()
    return render(request, "mrc/detail.html", {"form": form})

def submitted_view(request):
    #retrieve session variables from last view
    #should build in a way that it can't be accessed if not coming from previous page
    question = request.session["question"]
    answer = request.session["answer"]
    qa_context = {"question": question, "answer": answer}
    return render(request, "mrc/submitted.html", qa_context)

def contact_view(request):
    return render(request, "mrc/contact.html")