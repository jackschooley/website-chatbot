import torch
import transformers
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views import generic
from .ml.model import MRCModel
from .models import MRCResultsForm, Topic
from .mrc_pipeline import mrc_pipeline

def homepage_view(request):
    return render(request, "mrc/home.html")

class TopicListView(generic.ListView):
    context_object_name = "topics"
    model = Topic
    template_name = "mrc/questions.html"

def mrc_view(request, topic):
    #check to see if it's a valid page based on topic
    topic_instance = get_object_or_404(Topic, topic = topic)
    
    #get the context for question answering
    context = topic_instance.context
    
    #reset session variables
    if request.session.get("question"):
        del request.session["question"]
        del request.session["answer"]
    
    distilbert_config = transformers.DistilBertConfig(n_layers = 3, n_heads = 6,
                                                      dim = 384, hidden_dim = 1536)
    model = MRCModel(distilbert_config)
    model.load_state_dict(torch.load("mrc/ml/model_weights.pth", torch.device("cpu")))
    
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
            qa.topic = topic_instance
            qa.date_time = timezone.now()
            qa.answer_text = answer
            qa.save()
            
            return redirect("mrc:submitted_page", topic)
    else:
        form = MRCResultsForm()
    return render(request, "mrc/detail.html", {"form": form})

def submitted_view(request, topic):
    #check to see if it's a valid page based on topic
    topic_instance = get_object_or_404(Topic, topic = topic)
    
    try:
        #retrieve session variables from last page
        question = request.session["question"]
        answer = request.session["answer"]
    except KeyError:
        #the user tried to break the site logic by going to this page without a question
        question = "Can you (yes you, the person reading this) break this website?"
        answer = "Clearly not. Nice try though."
    qa_context = {"question": question, "answer": answer, "topic_instance": topic_instance}
    return render(request, "mrc/submitted.html", qa_context)

def contact_view(request):
    return render(request, "mrc/contact.html")