import pickle
import torch
import transformers
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views import generic
from ml.model import MRCModel
from .models import MRCResultsForm, Topic
from .mrc_pipeline import mrc_pipeline

class TopicListView(generic.ListView):
    context_object_name = "topics"
    model = Topic
    template_name = "mrc/questions.html"
    
    def __init__(self):
        super(TopicListView, self).__init__()
        
        # if the topics table is empty, fill it from text files
        if not self.model.objects.all().exists():
            topics_folder = "mrc/topics/"
            
            # this file has a list of the topics
            with open(topics_folder + "topics.txt") as topic_file:
                topics = topic_file.readlines()
            
            # use the topic name to point to the file with its context
            for topic_unstripped in topics:
                topic = topic_unstripped.strip()
                filename = topics_folder + topic + ".txt"
                context = ""
                with open(filename) as context_file:
                    for line in context_file:
                        context += line.strip() + " "
                self.model.objects.create(topic = topic, context = context)

def mrc_view(request, topic):
    # check to see if it's a valid page based on topic
    topic_instance = get_object_or_404(Topic, topic = topic)
    context = topic_instance.context
    
    # reset session variables
    if request.session.get("question"):
        del request.session["question"]
        del request.session["answer"]
    
    # load tokenizer, model, and parameters
    tokenizer = transformers.DistilBertTokenizerFast("ml/vocab.txt")
    distilbert_config = transformers.DistilBertConfig()
    model = MRCModel(distilbert_config)
    model.load_state_dict(torch.load("ml/model_weights.pth", torch.device("cpu")))
    
    # load threshold for evaluating answerability
    with open("ml/delta.pickle", "rb") as file:
        delta = pickle.load(file)
    
    if request.method == "POST":
        form = MRCResultsForm(request.POST)
        if form.is_valid():
            
            # question answering
            qa = form.save(commit = False)
            question = qa.question_text
            answer = mrc_pipeline(question, context, tokenizer, model, delta)
            
            # update session variables for passing to next view
            request.session["question"] = question
            request.session["answer"] = answer
            
            # update model
            qa.topic = topic_instance
            qa.date_time = timezone.now()
            qa.answer_text = answer
            qa.save()
            
            return redirect("mrc:submitted_page", topic)
    else:
        form = MRCResultsForm()
    return render(request, "mrc/detail.html", {"form": form})

def submitted_view(request, topic):
    # check to see if it's a valid page based on topic
    topic_instance = get_object_or_404(Topic, topic = topic)
    
    try:
        # retrieve session variables from last page
        question = request.session["question"]
        answer = request.session["answer"]
    except KeyError:
        # the user tried to break the site logic by going to this page without a question
        question = "Can you (yes you, the person reading this) break this website?"
        answer = "Clearly not. Nice try though."
        
    qa_context = {"question": question, "answer": answer, "topic_instance": topic_instance}
    return render(request, "mrc/submitted.html", qa_context)