import pickle
import torch
import transformers
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views import generic
from ml.model import MRCModel
from .models import MRCResultForm, Topic
from .pipeline import mrc_pipeline

class TopicListView(generic.ListView):
    context_object_name = "topics"
    model = Topic
    template_name = "mrc/questions.html"
    
    def __init__(self):
        super(TopicListView, self).__init__()
        
        # if the topics table is empty, fill it from text files
        if not self.model.objects.all().exists():
            topics_folder = "mrc/topics/"
            
            # this file has a list of the topics and descriptions
            with open(topics_folder + "topics.txt") as topics_file:
                topics = topics_file.readlines()
            
            # use the topic name to point to the file with its context
            for topic_data in topics:
                topic, description_unstripped = topic_data.split("|")
                description = description_unstripped.strip()
                filename = topics_folder + "contexts/" + topic + ".txt"
                context = ""
                with open(filename) as context_file:
                    for line in context_file:
                        context += line.strip() + " "
                
                self.model.objects.create(topic = topic, 
                                          description = description,
                                          context = context)

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
    model.load_state_dict(torch.load("ml/model_weights.pth", 
                                     torch.device("cpu")))
    
    # load threshold for evaluating answerability
    with open("ml/delta.pickle", "rb") as file:
        delta = pickle.load(file)
    
    if request.method == "POST":
        form = MRCResultForm(request.POST)
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
        form = MRCResultForm()
        example = "mrc/topics/examples/{}.txt".format(topic_instance.topic)
        with open(example) as file:
            examples = [line.strip() for line in file]
        context = {
            "examples": examples, 
            "form": form, 
            "topic_instance": topic_instance
        }
    return render(request, "mrc/detail.html", context)

def submitted_view(request, topic):
    # check to see if it's a valid page based on topic
    topic_instance = get_object_or_404(Topic, topic = topic)
    
    try:
        # retrieve session variables from last page
        question = request.session["question"]
        answer = request.session["answer"]
    except KeyError:
        # user broke the site logic by going to this page without a question
        question = "Can you (the person reading this) break this website?"
        answer = "Clearly not. Nice try though."
        
    context = {
        "question": question, 
        "answer": answer,
        "topic_instance": topic_instance
    }
    return render(request, "mrc/submitted.html", context)