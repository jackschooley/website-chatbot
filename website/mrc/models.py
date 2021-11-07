from django.db import models
from django.forms import ModelForm

class Topic(models.Model):
    topic = models.CharField(max_length = 30, unique = True)
    
    def __str__(self):
        return self.topic

class MRCResults(models.Model):
    topic = models.ForeignKey(Topic, to_field = "topic", on_delete = models.CASCADE)
    question_text = models.CharField("Question", max_length = 300)
    date_time = models.DateTimeField()
    answer_text = models.CharField(max_length = 512)
    
    def __str__(self):
        return self.question_text
    
class MRCResultsForm(ModelForm):
    class Meta:
        model = MRCResults
        fields = ["question_text"]