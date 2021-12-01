from django.db import models
from django.forms import ModelForm

class Topic(models.Model):
    topic = models.CharField(max_length = 32, unique = True)
    description = models.CharField(max_length = 128)
    context = models.CharField(max_length = 2048)
    
    def __str__(self):
        return self.topic

class MRCResult(models.Model):
    topic = models.ForeignKey(Topic, to_field = "topic", on_delete = models.CASCADE)
    question_text = models.CharField(max_length = 128)
    date_time = models.DateTimeField()
    answer_text = models.CharField(max_length = 1024)
    
    def __str__(self):
        return self.question_text
    
class MRCResultForm(ModelForm):
    class Meta:
        model = MRCResult
        fields = ["question_text"]
        labels = {"question_text": ""}