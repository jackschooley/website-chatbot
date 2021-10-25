from django.db import models
from django.forms import ModelForm


# Create your models here.
class MRCResults(models.Model):
    topic = models.CharField(max_length = 30)
    question_text = models.CharField("Question: ", max_length = 300)
    date = models.DateTimeField("Date")
    answer_text = models.CharField(max_length = 512)
    
    def __str__(self):
        return self.question_text
    
class MRCResultsForm(ModelForm):
    class Meta:
        model = MRCResults
        fields = ["question_text"]