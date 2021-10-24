from django.db import models

max_question_len = 300

# Create your models here.
class MRCResults(models.Model):
    question_text = models.CharField(max_length = max_question_len)
    date = models.DateTimeField("Date")
    answer_text = models.CharField(max_length = 512)
    
    def __str__(self):
        return self.question_text