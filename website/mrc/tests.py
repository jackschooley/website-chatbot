from django.shortcuts import reverse
from django.test import TestCase
from .models import MRCResults, Topic

def create_topic(topic, context):
    return Topic.objects.create(topic = topic, context = context)
    
class MRCViewTests(TestCase):
    
    def test_valid_topic(self):
        """This tests if the page loads properly for a valid topic"""
        topic = create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        
    def test_invalid_topic(self):
        """This tests if a 404 appears if the user tries an invalid url pattern"""
        create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        url = reverse("mrc:detail_page", args = ("background", ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)
        
    def test_environment_reset(self):
        """This tests if the session variables that get set during the pipeline
        and redirect process get reset once the user gets back to this page"""
        topic = create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        session = self.client.session
        session["question"] = "What web framework does this site use?"
        session["answer"] = "Django"
        session.save()
        
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        self.client.get(url)
        string = "This string proves the dictionary value does not exist"
        question = self.client.session.get("question", string)
        answer = self.client.session.get("answer", string)
        self.assertEqual(question, string)
        self.assertEqual(answer, string)
        
    def test_adds_to_database(self):
        """This tests if the form adds the question to the question database"""
        topic = create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        question = "What web framework does this site use?"
        self.client.post(url, {"question_text": question})
        self.assertEqual(MRCResults.objects.count(), 1)
        
    def test_redirect(self):
        """This tests if the redirect works as it's supposed to"""
        topic = create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        question = "What web framework does this site use?"
        response = self.client.post(url, {"question_text": question})
        redirect_url = reverse("mrc:submitted_page", args = (topic.topic, ))
        self.assertRedirects(response, redirect_url)
    
    def test_known_answer(self):
        """This will be the test to see if the MRC model can answer questions
        that it is supposed to know"""
        pass
    
    def test_unknown_answer(self):
        """This will be the test to see if the MRC model outputs a suitable
        response to questions that it is not supposed to know"""
        pass
        
class SubmittedViewTests(TestCase):
    
    def test_valid_topic(self):
        """This tests if the page loads properly for a valid topic"""
        topic = create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        url = reverse("mrc:submitted_page", args = (topic.topic, ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        
    def test_invalid_topic(self):
        """This tests if a 404 appears if the user tries an invalid url pattern"""
        create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        url = reverse("mrc:submitted_page", args = ("background", ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)
        
    def test_redirect_environment(self):
        """This tests if the page loads the session variables correctly from the
        previous page and properly displays the question and answer"""
        topic = create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        question = "What web framework does this site use?"
        answer = "Django"
        
        session = self.client.session
        session["question"] = question
        session["answer"] = answer
        session.save()
        
        url = reverse("mrc:submitted_page", args = (topic.topic, ))
        response = self.client.get(url)
        self.assertContains(response, question)
        self.assertContains(response, answer)
        
    def test_empty_environment(self):
        """This tests if the correct behavior occurs when the user tries to break
        the logic of the site and go to this page without going to the mrc page
        first. Right now I have it to give a fourth wall break question/answer
        but I might just throw up a 404. Idk if that seems right though"""
        topic = create_topic("stack", "This site uses PyTorch, Django, Docker, etc.")
        url = reverse("mrc:submitted_page", args = (topic.topic, ))
        response = self.client.get(url)
        question = "Can you (yes you, the person reading this) break this website?"
        answer = "Clearly not. Nice try though."
        self.assertContains(response, question)
        self.assertContains(response, answer)