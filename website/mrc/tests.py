from django.shortcuts import reverse
from django.test import TestCase
from .models import MRCResult, Topic

def create_topic(topic, description, context):
    return Topic.objects.create(topic = topic, 
                                description = description,
                                context = context)

class TopicListViewTests(TestCase):
    
    def test_topic_initialization(self):
        """This tests if the topics table is filled from the specified text 
        files if it happens to be empty"""
        url = reverse("mrc:questions_page")
        self.client.get(url)
        self.assertTrue(Topic.objects.all().exists())
        
    def test_description(self):
        """This tests if the topic description appears properly"""
        description = "Ask me about what I used to make this site"
        create_topic("stack", description, "I use Django.")
        url = reverse("mrc:questions_page")
        response = self.client.get(url)
        self.assertContains(response, description)
    
class MRCViewTests(TestCase):
    
    def test_valid_topic(self):
        """This tests if the page loads properly for a valid topic"""
        context = "This site uses PyTorch, Django, Docker, etc."
        topic = create_topic("stack", "", context)
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        
    def test_invalid_topic(self):
        """This tests if a 404 appears if the user tries an invalid url 
        pattern"""
        context = "This site uses PyTorch, Django, Docker, etc."
        create_topic("stack", "", context)
        url = reverse("mrc:detail_page", args = ("background", ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)
        
    def test_environment_reset(self):
        """This tests if the session variables that get set during the pipeline
        and redirect process get reset once the user gets back to this page"""
        context = "This site uses PyTorch, Django, Docker, etc."
        topic = create_topic("stack", "", context)
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
        context = "This site uses PyTorch, Django, Docker, etc."
        topic = create_topic("stack", "", context)
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        question = "What web framework does this site use?"
        self.client.post(url, {"question_text": question})
        self.assertEqual(MRCResult.objects.count(), 1)
        
    def test_redirect(self):
        """This tests if the redirect works as it's supposed to"""
        context = "This site uses PyTorch, Django, Docker, etc."
        topic = create_topic("stack", "", context)
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        question = "What web framework does this site use?"
        response = self.client.post(url, {"question_text": question})
        redirect_url = reverse("mrc:submitted_page", args = (topic.topic, ))
        self.assertRedirects(response, redirect_url)
    
    def test_known_answer(self):
        """This will be the test to see if the MRC model can answer questions
        that it is supposed to know"""
        context = "This site uses Django as a web framework."
        topic = create_topic("stack", "", context)
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        question = "What web framework does this site use?"
        self.client.post(url, {"question_text": question})
        answer = self.client.session["answer"]
        self.assertEqual(answer, "django")
    
    def test_unknown_answer(self):
        """This will be the test to see if the MRC model outputs a suitable
        response to questions that it is not supposed to know"""
        context = "This site uses Django as a web framework."
        topic = create_topic("stack", "", context)
        url = reverse("mrc:detail_page", args = (topic.topic, ))
        question = "What deep learning library does this site use?"
        self.client.post(url, {"question_text": question})
        
        answer = self.client.session["answer"]
        possible_responses = [
            "Good question.",
            "No idea honestly.",
            "Machines are too dumb to answer that question."
        ]
        self.assertIn(answer, possible_responses)
        
class ExampleQuestionTests(TestCase):
    """These tests are to see if the example questions for each topic provide
    the intended answer"""
    example_folder = "mrc/topics/examples/"
    
    def test_background_questions(self):
        # read example questions from file
        with open(self.example_folder + "background.txt") as file:
            questions = [line.strip() for line in file]
            
        answers = [
            "2016",
            "university of connecticut",
            "master of business analytics",
            "corvus insurance",
            "python",
            "full - stack data scientist or machine learning engineer"
        ]
        
        #visit main page to initialize topics database
        self.client.get(reverse("mrc:questions_page"))
        url = reverse("mrc:detail_page", args = ("background", ))
        for question, answer in zip(questions, answers):
            self.client.get(url)
            self.client.post(url, {"question_text": question})
            self.assertEqual(self.client.session["answer"], answer)
        
    def test_interests_questions(self):
        # read example questions from file
        with open(self.example_folder + "interests.txt") as file:
            questions = [line.strip() for line in file]
            
        answers = [
            "my first ep",
            "rock with synths",
            "indie and artsy",
            "center midfield",
            "college football and college basketball",
            "120 years old"
        ]
        
        #visit main page to initialize topics database
        self.client.get(reverse("mrc:questions_page"))
        url = reverse("mrc:detail_page", args = ("interests", ))
        for question, answer in zip(questions, answers):
            self.client.get(url)
            self.client.post(url, {"question_text": question})
            self.assertEqual(self.client.session["answer"], answer)
    
    def test_stack_questions(self):
        # read example questions from file
        with open(self.example_folder + "stack.txt") as file:
            questions = [line.strip() for line in file]
        
        answers = [
            "retrospective reader",
            "squad 2. 0",
            "2e - 6",
            "django",
            "a docker container",
            "traefik"
        ]
        
        #visit main page to initialize topics database
        self.client.get(reverse("mrc:questions_page"))
        url = reverse("mrc:detail_page", args = ("stack", ))
        for question, answer in zip(questions, answers):
            self.client.get(url)
            self.client.post(url, {"question_text": question})
            self.assertEqual(self.client.session["answer"], answer)

class SubmittedViewTests(TestCase):
    
    def test_valid_topic(self):
        """This tests if the page loads properly for a valid topic"""
        context = "This site uses PyTorch, Django, Docker, etc."
        topic = create_topic("stack", "", context)
        url = reverse("mrc:submitted_page", args = (topic.topic, ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        
    def test_invalid_topic(self):
        """This tests if a 404 appears if the user tries an invalid url 
        pattern"""
        context = "This site uses PyTorch, Django, Docker, etc."
        create_topic("stack", "", context)
        url = reverse("mrc:submitted_page", args = ("background", ))
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)
        
    def test_redirect_environment(self):
        """This tests if the page loads the session variables correctly from 
        the previous page and properly displays the question and answer"""
        context = "This site uses PyTorch, Django, Docker, etc."
        topic = create_topic("stack", "", context)
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
        """This tests if the correct behavior occurs when the user tries to 
        break the logic of the site and go to this page without going to the 
        mrc page first"""
        context =  "This site uses PyTorch, Django, Docker, etc."
        topic = create_topic("stack", "", context)
        url = reverse("mrc:submitted_page", args = (topic.topic, ))
        response = self.client.get(url)
        question = "Can you (the person reading this) break this website?"
        answer = "Clearly not. Nice try though."
        self.assertContains(response, question)
        self.assertContains(response, answer)