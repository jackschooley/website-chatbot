from django.views import generic

from .models import MRCResults

# Create your views here.
class IndexView(generic.ListView):
    template_name = "mrc/home.html"
    
    def get_queryset(self):
        return None

class DetailView(generic.DetailView):
    model = MRCResults
    template_name = "mrc/detail.html"