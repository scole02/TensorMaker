from django.shortcuts import render
from django.http import HttpResponseRedirect, Http404
from .forms import TrainingParamForm

def label(request):
    return render(request, 'trainer/label.html')
    
def setup_page(request):
    submitted = False
    if(request.method == "POST"):
        form = TrainingParamForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('setup?submitted=True')
                
    else:
        form = TrainingParamForm
        if 'submitted' in request.GET:
            submitted = True
            
    return render(request, 'trainer/setup.html', {'form':form, 'submitted':submitted})

# Create your views here.
def index(request):

    return render(request, 'trainer/base.html')