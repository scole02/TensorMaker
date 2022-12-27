from django.shortcuts import render
from django.http import HttpResponseRedirect, Http404
from .forms import TrainingParamForm, CategoryForm, CategoryFileForm
from django.forms import formset_factory

def label(request):
    return render(request, 'trainer/label.html')

def fileupload():
    pass

def save_categories(request):
    trained = False
    if(request.method == "POST"):
        CategoryFormSet = formset_factory(CategoryFileForm, extra=3)
        formset = CategoryFormSet(request.POST, request.FILES)
        for form in formset:
            if form.is_valid():
                print("saving")
                form.save()
            else:
                print("form not valid")
                print(form.errors)
        # f = CategoryForm(request.POST, request.FILES)
        # if f.is_valid():
        #     f.save()
        # else:
        #     print(f.errors)    
        print(request.POST)
        print(request.FILES)
        return render(request, 'trainer/base.html')

    # else:    
    return render(request, 'trainer/training.html')
   

def setup_page(request):
    submitted = False
    if(request.method == "POST"):
        params_form= TrainingParamForm(request.POST)
        if params_form.is_valid():
            params_form.save()
            number_categories = params_form.cleaned_data['number_of_categories']
            CategoryFormSet = formset_factory(CategoryFileForm, extra=number_categories)
            


            return render(request, 'trainer/setup.html', {'submitted':True, 'formset':CategoryFormSet, 'params_form':params_form})
                
    else:
        params_form = TrainingParamForm
        if 'submitted' in request.GET:
            submitted = True
            
    return render(request, 'trainer/setup.html', {'params_form':params_form, 'submitted':submitted})

# Create your views here. 
def index(request):

    return render(request, 'trainer/base.html')
