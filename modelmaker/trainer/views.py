from django.shortcuts import render
from django.http import HttpResponseRedirect, Http404
from .forms import TrainingParamForm, CategoryForm, CategoryFileForm
from django.forms import formset_factory
from .models import Image, Category, ModelTrainingParams

def label(request):
    return render(request, 'trainer/label.html')

def fileupload():
    pass

def save_categories(request, params_id):
    trained = False
    if(request.method == "POST"):
        CategoryFormSet = formset_factory(CategoryFileForm, extra=3)
        formset = CategoryFormSet(request.POST, request.FILES)
        print(request.FILES)
        for form, form_key in zip(formset, request.FILES.keys()): # loop through categories
            if form.is_valid():
                print("saving")
                category = form.save(commit=False)
                category.model = ModelTrainingParams.objects.get(pk=params_id)
                category.save()
                # print(form.clean())
                # for form_key in request.FILES.keys():
                files = request.FILES.getlist(form_key)
                # print(form_key)
                # print(files)
                for f in files:
                    new_file, created = Image.objects.get_or_create(category=Category(id=category.id), file=f)
                    new_file.save()
            else:
                # temporary you should just render a 
                print("form not valid")
                print(form.errors)
                print(form.clean())

                submitted = True
                error = True
                # get the parameters model and make a form
                params_model = ModelTrainingParams.objects.get(pk=params_id)

                return render(request, 'trainer/setup.html', {'params_model':params_model, 'submitted':submitted, 'error':error}) 

            #     for files in request.FILES.values(): # save all files to each category
            #         print(files)
            #         for f in files:
            #             File.objects.get_or_create(category=Category(id=category.id, file=f))
            # else:
            #     print("form not valid")
            #     print(form.errors)
        # f = CategoryForm(request.POST, request.FILES)
        # if f.is_valid():
        #     f.save()
        # else:
        #     print(f.errors)    
        # print(request.POST)
        # print(request.FILES)
        
    # else:    
    return render(request, 'trainer/training.html')
   

def setup_page(request):
    submitted = False
    
    if(request.method == "POST"):
        params_form= TrainingParamForm(request.POST)
        if params_form.is_valid():
            params_model = params_form.save()
            number_categories = params_form.cleaned_data['number_of_categories']
            CategoryFormSet = formset_factory(CategoryFileForm, extra=number_categories)
            # formset = CategoryFormSet(initial = [{'model_id':params_obj.id}])
            return render(request, 'trainer/setup.html', {'submitted':True, 'formset':CategoryFormSet, 'params_model':params_model})
                
    else:
        params_form = TrainingParamForm
        if 'submitted' in request.GET:
            submitted = True
            
    return render(request, 'trainer/setup.html', {'params_form':params_form, 'submitted':submitted, 'error': False})

# Create your views here. 
def index(request):

    return render(request, 'trainer/base.html')
