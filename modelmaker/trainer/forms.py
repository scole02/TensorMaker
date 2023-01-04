from django import forms
from django.forms import ModelForm, ClearableFileInput
from .models import ModelTrainingParams, Category, Image

# Create ModelTrainingParams form
class TrainingParamForm(ModelForm):
    class Meta:
        model =  ModelTrainingParams
        fields = ('model_name', 'number_of_categories')

class CategoryForm(ModelForm):

    class Meta:
        model =  Category
        exclude = ("model", )
        fields = ('name', ) # whats rendered

class CategoryFileForm(CategoryForm):
    file = forms.ImageField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
 
    class Meta(CategoryForm.Meta):
        model = Category
        fields = CategoryForm.Meta.fields   
        widgets =  {
            'images': ClearableFileInput(attrs={'multiple': True}),
        }
        