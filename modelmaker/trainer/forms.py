from django import forms
from django.forms import ModelForm
from .models import ModelTrainingParams, Category, File

# Create ModelTrainingParams form
class TrainingParamForm(ModelForm):
    class Meta:
        model =  ModelTrainingParams
        fields = ('model_name', 'number_of_categories')

class CategoryForm(ModelForm):

    class Meta:
        model =  Category
        fields = ('name',) # whats rendered

class CategoryFileForm(CategoryForm):
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
 
    class Meta(CategoryForm.Meta):
        model = Category
        fields = CategoryForm.Meta.fields + ('file',)   
        