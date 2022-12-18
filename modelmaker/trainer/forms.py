from django import forms
from django.forms import ModelForm
from .models import ModelTrainingParams

# Create ModelTrainingParams form
class TrainingParamForm(ModelForm):
    class Meta:
        model =  ModelTrainingParams
        fields = ('model_name', 'number_of_categories')


        