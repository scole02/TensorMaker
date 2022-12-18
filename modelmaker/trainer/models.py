from django.db import models
from django.forms import ModelForm
#from django.core.files import File
#from django.core.files.storage import default_storage
#from django.core.files.storage import FileSystemStorage

# Create your models here.
class ModelTrainingParams(models.Model):
    model_name = models.CharField(max_length=200)
    number_of_categories = models.IntegerField(default=2)

    def __str__(self):
        return self.model_name

class Category(models.Model):
    name = models.CharField(max_length=100)
    num_files = models.IntegerField(default=0)
    files = models.FileField(null=True)

    def __str__(self):
        return self.name