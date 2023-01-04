from django.db import models
from django.forms import ModelForm
import mimetypes
from os.path import splitext
from django.core.files.storage import FileSystemStorage
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from django.template.defaultfilters import filesizeformat
#from django.core.files import File
#from django.core.files.storage import default_storage
#from django.core.files.storage import FileSystemStorage

MAX_FILE_SIZE = 24 * 1024 * 1024 # 24 MBs 
# fs = FileSystemStorage(location='/media/imgs')

# Create your models here.
class ModelTrainingParams(models.Model):
    model_name = models.CharField(max_length=200)
    number_of_categories = models.IntegerField(default=2)

    def get_num_categories(self):
        return self.number_of_categories

    def __str__(self):
        return self.model_name

class Category(models.Model):
    model = models.ForeignKey(ModelTrainingParams, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    num_files = models.IntegerField(default=0)
    # file = models.FileField(blank=False, upload_to='PN_files/%Y/%m/%d/', verbose_name="Files", help_text=f'Allowed size is {MAX_FILE_SIZE / (1024*1024)} MBs')

    def __str__(self):
        return self.name

# https://gist.github.com/jrosebr1/2140738
# class FileValidator(object):


class Image(models.Model):
     category = models.ForeignKey(Category, on_delete=models.CASCADE)
    #  validator = FileValidator(max_size=MAX_FILE_SIZE)
     file = models.ImageField(blank=False, upload_to='PN_files/%Y/%m/%d/', verbose_name="Files", help_text=f'Allowed size is {MAX_FILE_SIZE / (1024*1024)} MBs')

# upload_to='PN_files/%Y/%m/%d/'

