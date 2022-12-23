from django.contrib import admin
from .models import ModelTrainingParams, Category, File

# Register your models here.
admin.site.register(ModelTrainingParams)
admin.site.register(Category)
admin.site.register(File)