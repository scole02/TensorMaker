from django.contrib import admin
from .models import ModelTrainingParams, Category, Image


# Register your models here.
admin.site.register(ModelTrainingParams)
admin.site.register(Category)
admin.site.register(Image)