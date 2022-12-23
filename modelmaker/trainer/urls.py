from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

app_name = 'trainer'

urlpatterns = [
    path('', views.index, name='index'),
    path('setup', views.setup_page, name='setup'),
    path('label', views.label, name='label'),
    path('save_categories', views.save_categories, name='save_categories'),
]