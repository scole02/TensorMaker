from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('setup', views.setup_page, name='setup'),
    path('label', views.label, name='label'),
]