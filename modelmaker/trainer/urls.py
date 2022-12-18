from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('setup', views.setup_page, name='setup'),
]