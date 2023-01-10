from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

app_name = 'trainer'

urlpatterns = [
    path('', views.index, name='index'),
    path('setup', views.setup_page, name='setup'),
    path('label', views.label, name='label'),
    path('<int:params_id>/save_categories', views.save_categories, name='save_categories'),
    path('<int:params_id>/training', views.training, name='training'),
    path('<uuid:task_id>/get_training_progress', views.get_training_progress, name='get_training_progress'),

]