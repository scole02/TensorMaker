from celery import shared_task
from .models import Image, Category, ModelTrainingParams
from .train import start_training
from time import sleep

# @shared_task
# def dummy_train_model():
#     print("starting training")
#     for i in range(20):
#       print(f'Epoch {i}')
#       sleep(1)  

@shared_task
def train_model(params_id):
    print(params_id)
    start_training(params_id)
