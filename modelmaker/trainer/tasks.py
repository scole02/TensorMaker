from celery import shared_task, current_task
from .models import Image, Category, ModelTrainingParams
from .train import start_training
from .celery import app
from time import sleep

# @shared_task
# def dummy_train_model():
#     print("starting training")
#     for i in range(20):
#       print(f'Epoch {i}')
#       sleep(1)  

@app.task(bind=True)
def train_model(self, params_id):
    print(params_id)
    print(self.request.id)
    start_training(self, params_id)
