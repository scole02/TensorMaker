import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from io import BytesIO
import base64
from PIL import Image as im
# from keras.applications import vgg16
# from keras.applications.imagenet_utils import decode_predictions
# from keras.preprocessing.image import img_to_array, load_img
# from tensorflow.python.keras.backend import set_session


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.imshow(inp)

def data(request):
    if request.method == "POST":
        #
        # Django image API
        #
        dir = request.FILES
        print(dir)
        print(dir.getlist('imageFiles'))
        dir_name = default_storage.save(dir.name, dir)
        data_dir = default_storage.path(dir_name)

        # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
        #                                           data_transforms[x])
        #                   for x in ['train', 'val']}
        # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
        #                                              shuffle=True, num_workers=4)
        #               for x in ['train', 'val']}
        # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        # class_names = image_datasets['train'].classes

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # # Get a batch of training data
        # inputs, classes = next(iter(dataloaders['train']))

        # # Make a grid from batch
        # out = torchvision.utils.make_grid(inputs)

        # inp = out.numpy().transpose((1, 2, 0))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean
        # inp = np.clip(inp, 0, 1)
        # data = im.fromarray(inp)
        # data.save('test.png')

    return render(request, "data.html")

def test(request):
    if request.method == "POST":
        #
        # Django image API
        #
        dir = request.FILES["imageFile"]
        dir_name = default_storage.save(dir.name, dir)
        data_dir = default_storage.path(dir_name)

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=4)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        inp = out.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        data = im.fromarray(inp)
        data.save('test.png')

        return render(request, "test.html")

    else:
        return render(request, "test.html")
    
    return render(request, "test.html")