import numpy as np
from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render
from io import BytesIO
import base64
from PIL import Image as im
#from .forms import NameForm
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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
import time
import os, shutil
import copy
import mpld3
from mpld3 import plugins, utils

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
    mpld3.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    mpld3.imshow(inp)

def data(request):
    if request.method == "POST":
        #
        # Django image API 
        #
        train_split = 0.8
        # form = NameForm(request.POST)
        files = request.FILES.getlist('classFiles1')
        className = request.POST.get('classinput')
        submitbutton= request.POST.get('Submit')
        print(className)
        # temp_file = dir[0].temporary_file_path
        paths = [f.temporary_file_path() for f in files]
        
        # make directory for class files
        train_dirname = os.path.join(os.getcwd(), "media", "train", className)
        val_dirname = os.path.join(os.getcwd(), "media", "val", className)
        
        # if directories do not exist, make them and copy over temp files
        if not os.path.exists(train_dirname):
            os.makedirs(train_dirname)  
            for p in paths[:int( len(paths) * train_split)]:
                new_path = os.path.join(train_dirname, os.path.basename(p))
                shutil.copy(p, new_path)
    
        if not os.path.exists(val_dirname):
            os.makedirs(val_dirname)
            for p in paths[int( len(paths) * train_split):]:
                new_path = os.path.join(val_dirname, os.path.basename(p))
                shutil.copy(p, new_path)
            

        # ImageFolder expects a folder with name of class and then images
        train_dataset = datasets.ImageFolder(os.path.dirname(train_dirname), data_transforms['train'])
        val_dataset = datasets.ImageFolder(os.path.dirname(val_dirname), data_transforms['val'])
        print(f'len train:{len(train_dataset)} len val: {len(val_dataset)}')

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                                     shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                                     shuffle=True, num_workers=4)
        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        dataset_sizes = {'train' : len(train_dataset), 'val' : len(val_dataset)}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes, dataloaders, device)


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

        return render(request, "test.html")

    else:
        return render(request, "test.html")
    
    return render(request, "test.html")

def train_model(model, criterion, optimizer, scheduler, dataset_sizes, dataloaders, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



class ImageDataset(Dataset):
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
    def __init__(self, filepaths: list):
        """
        Args:
            filepaths[] (string): list of the images filepaths.
        """
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img_name = self.filepaths[idx]
        image = io.imread(img_name)
        sample = {'image': image}

        return sample