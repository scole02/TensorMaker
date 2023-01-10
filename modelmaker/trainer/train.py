import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
import time
import os, shutil
import copy
from .models import ModelTrainingParams, Category, Image
from typing import Tuple
# Data augmentation and normalization for training
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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        super(MyDataset, self).__init__()
        self.imgs = [] # list of path strings
        self.labels = [] # list of ints where int corresponds to class idx for that img
        self.classes = [] # list of class strings
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # returns:
        #   tuple: (img, label) where img is loaded using default loader from pytorch
        #           and target is class index
        img_name = self.imgs[idx]
        img = default_loader(img_name)
        img = self.transform(img) # "normalizes" image and makes it a tensor
        label = self.labels[idx]
        
        return img, label
    
    def addImages(self, img_path_list:list, clas:str):
        self.imgs += img_path_list
        
        # add class if not already in classes
        if clas in self.classes:
            class_idx = self.classes.index(clas)
        else:
            class_idx = len(self.classes)
            self.classes.append(clas)
        
        # set class idx for each img we added to dataset
        self.labels += ([class_idx] * len(img_path_list))

# def createClassDatasets(category:Category, train_split:float) -> Tuple[MyDataset, MyDataset]:
#     # returns (train_dataset, val_dataset)
#     category_images = list(Image.objects.filter(category=category))
#     img_paths = [p.file.path for p in category_images]

#     train_dataset = MyDataset(img_paths[:int(len(img_paths) * train_split)], data_transforms['train'])
#     val_dataset = MyDataset(img_paths[int(len(img_paths) * train_split):], data_transforms['val'])
#     return  (train_dataset, val_dataset)

def start_training(self, params_id:int):
    print(self.request.id)
    train_split = 0.8
    files = []
    className = ModelTrainingParams.objects.get(id=params_id).model_name
    
    # ImageFolder expects a folder with name of class and then images
    # train_dataset = datasets.ImageFolder(os.path.dirname(train_dirname), data_transforms['train'])
    # val_dataset = datasets.ImageFolder(os.path.dirname(val_dirname), data_transforms['val'])
    params_model = ModelTrainingParams.objects.get(pk=params_id)
    categories = Category.objects.filter(model=params_model)
    # print(categories)

    train_dataset = MyDataset(transform=data_transforms['train'])
    val_dataset = MyDataset(transform=data_transforms['val'])
    
    
    # add imgs to datasets for each class
    for c in categories:
        category_images = list(Image.objects.filter(category=c))
        img_paths = [p.file.path for p in category_images]
        train_dataset.addImages(
            img_path_list=img_paths[:int(len(img_paths) * train_split)],
            clas=c.name
        )
        val_dataset.addImages(
            img_path_list=img_paths[int(len(img_paths) * train_split):],
            clas=c.name
        )
        # print(int(len(img_paths) * train_split))
    
    print(f'len train:{len(train_dataset)} len val: {len(val_dataset)}')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                                 shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                                 shuffle=True, num_workers=4)
    
    #################################################################
    
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
    train_model(self, model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes, dataloaders, device)

def train_model(self, model, criterion, optimizer, scheduler, dataset_sizes, dataloaders, device, num_epochs=15):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        self.update_state(state='PROGRESS', 
                          meta={'current_epoch': epoch, 
                                'total_epochs': num_epochs})
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