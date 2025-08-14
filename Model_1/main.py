import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    def __len__(self): #The dataloader needs to know how many datasets needed
        return len(self.data)
    def __getitem__(self, idx): #This method takes an index location in out dataset and return one item
        return self.data[idx]

    #property
    def classes(self):
        return self.data.classes

# dataset = PlayingCardDataset(
#     data_dir="C:/Users/X024936/Downloads/archive/train"
# )

# print(len(dataset))
#
# print(dataset[7000])
#
# image, label = dataset[4578]
# print(label)
# print(image)

#Get a dictionary associating target values with folder name

data_dir = 'C:/Users/X024936/Downloads/archive/train'

# target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}

# print(target_to_class)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# data_dir="C:/Users/X024936/Downloads/archive/train"

dataset = PlayingCardDataset(data_dir, transform)

image, label = dataset[5633]

# print(image.shape)
# print(label)

#Data loaders

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    break

# print(images.shape)
# print(labels.shape)
#
# print(labels)

#Step 2: Pytorch model

class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        #Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        #make a classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        #conduct these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

model = SimpleCardClassifier(num_classes=53)

# print(model)

example_out = model(images)

print(example_out.shape) # >>> torch.Size([32, 53]) [batch size, num_classes]

#Step 3: Training loop

#loss function

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(criterion(example_out, labels))

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_folder = 'C:/Users/X024936/Downloads/archive/train/'
valid_folder = 'C:/Users/X024936/Downloads/archive/valid/'
test_folder = 'C:/Users/X024936/Downloads/archive/test/'

train_dataset = PlayingCardDataset(train_folder, transform=transform)
val_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# device = torch.device("cuda" if torch.cuda_is_available() else "cpu")
# print(device)

num_epoch = 5 #epoch is one round of entire dataset
train_loss, val_losses = [], []

model = SimpleCardClassifier(num_classes=53)

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_loss.append(train_loss)

    #validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    #print epoch stats
    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss}, validation loss: {val_loss}")


