import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCEWithLogitsLoss
from tqdm import tqdm

from utils import my_transforms, my_device

#where you want to save new model
save_path = './cpu_working.pth'

train_path='./dataset/forest_fire_dataset/train/'
test_path='./dataset/forest_fire_dataset/test/'
#datasets
train_data = ImageFolder(train_path,transform=my_transforms)
test_data = ImageFolder(test_path,transform=my_transforms)
# dataloader
trainloader = DataLoader(train_data, shuffle = True, batch_size=16)
testloader = DataLoader(test_data, shuffle = True, batch_size=16)
# Count images
# train_count = len(os.listdir(train_path))
# test_count = len(os.listdir(test_path))

def make_train_step(model, optimizer, loss_fn):
    def train_step(x, y):
        #make prediction
        yhat = model(x)
        #enter train mode
        model.train()
        #compute loss
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss
    return train_step

model = models.resnet18(pretrained=True)

#freeze all params
for params in model.parameters():
    params.requires_grad_ = False

#add a new final layer
nr_filters = model.fc.in_features  #number of input features of last layer
model.fc = nn.Linear(nr_filters, 1)

model = model.to(my_device)

#loss function
loss_fn = BCEWithLogitsLoss()
#optimizer
optimizer = torch.optim.Adam(model.fc.parameters()) 
#train step
train_step = make_train_step(model, optimizer, loss_fn)

losses = []
val_losses = []
epoch_train_losses = []
epoch_test_losses = []
n_epochs = 2

for epoch in range(n_epochs):
    epoch_loss = 0
    for i ,data in tqdm(enumerate(trainloader), total = len(trainloader)):
        x_batch , y_batch = data
        x_batch = x_batch.to(my_device) 
        y_batch = y_batch.unsqueeze(1).float()
        y_batch = y_batch.to(my_device)
        loss = train_step(x_batch, y_batch)
        epoch_loss += loss/len(trainloader)
        losses.append(loss)
        epoch_train_losses.append(epoch_loss)
        print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))
        # validation doesnt requires gradient
    with torch.no_grad():
        cum_loss = 0
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(my_device)
            y_batch = y_batch.unsqueeze(1).float()
            y_batch = y_batch.to(my_device)
            #model to eval mode
            model.eval()
            yhat = model(x_batch)
            val_loss = loss_fn(yhat,y_batch)
            cum_loss += loss/len(testloader)
            val_losses.append(val_loss.item())
        epoch_test_losses.append(cum_loss)
        print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))  
        best_loss = min(epoch_test_losses)
        if cum_loss <= best_loss:
            best_model_wts = model.state_dict()
        
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), save_path)