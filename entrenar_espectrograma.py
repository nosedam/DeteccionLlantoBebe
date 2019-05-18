import torch # Tensor Package (for use on GPU)
from torch.autograd import Variable # for computational graphs
import torch.nn as nn ## Neural Network package
import torch.nn.functional as F # Non-linearities package
import torch.optim as optim # Optimization package
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
import torchvision # for dealing with vision data
import torchvision.transforms as transforms # for modifying vision data to run it through models
import torchaudio
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display

import matplotlib.pyplot as plt # for plotting
import numpy as np

from RedNeuronalAudio import ConvolucionalBebeAudioLiteV2

import os

traindir = "E:/BabyCrying/Data_Audio_5s/Espectrogramas/Train"
testdir = "E:/BabyCrying/Data_Audio_5s/Espectrogramas/Test"

normalizador = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.ImageFolder(root=traindir, transform=normalizador)
test_dataset = torchvision.datasets.ImageFolder(root=testdir, transform=normalizador)

train_loader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=True, num_workers=5)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True, num_workers=4)

avail = torch.cuda.is_available()

model = ConvolucionalBebeAudioLiteV2()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()

def save_models(epoch):
    torch.save(model.state_dict(), "audiobebe_{}.model".format(epoch))
    print("Modelo almacenado!")

def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):

        if avail:
            model.to(torch.device("cuda"))
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
       
        test_acc += torch.sum(prediction.data == labels.data)

    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc.item() / 335

    return test_acc

def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available

            if avail:
                model.to(torch.device("cuda"))
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)

            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            #train_loss += loss.cuda().data[0].item() * images.size(0).item()
            #train_loss += outputs.shape[0] * loss.item()

            train_loss += loss.item()

            _, prediction = torch.max(outputs.data, 1)
            
            train_acc += torch.sum(prediction.data == labels.data)

        # Evaluate on the test set
        test_acc = test()
        train_acc = train_acc.item() / 1451

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print("Entrenamiento {}, Eficacia al evaluar: {}, Eficacia del entrenamiento: {}".format(epoch, test_acc, train_acc))

if __name__ == "__main__":
    train(200)