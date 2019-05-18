# Import needed packages
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
import requests
import shutil
from io import open
import os
from PIL import Image
import json
from RedNeuronalAudio import ConvolucionalBebeAudioLiteV2

import os

def predict_image(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        model.to(torch.device("cuda"))

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)

    _, index = torch.max(output.data, 1)


    return index.item()


if __name__ == "__main__":

    lista_models = os.listdir("models")

    directorio_nada = "E:/BabyCrying/Espectrogramas_nada/"
    directorio_bebe = "E:/BabyCrying/Espectrogramas_bebe/"
    lista_bebe = os.listdir(directorio_bebe)
    lista_nada = os.listdir(directorio_nada)


    for dir_model in lista_models:

        model = ConvolucionalBebeAudioLiteV2()
        model.load_state_dict(torch.load("models/"+dir_model))
        model.eval()

        suma = 0
        for espectrograma in lista_bebe:
            index = predict_image(directorio_bebe+espectrograma)

            print(espectrograma + ": "+str(index))
            suma += 1 - index

        prediccion_de_bebe = suma / len(lista_bebe)
        
        suma = 0
        for espectrograma in lista_nada:
            index = predict_image(directorio_nada+espectrograma)

            #print(espectrograma + ": "+str(index))
            suma += index
        prediccion_de_nada = suma / len(lista_nada)

        print(dir_model)
        print("Precisión de bebe: " + str(prediccion_de_bebe))
        print("Precisión de nada: " + str(prediccion_de_nada))
    