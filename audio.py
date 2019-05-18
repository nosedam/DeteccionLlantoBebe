import sounddevice as sd

from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display

import torch
from torch.autograd import Variable

from PIL import Image

import matplotlib.pyplot as plt # for plotting
import numpy as np

from RedNeuronalAudio import ConvolucionalBebeAudioLiteV2

from io import BytesIO
import os

def convertir_audio_espectrograma(audio, sample_rate):
    melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, fmin=1000, fmax=5000)
    #mel_spec_db = librosa.amplitude_to_db(melspec, ref=np.max)
    mel_spec_pw = librosa.power_to_db(melspec, ref=np.max)

    imagen = plt.figure(figsize=(4, 4))
    #librosa.display.specshow(mel_spec_pw, fmin=1000,fmax=5000)
    librosa.display.specshow(mel_spec_pw)
    plt.tight_layout()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    #imagen.savefig(directorio_espectrogramas+directorio.split("/")[-2]+"_"+archivo[:-4]+".png", bbox_inches = 'tight', pad_inches = 0)
    imgdata = BytesIO()
    imagen.savefig(imgdata, format="png")
    imgdata.seek(0)
    resultado = Image.open(imgdata)
    plt.show()
    plt.close()
    return resultado

def predict_image(model, image):
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

i = 1

model = ConvolucionalBebeAudioLiteV2()
model.load_state_dict(torch.load("litev2_models/litev2_1_85.model"))
model.eval()

while True:

    #print("Grabando")

    X = sd.rec(int(5*44100), samplerate=44100, channels=2, blocking=True, device=1)

    #print("Grabado!")

    if X.ndim > 1: X = X[:,0]
    X = X.T

    imagen = convertir_audio_espectrograma(X, 44100)

    if predict_image(model, imagen) == 0:
        print("Bebe detectado en: " + str(i))
    else:
        print("Nada: " + str(i))

    i+=1