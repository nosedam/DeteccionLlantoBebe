import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display

import matplotlib.pyplot as plt # for plotting
import numpy as np

import os

directorio = 'E:/BabyCrying/Data_Youtube/Bebe/Bebe Llorando (Efecto de Sonido)/'
directorio_espectrogramas = 'E:/BabyCrying/Espectrogramas_bebe/'

lista_archivos = os.listdir(directorio)

for archivo in lista_archivos:

        if archivo[-3:] != 'png':

                print(archivo)

                audio, sample_rate = librosa.load(directorio+archivo, sr=None)

                melspec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, fmin=1000, fmax=5000)
                mel_spec_db = librosa.amplitude_to_db(melspec, ref=np.max)
                mel_spec_pw = librosa.power_to_db(melspec, ref=np.max)

                imagen = plt.figure(figsize=(4, 4))
                librosa.display.specshow(mel_spec_pw, fmin=1000,fmax=5000)

                plt.tight_layout()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                plt.gca().yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                imagen.savefig(directorio_espectrogramas+directorio.split("/")[-2]+"_"+archivo[:-4]+".png", bbox_inches = 'tight', pad_inches = 0)
                plt.close()