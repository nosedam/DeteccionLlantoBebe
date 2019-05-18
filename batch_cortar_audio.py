import os, subprocess

directorio = "E:/BabyCrying/Data_Audio_5s/All_NoCry/"

lista_archivos = os.listdir(directorio)

for archivo in lista_archivos:

    subprocess.run("ffmpeg -ss 00:00:00.0 -i " + directorio+archivo + " -t 5s -c copy "+directorio+"_"+archivo)