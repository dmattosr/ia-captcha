

import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


class Constants():
    IMGS_EXTENSION = '.jpg'
    RUTA_IMGS_CAPTCHA = './imagenes/captcha'
    RUTA_IMGS_CARACTERES = './imagenes/corregidos'
    ETIQUETAS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
    INDICE_ETIQUETA = dict(
        zip(ETIQUETAS, list(range(len(ETIQUETAS))))
    )
    DIMENSIONES = (30, 20, 1)  # alto x ancho x capas
    REPETIR_SHUFFLE = 100
    PORCENTAJE_ENTRENAMIENTO = 0.7
    PORCENTAJE_PRUEBAS = 1 - PORCENTAJE_ENTRENAMIENTO


constants = Constants()


def get_filename_tag(full_filename):
    return os.path.basename(full_filename)[:4]


def get_caracter(nombre_archivo):
    # caracter = nombre_archivo[-5]
    return nombre_archivo[0]



def cargar_data():
    contador_caracter = defaultdict(lambda: 0)

    lista_archivos = os.listdir(constants.RUTA_IMGS_CARACTERES)
    contador = 0
    while contador <= constants.REPETIR_SHUFFLE:
        random.shuffle(lista_archivos)
        contador += 1

    list_file_train = random.sample(lista_archivos, int(constants.PORCENTAJE_ENTRENAMIENTO * len(lista_archivos)))
    list_file_test = list(set(lista_archivos) - set(list_file_train))

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    lista_imagenes = []

    for (x, y, lista_imagenes) in ((x_train, y_train, list_file_train), (x_test, y_test, list_file_test)):
        for nombre_archivo in lista_imagenes:
            caracter = get_caracter(nombre_archivo)
            if caracter not in constants.ETIQUETAS:
                continue

            contador_caracter[caracter] += 1
            ruta_archivo = os.path.join(constants.RUTA_IMGS_CARACTERES, nombre_archivo)

            imagen = plt.imread(ruta_archivo)
            y.append(constants.INDICE_ETIQUETA[caracter])
            x.append(imagen)
    x_train = np.array(x_train, dtype=np.uint8)
    y_train = np.array(y_train)
    x_test = np.array(x_test, dtype=np.uint8)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)
