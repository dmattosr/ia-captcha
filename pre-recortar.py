
import numpy as np
import cv2
import os

from utils import constants, get_filename_tag

map_letra_contador = {}


# Definicion de los valores in y Max para cara color BGR
# [BGR B Min, B Max, G Min, G Max, R Min, R Max]

BGR_BLANCO = (255, 255, 255)
BGR_NEGRO = (0, 0, 0)

Amarillo = [0, 10, 210, 255, 210, 255]
Verde = [0, 15, 165, 255, 0, 15]
Negro = [0, 10, 0, 10, 0, 10]
Blanco = [200, 255, 200, 255, 200, 255]
Rojo_claro = [0, 12, 0, 12, 200, 255]

color = {
    'negro': Negro,
    'amarillo': Amarillo,
    'verde': Verde,
    'blanco': Blanco,
    'rojo': Rojo_claro,
}



# full_filename = './imagenes/1G22.jpg'


def detectar_color(I, nombre_color):
    global Letras
    global M
    M = I.copy()
    for i in range(2, 48):
        for j in range(2, 128):
            M[i][j] = (0, 0, 0)
            if color[nombre_color][0] <= I[i][j][0] <= color[nombre_color][1]:
                if color[nombre_color][2] <= I[i][j][1] <= color[nombre_color][3]:
                    if color[nombre_color][4] <= I[i][j][2] <= color[nombre_color][5]:
                        M[i][j] = BGR_BLANCO

    # Elimina el borde negro en caso se vuelva blanco
    for i in range(0, 2):
        for j in range(0, 130):
            M[i][j] = BGR_NEGRO
    for i in range(48, 50):
        for j in range(0, 130):
            M[i][j] = BGR_NEGRO
    for i in range(0, 50):
        for j in range(0, 2):
            M[i][j] = BGR_NEGRO
    for i in range(0, 50):
        for j in range(128, 130):
            M[i][j] = BGR_NEGRO

    # Convierte BGR a escalas grises
    M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)
    # Suaviza los margenes, elimina la línea

    # Encuentra la posición del caracter
    Posicion=np.zeros(130)

    for j in range(0, 130):
        for i in range(0, 50):
            Posicion[j]=Posicion[j]+M[i][j]
    for j in range(0, 130):
        if 0<=Posicion[j]<=1000:
            for i in range(0, 50):
                M[i][j]=0
    auxminI = 999
    auxminJ = 999
    auxmaxI = 0
    auxmaxJ = 0
    for i in range(0, 50):
        for j in range(0, 130):
            if M[i][j] > 0:
                if j < auxminJ:
                    auxminJ = j
                if i < auxminI:
                    auxminI = i
    for i in range(0, 50):
        for j in range(0, 130):
            if M[i][j] > 0:
                if j > auxmaxJ:
                    auxmaxJ = j
                if i > auxmaxI:
                    auxmaxI = i
    if (auxmaxJ-auxminJ)<=5:
        for j in range(0, 130):
            for i in range(0, 50):
                M[i][j]=0
        auxminJ = 999
    if auxminI != 999 and auxminJ != 999:
        diferencia = auxmaxJ - auxminJ
        if 4<diferencia <=30:
            # Caso 1 letra
            Letra = M[auxminI:auxmaxI, auxminJ:auxmaxJ]
            Letras.append([Letra,(auxmaxJ + auxminJ)/2])
            Posicion=np.zeros(130)
            for j in range(0, 130):
                for i in range(0, 50):
                    Posicion[j]=Posicion[j]+M[i][j]
        elif diferencia <=50:
            # Caso 2 letras juntas
            Posicion=np.zeros(130)

            for j in range(0, 130):
                for i in range(0, 50):
                    Posicion[j]=Posicion[j]+M[i][j]
            Separar_Letras_juntas(M, auxminJ, auxmaxJ)
        else:
            # Caso 2 letras separadas
            Separar_Letras_separadas(M)

def Separar_Letras_separadas(Letras_sin_recortar):
    M = Letras_sin_recortar.copy()
    global Letras
    Posicion=np.zeros(130)
    for j in range(0, 130):
        for i in range(0, 50):
            Posicion[j]=Posicion[j]+M[i][j]
    bandera = 0
    contador = 0
    anterior = 0
    for i in range(130):
        if bandera == 0:
            if Posicion[i] > 0:
                bandera = 1
                minx = i
            else:
                bandera = 0
        elif bandera == 1:
            if Posicion[i] == 0.0 and anterior > 0:
                contador = 0
            elif Posicion[i] == 0.0 and anterior == 0:
                contador += 1
            if contador == 10:
                bandera = 0
                contador = 0
                maxx = i -10
                auxminI = 999
                auxminJ = 999
                auxmaxI = 0
                auxmaxJ = 0
                for n in range(0, 50):
                    for j in range(minx, maxx):
                        if M[n][j] > 0:
                            if j < auxminJ:
                                auxminJ = j
                            if n < auxminI:
                                auxminI = n
                for n in range(0,50):
                    for j in range(minx, maxx):
                        if M[n][j] > 0:
                            if j > auxmaxJ:
                                auxmaxJ = j
                            if n > auxmaxI:
                                auxmaxI = n
                Letra = M[auxminI:auxmaxI, auxminJ:auxmaxJ]
                Letras.append([Letra, (minx+maxx)/2])
        anterior = Posicion[i]

def Separar_Letras_juntas(Letras_sin_recortar, auxminJ_var, auxmaxJ_var):
    M = Letras_sin_recortar.copy()
    global Letras
    diferencia = (auxmaxJ_var - auxminJ_var)/2
    Letras_unidas = M[0:50, auxminJ_var:auxmaxJ_var]
    a = 0
    b = round(diferencia)
    c = round(diferencia)
    d = round(diferencia*2)
    try:
        Letra1_sin_recortar = Letras_unidas[0:50, a:b]
    except Exception as e:
        print(e)
    try:
        Letra2_sin_recortar = Letras_unidas[0:50, c:d]
    except Exception as e:
        print(e)
    auxminI = 999
    auxminJ = 999
    auxmaxI = 0
    auxmaxJ = 0
    try:
        for n in range(0, 50):
            for j in range(0, b):
                if Letra1_sin_recortar[n][j] > 0:
                    if j < auxminJ:
                        auxminJ = j
                    if n < auxminI:
                        auxminI = n
    except Exception as e:
        print(e)
    for n in range(0,50):
        for j in range(0, b):
            if Letra1_sin_recortar[n][j] > 0:
                if j > auxmaxJ:
                    auxmaxJ = j
                if n > auxmaxI:
                    auxmaxI = n

    try:
        Letra1_recortada = Letra1_sin_recortar[auxminI:auxmaxI, auxminJ:auxmaxJ]
        Letras.append([Letra1_recortada, auxminJ_var+diferencia/2])
    except Exception as e:
        print(e)
    auxminI = 999
    auxminJ = 999
    auxmaxI = 0
    auxmaxJ = 0
    for n in range(0, 50):
        for j in range(0, d-c):
            if Letra2_sin_recortar[n][j] > 0:
                if j < auxminJ:
                    auxminJ = j
                if n < auxminI:
                    auxminI = n
    for n in range(0,50):
        for j in range(0, d-c):
            if Letra2_sin_recortar[n][j] > 0:
                if j > auxmaxJ:
                    auxmaxJ = j
                if n > auxmaxI:
                    auxmaxI = n
    Letra2_recortada = Letra2_sin_recortar[auxminI:auxmaxI, auxminJ:auxmaxJ]
    Letras.append([Letra2_recortada, auxminJ_var+3*diferencia/2])

# -------------------------------
# Buscar las letras por colores

I = []# BGR
M = []
P = []
Letras = []

def separar_letras(full_filename, map_letra_contador):
    etiqueta = get_filename_tag(full_filename)
    global I, M, P, Letras
    I = cv2.imread(full_filename, 1)  # BGR
    M = []
    P = []
    Letras = []

    for nombre_color in color:
        try:
            detectar_color(I, nombre_color)
        except:
            pass
    medias = []

    for letra in Letras:
        medias.append(letra[1])
    medias.sort()
    i = 0
    for media in medias:
        for letra in Letras:
            if media == letra[1]:
                if i >= len(etiqueta):
                    continue
                try:
                    caracter = etiqueta[i]
                    if caracter not in constants.ETIQUETAS:
                        continue

                    if caracter not in map_letra_contador:
                        map_letra_contador[caracter] = 0
                    map_letra_contador[caracter] += 1

                    imagen = cv2.resize(letra[0], [20,30])
                    filename = '%s-%s-%s-%s.jpg' % (
                        caracter,
                        str(map_letra_contador[caracter]).zfill(3),
                        etiqueta,
                        'X%s' % (i + 1),
                    )
                    ruta_archivogenerado = '%s/%s' % (constants.RUTA_IMGS_CARACTERES, filename)
                    print('>>> out', filename)
                    cv2.imwrite(ruta_archivogenerado,imagen)
                    i+=1
                except Exception as e:
                    print(etiqueta, i)


for filename in os.listdir(constants.RUTA_IMGS_CAPTCHA):
    if not filename.endswith(constants.IMGS_EXTENSION):
        continue
    print('>>>>>>>>>>>>>>>>>>', filename, get_filename_tag(filename))
    #archivo = archivo.split(".")[0]
    separar_letras(constants.RUTA_IMGS_CAPTCHA + '/' + filename, map_letra_contador)
