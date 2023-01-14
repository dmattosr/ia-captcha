from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils

import matplotlib.pyplot as plt
import numpy as np

from utils import cargar_data, constants

#
# Lectura, visualización y pre-procesamiento de los datos
#

(x_train, y_train), (x_test, y_test) = cargar_data()


# Visualizaremos 16 imágenes aleatorias tomadas del set x_train
ids_imgs = np.random.randint(0, x_train.shape[0], 16)
for i in range(len(ids_imgs)):
    img = x_train[ids_imgs[i], :, :]
    plt.subplot(4, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(y_train[ids_imgs[i]])
plt.suptitle('16 imágenes del set MNIST')
plt.show()
# plt.waitforbuttonpress()

# Pre-procesamiento: para introducirlas a la red neuronal debemos
# "aplanar" cada una de las imágenes en un vector de 28x28 = 784 valores

X_train = np.reshape(
    x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

# Adicionalmente se normalizarán las intensidades al rango 0-1
X_train = X_train/255.0
X_test = X_test/255.0

# Finalmente, convertimos y_train y y_test a representación "one-hot"
nclasses = len(constants.ETIQUETAS)
Y_train = np_utils.to_categorical(y_train, nclasses)
Y_test = np_utils.to_categorical(y_test, nclasses)

#
# Creación del modelo:
# - Capa de entrada: su dimensión será 784 (el tamaño de cada imagen aplanada)
# - Capa oculta: 15 neuronas con activación ReLU
# - Capa de salida: función de activación 'softmax' (clasificación multiclase) y un
#     total de 10 categorías
#

np.random.seed(1)		# Para reproducibilidad del entrenamiento
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]

modelo = Sequential()
modelo.add(Dense(15, input_dim=input_dim, activation='relu'))
modelo.add(Dense(output_dim, activation='softmax'))
print(modelo.summary())

#
# Compilación y entrenamiento: gradiente descendente, learning rate = 0.05, función
# de error: entropía cruzada, métrica de desempeño: precisión

sgd = SGD(learning_rate=0.2)
modelo.compile(loss='categorical_crossentropy',
               optimizer=sgd, metrics=['accuracy'])

# Para el entrenamiento se usarán 30 iteraciones y un batch_size de 1024
num_epochs = 50
batch_size = 1024
historia = modelo.fit(X_train, Y_train, epochs=num_epochs,
                      batch_size=batch_size, verbose=2)

#
# Resultados
#

# Error y precisión vs iteraciones
plt.subplot(1, 2, 1)
plt.plot(historia.history['loss'])
plt.title('Pérdida vs. iteraciones')
plt.ylabel('Pérdida')
plt.xlabel('Iteración')

plt.subplot(1, 2, 2)
plt.plot(historia.history['accuracy'])
plt.title('Precisión vs. iteraciones')
plt.ylabel('Precisión')
plt.xlabel('Iteración')

plt.show()

# Calcular la precisión sobre el set de validación
puntaje = modelo.evaluate(X_test, Y_test, verbose=0)
print('Precisión en el set de validación: {:.1f}%'.format(100*puntaje[1]))

# Realizar predicción sobre el set de validación y mostrar algunos ejemplos
# de la clasificación resultante
Y_pred = modelo.predict(X_test)

ids_imgs = np.random.randint(0, X_test.shape[0], 9)
for i in range(len(ids_imgs)):
    idx = ids_imgs[i]
    img = X_test[idx, :].reshape(*constants.DIMENSIONES[:2])
    # img = X_test[idx, :].reshape(30, 20)
    # img = X_test[idx, :]
    cat_original = np.argmax(Y_test[idx, :])
    cat_prediccion = Y_pred[idx]

    plt.subplot(3, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('"{}" clasificado como "{}"'.format(
        cat_original, cat_prediccion))
plt.suptitle('Ejemplos de clasificación en el set de validación')
plt.show()


"""
Fuente:
https://www.codificandobits.com/blog/tutorial-clasificacion-imagenes-redes-neuronales-python/
"""