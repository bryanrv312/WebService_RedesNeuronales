# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:28:09 2022

@author: bryam
"""

from tensorflow.keras.datasets import fashion_mnist
import imageio


# Agregamos las variables de entrenamiento y testeo 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Guardar las img y su localzacion
for i in range(5):
    imageio.imwrite("uploads/{}.png".format(i),im=x_test[i])
    
    
import os
import requests
import numpy as np
import tensorflow as tf


# Para guardar imagenes  y leerlas
from imageio import imwrite, imread

# Crear con FLASK
from flask import Flask, request, jsonify


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#Normalizar imgs a 255px
x_train = x_train / 255
x_test = x_test / 255

# Reformar datos 
x_train = x_train.reshape(-1,28*28)
x_train.shape
x_test = x_test.reshape(-1,28*28)
x_test.shape


#Creando la red neuronal
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dense(units = 64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units = 64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units = 64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units=10,activation='softmax'))


# Compilando el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

# Entrenando al modelo
model.fit(x_train, y_train, epochs=5)

# Evaluación de la RN
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Test Accuracy: {}'.format(test_accuracy))
   
score = model.evaluate(x_test, y_test)
print("\n %s: %.2f%%" % (model.metrics_names[1],score[1]*100))



# generamos el modelo en formato json
json_model = model.to_json()

# guardamos el modelo en una arquitectura de archivo json
with open('fashionmnist_model.json', 'w') as json_file:
    json_file.write(json_model)
    
# Grabamos los pesos del modelo
model.save_weights('FashionMNIST_weights.h5')

# cargamos los archivos generados 
with open('fashionmnist_model.json', 'r') as f:
    model_json = f.read()
    
# se carga el archivo para q valla al webservice
model = tf.keras.models.model_from_json(model_json)


model.load_weights("FashionMNIST_weights.h5")


# Crear la aplicacion Flask
app = Flask(__name__)

# Crear API de clasificación de imagenes
@app.route("/api/v1/<string:img_name>", methods=["POST"])  # RUTA
def classify_image_1(img_name):
    upload_dir = "uploads/"
    image = imread(upload_dir + img_name)
    classes = ["T-shirt/top", "Trouser", "pullover", "Dress", "Coat", "Sandal", "Shirt", "Bag", "Ankle boot"]
    
# Uso de imagenes de 28x28 para la prediccion
    prediccion = model.predict([image.reshape(1, 28*28)])
    return jsonify({"object_detected":classes[np.argmax(prediccion[0])]})


# Iniciar app de Flask y hacer predicciones
app.run(port=7015, debug=(False))





###################################################################################################################
####################### HACIENDO MIS PROPIAS PREDICCIONES #########################################################
###################################################################################################################

def classify_image_11(nameimg):
    upload_dir = "uploads/"
    image = imread(upload_dir + nameimg)
    classes = ["T-shirt/top", "Trouser", "pullover", "Dress", "Coat", "Sandal", "Shirt", "Bag", "Ankle boot"]
    predi = model.predict([image.reshape(1, 28*28)])
    #return jsonify({"object_detected":classes[np.argmax(prediccion[0])]})
    return classes[np.argmax(predi[0])]


dato = classify_image_11("2.png")
print("La imagen es: ", dato)


imm = imread("uploads/7.png")
imm = imm.reshape(1, 28*28)














