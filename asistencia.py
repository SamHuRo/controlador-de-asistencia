import cv2
import face_recognition as fr
import numpy as np 
import os

#Crear la base de datos
ruta = 'Empleados'
mis_imagenes = []
nombres_empleados = []
lista_empleados = os.listdir(ruta)

#Ciclo para capturar las imagenes
for nombre in lista_empleados:
    imagen_actual = cv2.imread(f'{ruta}/{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

#Codificar imagenes
def codificar(imagenes):
    #Crear una lista nueva
    lista_codificada = []
    #Pasar las imagenes a RGB
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        #Encontrar la cara dentro de la imagen
        codificada = fr.face_encodings(imagen)[0]
        #Agregar a la lista
        lista_codificada.append(codificada)
    
    return lista_codificada

lista_empleados_codificada = codificar(mis_imagenes)
