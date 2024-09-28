import sys
import cv2
import face_recognition as fr
import numpy as np 

print("Estoy funcionando")

#Cargar imagenes
foto_control = fr.load_image_file('Imagenes/FotoA.jpg')
foto_prueba = fr.load_image_file('Imagenes/FotoB.jpg')

#Cambiar el formato de las imagenes
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#Mostrar Imagenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

#Mantener el programa abierto
cv2.waitKey(0)