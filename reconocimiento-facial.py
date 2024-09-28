import sys
import cv2
import face_recognition as fr
import numpy as np 

print("Estoy funcionando")

#Cargar imagenes
foto_control = fr.load_image_file('C:\Users\samhu\Documents\controlador-de-asistencia\Imagenes\FotoA.jpg')
foto_prueba = fr.load_image_file('C:\Users\samhu\Documents\controlador-de-asistencia\Imagenes\FotoB.jpg')

#Cambiar el formato de las imagenes
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)

#Mostrar Imagenes