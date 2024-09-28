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

#Localizar la cara de la foto control
lugar_cara_a = fr.face_locations(foto_control)[0]
cara_codificada_a = fr.face_encodings(foto_control)[0]

lugar_cara_b = fr.face_locations(foto_prueba)[0]
cara_codificada_b = fr.face_encodings(foto_prueba)[0]

#Mostrar rectangulo
#print(lugar_cara_a)
cv2.rectangle(foto_control, #Imagen donde dibujar rectangulo
              (lugar_cara_a[3], lugar_cara_a[0]), #Punto superior del rectangulo
              (lugar_cara_a[1], lugar_cara_a[2]), #Punto inferior del rectangulo
              (0,255,0), #Color del rectangulo
              2 #Grosor del borde del rectangulo
)

cv2.rectangle(foto_prueba, #Imagen donde dibujar rectangulo
              (lugar_cara_b[3], lugar_cara_b[0]), #Punto superior del rectangulo
              (lugar_cara_b[1], lugar_cara_b[2]), #Punto inferior del rectangulo
              (0,255,0), #Color del rectangulo
              2 #Grosor del borde del rectangulo
)

#Realizar comparacion
resultado = fr.compare_faces([cara_codificada_a], #Lista de las caras a comparar
                             cara_codificada_b, #Imagen a comparar
                             0.3 #Tolerancia
)
print(resultado)

#Mostrar Imagenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

#Medida de distancia
distancia = fr.face_distance([cara_codificada_a], cara_codificada_b)
print(distancia)

#Mantener el programa abierto
cv2.waitKey(0)