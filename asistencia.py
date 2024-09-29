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

#Tomar una imagen de la camara web
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Leer la imagen de la camara
exito, imagen = captura.read()

if not exito:
    print('No se tomo la captura')
else:
    #Reconocer cara en la captura
    cara_captura = fr.face_locations(imagen)

    #Codificar la cara capturada
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)

    #Buscar las coicidencias entre la lista de empleados
    for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):

        coicidencias = fr.compare_faces(lista_empleados_codificada, caracodif)
        distancias = fr.face_distance(lista_empleados_codificada, caracodif)

        print(distancias)

        indice_coicidencia = np.argmin(distancias)

        #Mostrar coicidencias
        if distancias[indice_coicidencia] > 0.6:
            print('NO es un empleado')
        else:
            #Buscar el nombre del empleado encontrado
            nombre = nombres_empleados[indice_coicidencia]

            #Mostrar el rectangulo de la imagen
            y1, x2, x1, y2 = caraubic

            #Crear el rectangulo
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.rectangle(imagen, (x1, y2 - 35), (x2, y2), (0,255,0), cv2.FILLED)
            cv2.putText(imagen, nombre, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            #Mostrar la imagen obtenida
            cv2.imshow('Imagen web', imagen)
            #Mantener la ventana abierta
            cv2.waitKey(0)
            print(f'Bienvendio {nombre}')
