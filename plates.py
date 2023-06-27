import cv2 as cv
import os
import imutils
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import pandas as pd

# Obtener la ruta del directorio actual
dir_path = os.path.dirname(os.path.realpath(__file__))

# Obtener la lista de archivos de imágenes en la carpeta 'plates'
fotos = os.listdir(dir_path + '/static/plates')

# Lista para almacenar las matrículas detectadas
matriculas = []

# Iterar sobre cada imagen
for x in fotos:
    # Leer la imagen y convertirla a formato RGB
    img = cv.imread(dir_path + '/static/plates/' + x)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Redimensionar la imagen para facilitar el procesamiento
    img = cv.resize(img, (620, 480))
    
    # Convertir la imagen a escala de grises y aplicar un filtro bilateral
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 11, 17, 17)
    
    # Detectar los bordes en la imagen usando el algoritmo Canny
    edged = cv.Canny(gray, 30, 200)
    
    # Encontrar los contornos en la imagen
    cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:12]
    screenCnt = None
    
    # Dibujar los contornos encontrados en la imagen
    image_with_contours = img.copy()
    cv.drawContours(image_with_contours, cnts, -1, (0, 255, 0), 2)
    plt.imshow(cv.cvtColor(image_with_contours, cv.COLOR_BGR2RGB))
    
    # Encontrar el contorno con cuatro lados (asumiendo que es la matrícula)
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # Crear una máscara para aislar la matrícula
    mask = np.zeros_like(img)
    cv.drawContours(mask, [screenCnt], -1, (255, 255, 255), -1)
    license_plate = cv.bitwise_and(img, mask)
    
    # Recortar la imagen de la matrícula
    indices = np.where(mask == 255)
    x = indices[0]
    y = indices[1]
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx, topy + 7:bottomy]
    
    # Utilizar OCR para leer el texto de la matrícula
    reader = easyocr.Reader(['es'])
    result = reader.readtext(Cropped)
    x = result[0][1]
    matriculas.append(x)

# Crear un DataFrame con las matrículas detectadas
df = pd.DataFrame({'Plates': matriculas})

# Guardar el DataFrame como un archivo CSV
file_path = os.path.join(dir_path, 'matriculas.csv')
df.to_csv(file_path)
