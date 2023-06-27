import cv2 as cv
import os

# Nos posicionamos en el mismo directorio en el que queremos ejecutar el script
dir_path = os.path.dirname(os.path.realpath(__file__))

# Cargamos la imagen
img = cv.imread(dir_path + '/static/test/23.jpeg')
cv.imshow('Grupo de 4 personas', img)

# Convertimos la imagen a escala de grises
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Redimensionamos la imagen a un tamaño específico
img_small = cv.resize(gray, (800, 400))

# Aplicamos un filtro de desenfoque a la imagen
img_filtered = cv.blur(img_small, (3, 3))

cv.imshow('Personas en gris', img_filtered)

# Cargamos el clasificador de cascada Haar para la detección de caras
haar_cascade = cv.CascadeClassifier(dir_path + '/models/haar_face.xml')

# Detectamos las caras en la imagen utilizando el clasificador Haar
faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.01, minNeighbors=19, minSize=(50, 50))

# Imprimimos el número de caras detectadas
print(f'Número de caras = {len(faces_rect)}')

# Dibujamos rectángulos alrededor de las caras detectadas en la imagen original
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

cv.imshow('Caras detectadas', img)

cv.waitKey(0)
