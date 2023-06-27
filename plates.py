import cv2 as cv
import os 
import imutils
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import pandas as pd



dir_path = os.path.dirname(os.path.realpath(__file__))

fotos = os.listdir(dir_path + '/static/plates')

matriculas = []


for x in fotos:
    img = cv.imread(dir_path + '/static/plates/'+ x)
    print(x)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (620,480) )
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 11, 17, 17)
    edged = cv.Canny(gray, 30, 200)

    cnts = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:12]
    screenCnt = None

    image_with_contours = img.copy()
    cv.drawContours(image_with_contours, cnts, -1, (0, 255, 0), 2) 
    plt.imshow(cv.cvtColor(image_with_contours, cv.COLOR_BGR2RGB))

    for c in cnts:
                # approximate the contour
                peri = cv.arcLength(c, True)
                approx = cv.approxPolyDP(c, 0.018 * peri, True)
                # si nuestro contorno mas similar tiene 4 lados, entonces
                # asumimos que lo hemos encontrado, por eso paramos
                if len(approx) == 4:
                      screenCnt = approx
                      break
                
    mask = np.zeros_like(img)
    cv.drawContours(mask, [screenCnt], -1, (255, 255, 255), -1)
    license_plate = cv.bitwise_and(img, mask)
    

    indices = np.where(mask == 255)
    x = indices[0]
    y = indices[1]
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    # Topx +1 recortamos desde arriba, Bottomx +1 alargamos hacia abajo (Vertical)
    # Topy +1 acorta hacia la derecha, Bottomy +1 alarga hacia la derecha (Horizontal)
    Cropped = gray[topx:bottomx, topy+7:bottomy]

    reader = easyocr.Reader(['es'])
    result = reader.readtext(Cropped)
    x = result[0][1]
    matriculas.append(x)

df = pd.DataFrame({'Plates':matriculas})
file_path = os.path.join(dir_path, 'matriculas.csv')
df.to_csv(file_path)