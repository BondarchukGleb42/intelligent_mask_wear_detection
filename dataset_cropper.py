# установим нужные библиотеки
import cv2
from cv2 import CascadeClassifier
import glob # библиотека для создания удобной работы с путями

idx = 0
faceCascade = CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')
for img_path in glob.glob('dataset\\no_mask_no_cropped' + '/*'): # итерируемся по каждому изображению в папке
    image = cv2.imread(img_path) # считаем изображение
    try: # данная конструкция нужна для того, чтобы пропускать файлы неподходящего расширения
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
    except:
        continue

    for (x, y, w, h) in faces: # итерируемся по каждому найденному на изображение лицу
        cropped_img = image[y:y+h, x:x+w]
        cropped_img = cv2.resize(cropped_img, (256, 256)) # изменим размер изображения до 64x64
        cv2.imwrite(f"dataset\\no_mask\\img{idx}.jpg", cropped_img) # сохраним изображение в нужную папку
        idx += 1
