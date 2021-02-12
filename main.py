# импортируем нужные библиотеки
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from network import MaskDetector

face_detector = MTCNN() # запустим классификатор
video_capture = cv2.VideoCapture(0)

mask_classificator = MaskDetector()
CLASSES = ["NO MASK", "Mask on"]

while True:
    ret, frame = video_capture.read() # возьмём снимок экрана с веб-камеры
    frame = cv2.flip(frame, -1)
    faces = face_detector.detect_faces(frame) # прогоним снимок через классификатор
    for face in faces: # проитерируемся по каждому найденному лицу
        x, y, w, h = face["box"]
        cropped_frame = frame[y:y+h, x:x+w].copy()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # нарисуем рамку вокруг лица

        # приведем изображение к виду, пригодному для подачи в нейронную сеть
        cropped_frame = cv2.resize(cropped_frame, (128, 128))
        cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        cropped_frame = cv2.normalize(cropped_frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cropped_frame = np.transpose(cropped_frame, (2, 1, 0))

        predict = mask_classificator.predict(cropped_frame)
        cv2.putText(frame, CLASSES[np.argmax(predict)], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(CLASSES[np.argmax(predict)])
        print(predict)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
