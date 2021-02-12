

import numpy as np
import cv2



#this is the cascade we just made. Call what you want
watch_cascade = cv2.CascadeClassifier('watchcascade.xml')
cal_cascade =  cv2.CascadeClassifier('calcascade.xml')


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # add this
    # image, reject levels level weights.
    watches = watch_cascade.detectMultiScale(gray, 50, 50)

    cal = cal_cascade.detectMultiScale(gray, 50, 50)
    
    
    # add this
    for (x,y,w,h) in watches:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Watch',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
    
        
    for (x,y,w,h) in cal:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Calculator',(x-w,y-h), font, 0.5, (11,255,255), 2, cv2.LINE_AA)


    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()