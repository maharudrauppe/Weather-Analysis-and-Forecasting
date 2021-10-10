import cv2
import numpy as np
import cv2
import urllib.request
import numpy as np
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:
    url = "http://10.10.10.103:8080/shot.jpg"
    response = urllib.request.urlopen(url).read()
    imgnp = np.array(bytearray(response),dtype = np.uint8)
    img = cv2.imdecode(imgnp,-1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        img1 = img[y:y+h,x:x+h]
        img2 = gray[y:y+h,x:x+h]
        eyes = eye_cascade.detectMultiScale(img2)
        for (ex,ey,eh,ew) in eyes:
            cv2.rectangle(img1,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow("face detection",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
