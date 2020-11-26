import cv2
import os
import imutils
harcascade = '/home/jitshil/Desktop/Pantech-solution/day5/file/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(harcascade)
path='basak.mp4'
webcam = cv2.VideoCapture(path)
# url ='http:192.168.43.1:8080/video'
# webcam = cv2.VideoCapture(url)
# img = cv2.imread('jitshil.jpg')
image_path = '/home/jitshil/Desktop/Pantech-solution/day7/img/Dr. Arun Kumar Basak/'
if not os.path.isdir(image_path):
	os.mkdir(image_path)


width,height = 100, 100

count = 1
while (count < 201):
    print(count)
    __,img=webcam.read()
    img = imutils.resize(img, width=700)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)
    
    
    for (x,y,w,h) in faces: 
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        onlyFace = gray[y:y+h, x:x+w]
        resizeImg = cv2.resize(onlyFace, (width, height))
        # key1 = cv2.waitKey(1) & 0xFF    
        # if key1 == ord('s'):
            # cv2.imwrite('%s/%s.png' %(image_path,count), resizeImg)
            # count += 1
        cv2.imwrite('%s/%s.png' %(image_path,count), resizeImg)
        count += 1    
    cv2.imshow('facedetection', img)    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
webcam.release() 
cv2.destroyAllWindows()   
