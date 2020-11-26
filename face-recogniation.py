import cv2, os
import numpy as np
import imutils

haar_file = '/home/jitshil/Desktop/Pantech-solution/day5/file/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = '/home/jitshil/Desktop/Pantech-solution/day7/img/'
(images, labels, names, id) = ([], [], {}, 0)


for (subdirs, dirs, files) in os.walk(datasets):
	for subdir in dirs:
		names[id] = subdir
		subject_path = os.path.join(datasets, subdir)
		for fileName in os.listdir(subject_path):
			path = subject_path+'/'+fileName
			label = id
			images.append(cv2.imread(path, 0))
			labels.append(int(label))
		id += 1


(width, height) = (100,100)
(images, labels) = [np.array(lis) for lis in [images, labels]]
#print(images, labels)
print(labels)


# model = cv2.face.LBPHFaceRecognizer_create()
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)
cam_path='basak.mp4'
# url ='http:192.168.43.1:8080/video'
webcam = cv2.VideoCapture(cam_path)


while True:
	__,img = webcam.read()
	# img = cv2.imread('2.jpg')
	img = imutils.resize(img, width=700)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 6)
	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
		face = gray_img[x:x+w, y:y+h]
		resize_img = cv2.resize(face , (width, height))

		predict = model.predict(resize_img)
		print(predict)
		if predict[1] < 800  :
			if os.path.isdir(datasets+'/'+names[predict[0]]):
				cv2.putText(img, '{}'.format(names[predict[0]]), (x, y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,0,255), 2)
				if predict[0] == 1:
					cv2.putText(img, '**Happy Birthday**,Sir', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,240), 3)
					
				print(names[predict[0]])

				cnt = 0
				# if predict[0] == [0]:
					# cv2.putText(img, '{}'.format(names[predict[0]]), (x+100, y+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 4)

		else:
			cnt += 1
			cv2.putText(img,'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 4)

			# if (cnt > 100):
			# 	print('Unknown')
			# 	cv2.imwrite('input.jpg',img)
			cnt = 0
	cv2.imshow('Face Recogniation', img)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break


webcam.release()
cv2.destroyAllWindows()

