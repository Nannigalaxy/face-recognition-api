import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import cv2
import math
from PIL import Image

def distance(a, b):
	x1 = a[0]; y1 = a[1]
	x2 = b[0]; y2 = b[1]
	
	return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

	

def detectFace(image_path, target_size=(224, 224), grayscale = False):
	
	
	
	path =  os.path.dirname(os.path.abspath(__file__))

	face_detector_path = path+"/data/haarcascade/haarcascade_frontalface_default.xml"
	eye_detector_path = path+"/data/haarcascade/haarcascade_eye.xml"
	
	
	#--------------------------------
	
	face_detector = cv2.CascadeClassifier(face_detector_path)
	eye_detector = cv2.CascadeClassifier(eye_detector_path)
	
	img = image_path
	
	img_raw = img.copy()
	
	#--------------------------------
	
	faces = face_detector.detectMultiScale(img, 1.3, 5)
	
	#print("found faces in ",image_path," is ",len(faces))
	
	if len(faces) > 0:
		x,y,w,h = faces[0]
		detected_face = img[int(y):int(y+h), int(x):int(x+w)]
		detected_face_gray = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
		
		#---------------------------
		#face alignment
		
		eyes = eye_detector.detectMultiScale(detected_face_gray)
		
		if len(eyes) >= 2:
			#find the largest 2 eye
			base_eyes = eyes[:, 2]
			
			items = []
			for i in range(0, len(base_eyes)):
				item = (base_eyes[i], i)
				items.append(item)
			
			df = pd.DataFrame(items, columns = ["length", "idx"]).sort_values(by=['length'], ascending=False)
			
			eyes = eyes[df.idx.values[0:2]]
			
			#-----------------------
			#decide left and right eye
			
			eye_1 = eyes[0]; eye_2 = eyes[1]
			
			if eye_1[0] < eye_2[0]:
				left_eye = eye_1
				right_eye = eye_2
			else:
				left_eye = eye_2
				right_eye = eye_1
			
			#-----------------------
			#find center of eyes
			
			left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
			left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
			
			right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
			right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
			
			#-----------------------
			#find rotation direction
				
			if left_eye_y > right_eye_y:
				point_3rd = (right_eye_x, left_eye_y)
				direction = -1 #rotate same direction to clock
			else:
				point_3rd = (left_eye_x, right_eye_y)
				direction = 1 #rotate inverse direction of clock
			
			#-----------------------
			#find length of triangle edges
			
			a = distance(left_eye_center, point_3rd)
			b = distance(right_eye_center, point_3rd)
			c = distance(right_eye_center, left_eye_center)
			
			#-----------------------
			#apply cosine rule
			
			cos_a = (b*b + c*c - a*a)/(2*b*c)
			angle = np.arccos(cos_a) #angle in radian
			angle = (angle * 180) / math.pi #radian to degree
			
			#-----------------------
			#rotate base image
			
			if direction == -1:
				angle = 90 - angle
			
			img = Image.fromarray(img_raw)
			img = np.array(img.rotate(direction * angle))
			
			#you recover the base image and face detection disappeared. apply again.
			faces = face_detector.detectMultiScale(img, 1.3, 5)
			if len(faces) > 0:
				x,y,w,h = faces[0]
				detected_face = img[int(y):int(y+h), int(x):int(x+w)]
			
			#-----------------------
		
		#face alignment block end
		#---------------------------
		
		#face alignment block needs colorful images. that's why, converting to gray scale logic moved to here.
		if grayscale == True:
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
		
		detected_face = cv2.resize(detected_face, target_size)
		
		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		
		#normalize input in [0, 1]
		img_pixels /= 255
		
		return img_pixels
		
	else:
		raise ValueError("Face could not be detected. Please confirm that the picture is a face photo.")