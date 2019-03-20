import pandas as pd
import os
from collections import defaultdict
import tensorflow as tf
import numpy as np

import matplotlib.image as mpimg

from skimage.transform import resize, rotate
from random import randint
import numpy as np

def prepare_image(image, target_width = 300, target_height = 300):

	angle = randint(-30, 30) 
	ro_image = rotate(image, angle, mode = 'edge')

	if np.random.rand() < .5:
		ro_image = np.fliplr(ro_image)

	image = resize(ro_image, (target_width, target_height))
	return image.astype(np.float32)



faces_path = os.path.join('datasets', 'faces').replace('\\', '/')
face_classes = sorted([dirname for dirname in os.listdir(faces_path)])

tf.reset_default_graph()


import cv2
import sys
import gc
saver = tf.train.import_meta_graph("models/my_face_model.ckpt.meta")

with tf.Session() as sess:
	
	saver.restore(sess, 'models/my_face_model.ckpt')
	graph = tf.get_default_graph()
	X = graph.get_tensor_by_name('inputs/X:0')
	y_proba = graph.get_tensor_by_name('output/y_proba:0')

	classifier = cv2.CascadeClassifier('models/face/haarcascade_frontalface_alt2.xml')


	color = (0, 255, 0)
	cap = cv2.VideoCapture(0)

	while cap.isOpened():
		_, frame = cap.read()

		
		grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faceRects = classifier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))

		if len(faceRects) > 0:
			for faceRect in faceRects:  #单独框出每一张人脸
				x, y, w, h = faceRect        
				image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
	
				image = prepare_image(image)
				faceID = y_proba.eval(feed_dict = {X : [image]})

				faceID = np.argmax(faceID.reshape(-1)[0])

				cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
				cv2.putText(frame,face_classes[faceID], 
									(x + 30, y + 30),                      #坐标
									cv2.FONT_HERSHEY_SIMPLEX,              #字体
									1,                                     #字号
									(255,0,255),                           #颜色
									2)                                     #字的线宽

		cv2.imshow('Recognise myself', frame)

		k = cv2.waitKey(10)

		if k & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()





