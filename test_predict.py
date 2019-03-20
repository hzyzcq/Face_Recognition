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
	return image.astype(np.float32) / 255



faces_path = os.path.join('datasets', 'faces').replace('\\', '/')
face_classes = sorted([dirname for dirname in os.listdir(faces_path)])


height = 300
width = 300
channels = 3


conv1_fmaps = 32
conv1_ksize  = 3
conv1_stride = 1
conv1_pad = 'SAME'

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = 'SAME'

pool3_fmaps = conv2_fmaps

conv4_fmaps = 16
conv4_ksize = 3
conv4_stride = 1
conv4_pad = 'SAME'
conv4_dropout_rate = .25

pool5_fmaps = conv4_fmaps

n_fc1 = 256
fc1_dropout_rate = .5

n_outputs = len(face_classes)

tf.reset_default_graph()

with tf.name_scope('inputs'):
	X = tf.placeholder(tf.float32, shape = [None, height, width, channels], name = 'X')
	y = tf.placeholder(tf.int32, shape = [None], name= 'y')
	training = tf.placeholder_with_default(False, shape = [], name = 'training')

conv1 = tf.layers.conv2d(X, filters = conv1_fmaps, kernel_size = conv1_ksize,
	strides = conv1_stride, padding = conv1_pad,
	activation = tf.nn.relu, name = 'conv1')

conv2 = tf.layers.conv2d(conv1, filters = conv2_fmaps, kernel_size = conv2_ksize, 
	strides = conv2_stride, padding = conv2_pad,
	activation = tf.nn.relu, name = 'conv2')

with tf.name_scope('pool3'):
	pool3 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
	
conv4 = tf.layers.conv2d(pool3, filters = conv4_fmaps, kernel_size = conv4_ksize, 
	strides = conv4_stride, padding = conv4_pad,
	activation = tf.nn.relu, name = 'conv4')


with tf.name_scope('pool5'):
	pool5 = tf.nn.max_pool(conv4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
	pool5_flat = tf.reshape(pool5, shape = [-1, pool5_fmaps * 75* 75])
	pool5_flat_drop = tf.layers.dropout(pool5_flat, conv4_dropout_rate, training = training)





with tf.name_scope('fc1'):
	fc1 = tf.layers.dense(pool5_flat_drop, n_fc1, activation = tf.nn.relu, name = 'fc1')
	fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training = training)



with tf.name_scope('output'):

	logits = tf.layers.dense(fc1, n_outputs, name = 'output')
	y_proba = tf.nn.softmax(logits, name = 'y_proba')

with tf.name_scope('train'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)
	loss = tf.reduce_mean(xentropy)
	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_save'):
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()



import cv2
import sys
import gc


with tf.Session() as sess:
	saver.restore(sess, 'models/my_face_model.ckpt')

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
				print(faceID)
				faceID = np.argmax(faceID.reshape(-1))

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





