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

image_paths = defaultdict(list)

for face_class in face_classes:
	image_dir = os.path.join(faces_path, face_class).replace('\\', '/')
	for filepath in os.listdir(image_dir):
		if filepath.endswith('.jpg'):
			image_paths[face_class].append(os.path.join(image_dir, filepath).replace('\\', '/'))



face_class_ids = {face_class: index for index, face_class in enumerate(face_classes)}

face_paths_and_classes = []
for face_class, paths in image_paths.items():
	for path in paths:
		face_paths_and_classes.append((path, face_class_ids[face_class]))

test_ratio = .2
train_size = int(len(face_paths_and_classes) * (1 - test_ratio))

np.random.shuffle(face_paths_and_classes)

train_set = face_paths_and_classes[:train_size]
test_set = face_paths_and_classes[train_size:]


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
conv2_dropout_rate = .25

pool3_fmaps = conv2_fmaps

n_fc1 = 128
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
	pool3_flat = tf.reshape(pool3, shape = [-1, pool3_fmaps * 150 * 150])
	pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training = training)

with tf.name_scope('fc1'):
	fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation = tf.nn.relu, name = 'fc1')
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





#批处理
from random import sample

def random_batch(paths, batch_size):
	batch_paths = sample(paths, batch_size)
	images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths]
	prepared_images = [prepare_image(image) for image in images]
	X_batch = 2 * np.stack(prepared_images) - 1
	y_batch = np.array([labels for path, labels in batch_paths], dtype = np.int32)
	return X_batch, y_batch


n_epochs = 20
batch_size = 10
n_iterations_per_epoch = len(train_set) // batch_size

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		print('Epoch', epoch, end = '')
		for iteration in range(n_iterations_per_epoch):
			print('.', end = '')
			X_batch, y_batch = random_batch(train_set, batch_size)
			sess.run(training_op, feed_dict = {X:X_batch, y:y_batch, training : True})

		acc_batch = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})

		print('   Last batch accuracy:', acc_batch)

		save_path = saver.save(sess, 'models/my_face_model.ckpt')