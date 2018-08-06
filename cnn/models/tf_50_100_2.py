import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


#data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#funcs
def weight(shape):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1) )
	return w

def bias(shape):
	b = tf.Variable(tf.constant(0.1, shape=shape) )
	return b
	
def conv2d(x, W):
	c = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")
	return c
	
def max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1]):
	m = tf.nn.max_pool(x, ksize= ksize, strides=strides, padding="SAME")
	return m

def fc_layer(x, W):
	f = tf.matmul(x, W)
	return f

def flatten(x):
	fshape = x.get_shape()[1] * x.get_shape()[2] * x.get_shape()[3]
	shape = [-1, fshape]
	fl = tf.reshape(x, shape)
	return fl

def get_fullshape(x):
	return tf.cast(x.get_shape(), 'int32' )

def get_shape(x, index):
	return tf.cast(x.get_shape()[index], 'int32')

def cross_entropy(ys, prediction):
	return tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction) , reduction_indices=[1]) )

def compute_accuracy(xs_val, ys_val, keep_val):
	global prediction
	y_prediction = sess.run(prediction, feed_dict={xs:xs_val, keep:keep_val})
	correct = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(ys_val, 1) )
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	acc = sess.run(accuracy, feed_dict={xs:xs_val, ys:ys_val, keep:keep_val})
	return acc

def compute_loss(xs_val, ys_val, keep_val):
	#global loss
	global prediction
	y_prediction = sess.run(prediction, feed_dict={xs:xs_val, keep:keep_val})
	l = sess.run(loss, feed_dict={ys:ys_val, prediction: y_prediction })
	return l


#Tensorboard utils:
def var_summaries(var):
	name= 'Summaries'
	with tf.name_scope(name):
		with tf.name_scope("Mean"):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
		with tf.name_scope("Std_dev"):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)) )
			tf.summary.scalar('stddev', stddev)
		with tf.name_scope("Others"):
			tf.summary.scalar("max",tf.reduce_max(var) )
			tf.summary.scalar("min", tf.reduce_min(var) )
			tf.summary.histogram("histogram", var)

def visualize_weights(W_conv1, num_kernels=32):					
	with tf.name_scope("Visualize_weights_conv"):
		#concatenate the filters into one image of row size 8 images
		W_a = W_conv1					#i.e, [5,5,1,32]
		W_b = tf.split(W_a, num_kernels, 3)		#i.e, [32,5,5,1,1]
		rows = []
		for i in range(int(num_kernels/8)):
			x1 = i*8
			x2 = (i+1)*8
			row = tf.concat(W_b[x1:x2], 0)
			rows.append(row)
		W_c = tf.concat(rows, 1)
		c_shape = W_c.get_shape().as_list()
		W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1] )
		tf.summary.image("Visualize_kernels_conv", W_d, 1024)

			
#########################################################################

##Model

print("\nBuilding the model.")

#set placeholder
#-----------------------------------------------------------------------
with tf.name_scope("Input"):
	xs = tf.placeholder(tf.float32, [None, 784], name="x_input")  #28x28
	ys = tf.placeholder(tf.float32, [None, 10], name="y_label")
#-----------------------------------------------------------------------
with tf.name_scope("Input_reshape"):
	x_reshaped = tf.reshape(xs, [-1, 28,28,1])
	tf.summary.image("input", x_reshaped, 10)
#-----------------------------------------------------------------------	
with tf.name_scope("Dropout"):
	keep_prob = tf.placeholder(tf.float32)
	tf.summary.scalar("keep_probability", keep_prob)
#-----------------------------------------------------------------------
#	Conv 1 layer + pool layer
#-----------------------------------------------------------------------
with tf.name_scope("Conv1"):
	with tf.name_scope("weights"):
		W_conv1 = weight([5,5,1,32])
	with tf.name_scope("biases"):
		b_conv1 = bias([32])
	with tf.name_scope("activation"):
		A1_conv = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1 )

	with tf.name_scope("Pool1"):
		A1_pool = max_pool2d(A1_conv, ksize=[1,2,2,1])

	#save output of Conv layer to TB- first 16 filters
	with tf.name_scope("Image_conv1"):
		image = A1_conv[0:1, :, :, 0:16]
		image = tf.transpose(image, perm=[3,1,2,0])
		tf.summary.image("image_conv1", image)
	#save a visual representation of weights to TB

with tf.name_scope("Visualize_weights_conv1"):
	#concatenate the filters into one image of row size 8 images
	W_a = W_conv1					#i.e, [5,5,1,32]
	W_b = tf.split(W_a, 32, 3)		#i.e, [32,5,5,1,1]
	rows = []
	for i in range(int(32/8)):
		x1 = i*8
		x2 = (i+1)*8
		row = tf.concat(W_b[x1:x2], 0)
		rows.append(row)
	W_c = tf.concat(rows, 1)
	c_shape = W_c.get_shape().as_list()
	W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1] )
	tf.summary.image("Visualize_kernels_conv1", W_d, 1024)

#-----------------------------------------------------------------------
#	Conv 2 layer + pool layer
#-----------------------------------------------------------------------	
with tf.name_scope("Conv2"):
	with tf.name_scope("weights"):
		W_conv2 = weight([5,5,32,64])
	with tf.name_scope("biases"):
		b_conv2 = bias([64])
	with tf.name_scope("activation"):
		A2_conv = tf.nn.relu(conv2d(A1_pool, W_conv2) + b_conv2)
	
	with tf.name_scope("Pool2"):
		A2_pool = max_pool2d(A2_conv, ksize=[1,2,2,1])
		
		#save output of Conv layer to TB- first 16 filters
		with tf.name_scope("Image_conv2"):
			image = A2_conv[0:1, :, :, 0:16]
			image = tf.transpose(image, perm=[3,1,2,0])
			tf.summary.image("image_conv2", image)

with tf.name_scope("Visualize_weights_conv2"):
	W_a = W_conv2
	W_b = tf.split(W_a, 64, 3)
	rows = []
	for i in range(int(64/8)):
		x1 = i*8
		x2 = (i+1)*8
		row = tf.concat(W_b[x1:x2], 0)
		rows.append(row)
	W_c = tf.concat(rows, 1)
	c_shape = W_c.get_shape().as_list()
	W_d = tf.reshape(W_c, [c_shape[2], c_shape[1], 1])
	tf.summary.image("Visualize_kernels_conv2", W_d, 1024)
		

		

	
	







