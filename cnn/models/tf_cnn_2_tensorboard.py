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
	
def max_pool2d(x):
	m = tf.nn.max_pool(x, ksize= [1,2,2,1], strides=[1,2,2,1], padding="SAME")
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
	
#Tensorboard utils:
def var_summaries(var,name):
	name= 'Summaries_' + str(name)
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
	
	
	
#create model:
sess = tf.Session()

##Placeholders
xs = tf.placeholder(tf.float32, [None, 784]) /255 #28x28
ys = tf.placeholder(tf.float32, [None, 10])

keep = tf.placeholder(tf.float32)

x_img = tf.reshape(xs, [-1, 28,28,1])
print("x_img shape- ", x_img.shape)

##conv1 
W_conv1 = weight([5,5,1,32])
b_conv1 = bias([32])
A1_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1 )
var_summaries(A1_conv1, name="A1_conv1")


##pool1
A1_pool1 = max_pool2d(A1_conv1)

##conv2
W_conv2 = weight([5,5,32,64])
b_conv2 = bias([64])
A2_conv2 = tf.nn.relu(conv2d(A1_pool1, W_conv2) + b_conv2 )

##pool2
A2_pool2 = max_pool2d(A2_conv2)
var_summaries(A2_conv2, "A2_conv2")

##flatten
## [m, 7,7,64] -> [m,7*7*64]
A2_flatten = flatten(A2_pool2)
#print("A2_flatten shape-", A2_flatten.shape)

##fc layer 1
#W_fc1 = weight(shape= [tf.cast(A2_flatten.get_shape()[1] , 'int32'), 1024])
W_fc1 = weight(shape=[get_shape(A2_flatten, index=1), 1024])
#W_fc1 = weight([7*7*64, 1024])
b_fc1 = bias([1024])
A3_fc1 = tf.nn.relu(fc_layer(A2_flatten, W_fc1) + b_fc1)

##dropout
A3_dropout = tf.nn.dropout(A3_fc1, keep)

##fc layer 2
W_fc2 = weight( shape= [1024, 10] )
b_fc2 = bias([10])
A4_fc2 = tf.nn.softmax(fc_layer(A3_dropout, W_fc2) + b_fc2 )

prediction = A4_fc2

#Error between prediction & label
loss = cross_entropy(ys, prediction)

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)


#Session
init = tf.global_variables_initializer()
sess.run(init)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/1/", sess.graph)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep: 0.5} )
	if i%5 ==0:
		print("Epoch ", i, ", Train Accuracy =", compute_accuracy(mnist.train.images[:1000], mnist.train.labels[:1000], 0.5) )
		print("Epoch ", i, ", Test Accuracy =", compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000], 1) )
		print("------------------------------")
		sum_run = sess.run(merged, feed_dict={xs:batch_xs,ys:batch_ys})
		writer.add_summary(sum_run, i)
		

sess.close()

