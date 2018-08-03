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
	


###-------------------@@@@@@@@@@@@@@@@@@@@@@---------------------------###
###-------------------@@@@@@@@@@@@@@@@@@@@@@---------------------------###


#create model:
#sess = tf.Session()

##Placeholders
xs = tf.placeholder(tf.float32, [None, 784]) /255 #28x28
ys = tf.placeholder(tf.float32, [None, 10])

keep = tf.placeholder(tf.float32)

x_img = tf.reshape(xs, [-1, 28,28,1])
print("x_img shape- ", x_img.shape)

##conv1 
with tf.name_scope("Conv1"):		
	W_conv1 = weight([3,3,1,256])
	b_conv1 = bias([256])
	A1_conv = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1 )
	var_summaries(A1_conv, name="A1_conv")

##pool1
with tf.name_scope("Pool1"):
	A1_pool = max_pool2d(A1_conv, ksize=[1,2,2,1])

##conv2
with tf.name_scope("Conv2"):		
	W_conv2 = weight([3,3,256,128])
	b_conv2 = bias([128])
	A2_conv = tf.nn.relu(conv2d(A1_pool, W_conv2) + b_conv2 )
	var_summaries(A2_conv, "A2_conv")

##pool2
with tf.name_scope("Pool2"):		
	A2_pool = max_pool2d(A2_conv,ksize=[1,2,2,1])
	
##conv3
with tf.name_scope("Conv3"):		
	W_conv3 = weight([3,3,128,32])
	b_conv3 = bias([32])
	A3_conv = tf.nn.relu(conv2d(A2_pool, W_conv3) + b_conv3 )
	var_summaries(A3_conv, "A3_conv")

##pool3
with tf.name_scope("Pool3"):		
	A3_pool = max_pool2d(A3_conv,ksize=[1,2,2,1])	

##conv4
with tf.name_scope("Conv4"):		
	W_conv4 = weight([3,3,32,16])
	b_conv4 = bias([16])
	A4_conv = tf.nn.relu(conv2d(A3_pool, W_conv4) + b_conv4 )
	var_summaries(A4_conv, "A4_conv")

##pool4
with tf.name_scope("Pool4"):		
	A4_pool = max_pool2d(A4_conv,ksize=[1,2,2,1])


##flatten
## [m, 7,7,64] -> [m,7*7*64]cost_sum
with tf.name_scope("Flatten"):
	A4_flatten = flatten(A4_pool)
	
##fc layer 1
with tf.name_scope("FC1"):		
	#W_fc1 = weight(shape= [tf.cast(A2_flatten.get_shape()[1] , 'int32'), 1024])
	W_fc1 = weight(shape=[get_shape(A4_flatten, index=1), 1024])
	b_fc1 = bias([1024])
	A5_fc1 = tf.nn.relu(fc_layer(A4_flatten, W_fc1) + b_fc1)

##dropout
A5_dropout = tf.nn.dropout(A5_fc1, keep)

##fc layer 2
with tf.name_scope("FC2"):		
	W_fc2 = weight( shape= [1024, 10] )
	b_fc2 = bias([10])
	A6_fc2 = tf.nn.softmax(fc_layer(A5_dropout, W_fc2) + b_fc2 )

prediction = A6_fc2



#Error between prediction & label
#loss = cross_entropy(ys, prediction)
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

#Session
sess = tf.Session()
writer = tf.summary.FileWriter("./logs/1/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

counter=0
for i in range(100000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	batch_xs, batch_ys = mnist.test.next_batch(100)
	
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep: 0.5} )
	if i%10 ==0:
		#Losses
		with tf.name_scope("train_loss"):
			train_loss = compute_loss(mnist.train.images[:1000], mnist.train.labels[:1000], 0.5) 
			print("Epoch ", i, ", Train Loss =", train_loss)
			#var_summaries(train_loss, "trainingloss")
			trainlosssum = tf.summary.scalar("trainloss", train_loss)
		with tf.name_scope("test_loss"):
			test_loss = compute_loss(mnist.test.images[:1000], mnist.test.labels[:1000], 1) 
			print("Epoch ", i, ", Test Loss =", test_loss)
			#testlosssum = tf.summary.scalar("testloss", test_loss)
			
		#Accuracy
		with tf.name_scope("train_accu"):
			train_acc = compute_accuracy(mnist.train.images[:1000], mnist.train.labels[:1000], 0.5) 
		print("Epoch ", i, ", Train Accuracy =", train_acc)
		with tf.name_scope("test_accu"):
			test_accu = compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000], 1) 
			print("Epoch ", i, ", Test Accuracy =", test_accu)
		print("------------------------------")
		
		
		merged = tf.summary.merge_all()
		sum_run = sess.run(merged, feed_dict={xs:batch_xs,ys:batch_ys})
		
		counter = counter + 1
		writer.add_summary(sum_run, counter)
		

sess.close()

