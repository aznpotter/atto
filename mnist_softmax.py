# copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

#from tensorflow.examples.tutorials.mnist import input_data
import input_data
import tensorflow as tf

FLAGS = None

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def main(_):
	# Import data
	#mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  	mnist = input_data.read_data_sets(one_hot=True)

	# Create the model
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.matmul(x, W) + b
	  
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1,28,28,1])
	
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)  

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	# image size is 7*7*1*64

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
  
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  

# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 10])

	# The raw formulation of cross-entropy,
	#
	# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
	#                                 reduction_indices=[1]))
	#
	# can be numerically unstable.
	#
	# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
	# outputs of 'y', and then average across the batch.


	sess = tf.InteractiveSession()


#	tf.global_variables_initializer().run()
#	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)	
#  Train
#	for _ in range(1000):
#		batch_xs, batch_ys = mnist.train.next_batch(100)
#		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#  Test trained model
#	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#	print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	
 	W1_a = W_conv1                       # [5, 5, 1, 32]
	W1pad= tf.zeros([5, 5, 1, 1])        # [5, 5, 1, 4]  - four zero kernels for padding
	# We have a 6 by 6 grid of kernepl visualizations. yet we only have 32 filters
	# Therefore, we concatenate 4 empty filters
	W1_b = tf.concat(3, [W1_a, W1pad, W1pad, W1pad, W1pad])   # [5, 5, 1, 36]  
	W1_c = tf.split(3, 36, W1_b)         # 36 x [5, 5, 1, 1]
	W1_row0 = tf.concat(0, W1_c[0:6])    # [30, 5, 1, 1]
	W1_row1 = tf.concat(0, W1_c[6:12])   # [30, 5, 1, 1]
	W1_row2 = tf.concat(0, W1_c[12:18])  # [30, 5, 1, 1]
	W1_row3 = tf.concat(0, W1_c[18:24])  # [30, 5, 1, 1]
	W1_row4 = tf.concat(0, W1_c[24:30])  # [30, 5, 1, 1]
	W1_row5 = tf.concat(0, W1_c[30:36])  # [30, 5, 1, 1]
	W1_d = tf.concat(1, [W1_row0, W1_row1, W1_row2, W1_row3, W1_row4, W1_row5]) # [30, 30, 1, 1]
	W1_e = tf.reshape(W1_d, [1, 30, 30, 1])
  	Wtag = tf.placeholder(tf.string, None)
  	imagemash_summary=tf.summary.image("Visualize_kernels", W1_e)

	W_conv1_reshape=tf.reshape(W_conv1, [32, 5,5,1])
	#W_conv2_reshape=tf.reshape(W_conv2, [64, 5,5,32])
	image_summary_Wconv1 = tf.summary.image(W_conv1.name, W_conv1_reshape, max_outputs=3)
	#image_summary_Wconv2 = tf.summary.image(W_conv2.name, W_conv2_reshape, max_outputs=3)
	
	loss_summary = tf.summary.scalar(cross_entropy.name, cross_entropy)

	merged = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())
	summary_writer = tf.summary.FileWriter("folder_summary", sess.graph)
		
	for i in range(300):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		#train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		#_,image_summ = sess.run([train_step,image_summary], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		#_,loss_summ = sess.run([train_step,loss_summary], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		#_,image_summ1,image_summ2,loss_summ = sess.run([train_step,image_summary_Wconv1,image_summary_Wconv2,loss_summary], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		result=sess.run([train_step,merged], feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
		if i%100 == 0:
			#summary_writer.add_summary(image_summ1,i)
			#summary_writer.add_summary(image_summ2,i)	
			#summary_writer.add_summary(loss_summ,i)
			summary_writer.add_summary(result[1],i)
			summary_writer.flush()
	print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

	sess.close()
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
		help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
