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

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W, name):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME', name=name)

def main(_):
	# Import data
	#mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  	mnist = input_data.read_data_sets(one_hot=True)

	# Create the model
	x = tf.placeholder(tf.float32, [None, 784])
	x_image = tf.reshape(x, [-1,28,28,1])
	
	with tf.name_scope("filter1") as scope:
		W_conv1 = weight_variable([5, 5, 1, 32], "conv1")
		b_conv1 = bias_variable([32],"bias_conv1")
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, "relu1") + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1, "pool1")  
	with tf.name_scope("filter2") as scope:
		W_conv2 = weight_variable([5, 5, 32, 64], "conv2")
		b_conv2 = bias_variable([64],"bias_conv2")
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2,"relu2") + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2, "pool2")
	# image size is 7*7*1*64
	with tf.name_scope("fully_connected1") as scope:
		W_fc1 = weight_variable([7 * 7 * 64, 1024], "conn1")
		b_fc1 = bias_variable([1024],"bias_conn1")
  		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	with tf.name_scope("fully_connected2") as scope:	
		W_fc2 = weight_variable([1024, 10], "conn2")
		b_fc2 = bias_variable([10],"bias_conn2")
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

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	
	with tf.name_scope("kernelmash") as scope:
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

	#W_conv1_reshape=tf.reshape(W_conv1, [32, 5,5,1])
	#image_summary_Wconv1 = tf.summary.image(W_conv1.name, W_conv1_reshape, max_outputs=3)
	
	loss_summary = tf.summary.scalar(cross_entropy.name, cross_entropy)

	merged = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())
	summary_writer = tf.summary.FileWriter("folder_summary", sess.graph)
		
	for i in range(500):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		result=sess.run([train_step,merged], feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
		if i%100 == 0:
			summary_writer.add_summary(result[1],i)
			summary_writer.flush()
		if i%100==1:
			run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        		run_metadata = tf.RunMetadata()
        		summary, _ = sess.run([merged, train_step],
                		feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5},
                		options=run_options,
                		run_metadata=run_metadata) 
				#feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
        		summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
        		summary_writer.add_summary(summary, i)
        		print('Adding run metadata for', i)
	print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

	sess.close()
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
		help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
