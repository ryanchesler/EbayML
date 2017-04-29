import tensorflow as tf
import numpy as np
import pickle
import random
import time
from random import shuffle
datafile = "./Data/processeddata2.pk1"
datapoints = open(datafile, 'rb')
data = pickle.load(datapoints)
datapoints.close
featuresize = 313
##featuresize = 784
#make train set

shuffle(data)
totaldata = 0
data = np.array(data)
batch_xs = []
batch_ys = []
for x in data[0:-500]:
  batch_xs.append(x[0])
  batch_ys.append(x[1])
  totaldata +=1
test_xs = []
test_ys = []
for x in data[-499:-1]:
  test_xs.append(x[0])
  test_ys.append(x[1])
labelcount = 2
a_0 = tf.placeholder(tf.float32, shape=[None, featuresize])
y = tf.placeholder(tf.float32, shape=[None, labelcount])
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
middle = 2000
middle2 = 2000
middle3 = 1500
w_1 = weight_variable([featuresize, middle])
b_1 = bias_variable([1, middle])
w_2 = weight_variable([middle, middle2])
b_2 = bias_variable([1, middle2])
w_3 = weight_variable([middle2, middle3])
b_3 = bias_variable([1, middle3])
w_4 = weight_variable([middle3, labelcount])
b_4 = bias_variable([1, labelcount])
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
keep_prob4 = tf.placeholder(tf.float32)
drop1 = tf.nn.dropout(w_1, keep_prob1)
drop2 = tf.nn.dropout(w_2, keep_prob2)
drop3 = tf.nn.dropout(w_3, keep_prob3)
drop4 = tf.nn.dropout(w_4, keep_prob4)

a_1 = tf.nn.relu(tf.add(tf.matmul(a_0, drop1), b_1))
a_2 = tf.nn.relu(tf.add(tf.matmul(a_1, drop2), b_2))
a_3 = tf.nn.relu(tf.add(tf.matmul(a_2, drop3), b_3))
a_4 = tf.add(tf.matmul(a_3, drop4), b_4)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = a_4, labels = y))
step = tf.train.AdamOptimizer(learning_rate = .000001).minimize(cross_entropy)
acct_mat = tf.equal(tf.argmax(a_4, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.Session()
saver = tf.train.Saver()

batchsize = 50
with tf.Session() as sess:
  #sess.run(tf.global_variables_initializer())
  saver.restore(sess, "./Data/Modelv1")
  for i in range(5000000):
      randomvar = random.randrange(0, len(batch_xs) -batchsize)
      sess.run(step, feed_dict = {a_0: batch_xs[randomvar:randomvar + batchsize],
                                  y : batch_ys[randomvar:randomvar + batchsize], keep_prob1: .92, keep_prob2 : .45 , keep_prob3: .45, keep_prob4:.92})
      if i % 1000 == 0:
          res = sess.run(acct_res, feed_dict =
                         {a_0: test_xs, y: test_ys, keep_prob1: 1.0, keep_prob2 : 1.0 ,keep_prob3: 1.0, keep_prob4:1.00})
          training = sess.run(acct_res, feed_dict =
                         {a_0: batch_xs, y: batch_ys, keep_prob1: 1.0, keep_prob2 : 1.0 ,keep_prob3: 1.0, keep_prob4:1.00})
          print ("Test Error: " , res/500)
          print("Training Error: " , training/totaldata)

      if i % 5000 == 0:
        saver.save(sess, "./Data/Modelv1")
