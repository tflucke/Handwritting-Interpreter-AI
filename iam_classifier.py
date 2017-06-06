# From the tutorial on 
# https://www.tensorflow.org/get_started/mnist/pros

import argparse
import sys

########################################################################################
# Declare the data format

import random
import _pickle as pickle

N_INPUT = 972
N_OUTPUT = 10 + 26 + 26

class Datum:
    def __init__(self, label, img):
        self.label = [0] * N_OUTPUT
        self.label[label - 1] = 1
        self.img = img

class IAM:
    def __init__(self):
        print("Building dataset...")
        self.train = []
        self.test = []
        for x in range(1, N_OUTPUT + 1):
            print("Preparing sample %d..." % x)
            for f in os.listdir(DATA_FOLDER % x):
                img = scipy.ndimage.imread(IMG_TEMPLATE % (x, f), True)
                img = scipy.misc.imresize(img, 0.03)
                img = list(itertools.chain.from_iterable(img))
                if len(self.test) < (5 * x):
                    self.test.append(Datum(x, img))
                else:
                    self.train.append(Datum(x, img))
    def nextBatch(self, size):
        used = []
        res = []
        labels = []
        while len(used) < size:
            i = random.randint(0, len(self.train) - 1)
            if i in used:
                continue
            else:
                used.append(i)
                res.append(self.train[i].img)
                labels.append(self.train[i].label)
        return res, labels
    def testSet(self):
        res = []
        labels = []
        for i in range(0, len(self.test)):
            res.append(self.test[i].img)
            labels.append(self.test[i].label)
        return res, labels

print(sys.argv)

LEARNING_CONST = 0.05
TRAIN_CYCLES = 1000
BATCH_SIZE = 100

# Turn off GPU Warnings/All other warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf

FLAGS = None


def deepnn(x):
  # Reshape to use within a convolutional neural net.
  x_image = tf.reshape(x, [-1, 18, 54, 1]) # tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 
  W_fc1 = weight_variable([7 * 10 * 64, 1024]) # weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*10*64]) # tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 62 classes, one for each symbol
  W_fc2 = weight_variable([1024, 62]) # weight_variable([1024, 10])
  b_fc2 = bias_variable([62]) # bias_variable([10])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with open('iamDataset.obj', 'rb') as input:
    iam = pickle.load(input)

  # Create the model
  x = tf.placeholder(tf.float32, [None, N_INPUT]) # tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 62]) # tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(101): # 20000
      batch = iam.nextBatch(BATCH_SIZE) # mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
