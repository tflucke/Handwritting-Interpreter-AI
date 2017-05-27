########################################################################################
# Author: Thomas Flucke
# Date: 2017-05-13

# Abreviations:
# vect = Vector
# ANN  = Artifical Neural Network
# corr  = Correct version

########################################################################################
# Download the MNIST dataset

print "Importing MNIST parsing libraries..."
from tensorflow.examples.tutorials.mnist import input_data
print "done"
print "Downloading MNIST dataset..."
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print "done"

# mnist.train = 55 000 train points
# mnist.test  = 1 000  test points
# mnist.validation = 5 000 ? points

# Inputs =  array[784] : Each index = 1px
# Outputs = array[10]  : Each index = corresponding digit

########################################################################################
# Import tensorflow library and define AAN

import tensorflow as tf
LEARNING_CONST = 0.5

# Create a tensorflow placeholder for an array [nx784] float32's (a.k.a. n MNIST vectors)
inVect = tf.placeholder(tf.float32, [None, 784])
# Initalize the ANN with zero's
# Define tensorflow variable for the weight matrix [784x10] so we can matrix multiply
weights = tf.Variable(tf.zeros([784, 10]))
# Define tensorflow variable for the bias vector
biases = tf.Variable(tf.zeros([10]))
# Define formula for calculating output [nx10]
outVect = tf.nn.softmax(tf.matmul(inVect, weights) + biases)
# Create a tensorflow placeholder for the correct answer vector
outVectCorr = tf.placeholder(tf.float32, [None, 10])

# Calculate how incorrect the solutions arrive were
crossEntropy = tf.reduce_mean(
    -tf.reduce_sum(
        outVectCorr * tf.log(outVect),
        # Tells reduce_sum to use the 10-length array, and not the n-length
        reduction_indices=[1]
    )
)

trainStep = tf.train.GradientDescentOptimizer(LEARNING_CONST).minimize(crossEntropy)

########################################################################################
# Define accuracy checking conditions

# Define formula for determining correctness
# Highest value in outVect 1st index == highest value in correct outVect 1st index
predictionCorr = tf.equal(tf.argmax(outVect, 1), tf.argmax(outVectCorr, 1))

# Calculate how accurate the system was
accuracy = tf.reduce_mean(tf.cast(predictionCorr, tf.float32))

########################################################################################
# Run the system

# Create interactive session
sess = tf.InteractiveSession()

# Initialize variables
tf.global_variables_initializer().run()

for _ in range(1000) :
    # Get 100 random digits from training set
    batchIns, batchOuts = mnist.train.next_batch(100)
    # Run the training step in the interactive session with the given inputs/outputs
    sess.run(trainStep, feed_dict={inVect: batchIns, outVectCorr: batchOuts})

# Check accuracy
print(sess.run(accuracy, feed_dict={inVect: mnist.test.images, outVectCorr: mnist.test.labels}))
