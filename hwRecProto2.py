########################################################################################
# Author: Thomas Flucke
# Date: 2017-05-13

# Abreviations:
# vect = Vector
# ANN  = Artifical Neural Network
# corr  = Correct version

########################################################################################
# Declare the data format

import random

class Datum:
    def __init__(self, label, img):
        self.label = [0] * (10 + 26 + 26)
        self.label[label - 1] = 1
        self.img = img

class IAM:
    def __init__(self):
        print "Building dataset..."
        self.train = []
        self.test = []
        for x in range(1, (10 + 26 + 26) + 1):
            print "Preparing sample %d..." % x
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

########################################################################################
# Load IAM dataset

import cPickle as pickle

with open('iamDataset.obj', 'rb') as input:
    iam = pickle.load(input)

########################################################################################
# Import tensorflow library and define AAN

import tensorflow as tf
LEARNING_CONST = 0.5

# Create a tensorflow placeholder for an array [nx784] float32's (a.k.a. n MNIST vectors)
inVect = tf.placeholder(tf.float32, [None, 972])
# Initalize the ANN with zero's
# Define tensorflow variable for the weight matrix [784x10] so we can matrix multiply
weights = tf.Variable(tf.zeros([972, (10 + 26 + 26)]))
# Define tensorflow variable for the bias vector
biases = tf.Variable(tf.zeros([(10 + 26 + 26)]))
# Define formula for calculating output [nx10]
outVect = tf.nn.softmax(tf.matmul(inVect, weights) + biases)
# Create a tensorflow placeholder for the correct answer vector
outVectCorr = tf.placeholder(tf.float32, [None, (10 + 26 + 26)])

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
    batchIns, batchOuts = iam.nextBatch(100)
    # Run the training step in the interactive session with the given inputs/outputs
    sess.run(trainStep, feed_dict={inVect: batchIns, outVectCorr: batchOuts})

# Check accuracy
print(sess.run(accuracy, feed_dict={inVect: (o.img for o in iam.test), outVectCorr: (o.label for o in iam.test)}))
