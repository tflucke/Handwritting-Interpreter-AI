########################################################################################
# Author: Thomas Flucke
# Date: 2017-05-13

# Abreviations:
# vect = Vector
# ANN  = Artifical Neural Network
# corr  = Correct version

########################################################################################
# Set up the png library

import os
import scipy.ndimage
import itertools

DATA_FOLDER = "pngChars/Hnd/Img/Sample%03d"
IMG_TEMPLATE = "pngChars/Hnd/Img/Sample%03d/%s"

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

iam = IAM()

print "Test Points: %d" % len(iam.test)
print "Train Points: %d" % len(iam.train)

print "Saving data..."

import cPickle as pickle

with open("iamDataset.obj", 'wb') as output:
    pickle.dump(iam, output, -1)
