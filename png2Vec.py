########################################################################################
# Author: Thomas Flucke
# Date: 2017-05-13

# Abreviations:
# vect = Vector
# ANN  = Artifical Neural Network
# corr  = Correct version

########################################################################################
# Set up the png library

import scipy.ndimage

# Import random image of letter A
img = scipy.ndimage.imread("English/Hnd/Img/Sample011/img011-016.png", True)

rows, cols = img.shape

if rows != 900 or cols != 1200:
    print "Bad Image Dimensions!"
    exit()

import itertools

smlImg = scipy.misc.imresize(img, 0.05)

vect = list(itertools.chain.from_iterable(smlImg))

print vect
