import cv2
import numpy as np

img = cv2.imread('biker0001.jpg')

resized_img = cv2.resize(img, (20, 20))



winSize = (20, 20)
blockSize = (4, 4)
blockStride = (2, 2)
cellSize = (4, 4)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels, signedGradients)

descriptor = hog.compute(resized_img)

print("here")

