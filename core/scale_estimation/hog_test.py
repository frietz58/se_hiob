import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
from skimage import io

# img = cv2.imread('biker0001.jpg')
#
# resized_img = cv2.resize(img, (, 16))
#
# winSize = (24, 16)
# blockSize = (4, 4)
# blockStride = (2, 2)
# cellSize = (4, 4)
# nbins = 9
# derivAperture = 1
# winSigma = -1.
# histogramNormType = 0
# L2HysThreshold = 0.2
# gammaCorrection = 1
# nlevels = 64
# signedGradients = True
#
# hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
#                         L2HysThreshold, gammaCorrection, nlevels, signedGradients)
#
# descriptor = hog.compute(resized_img)
#
# print("here")




image = io.imread("/home/finn/PycharmProjects/code-git/HIOB/images/Singer1/1-32_patch.png")

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

plt.axis('off')
plt.imshow(image, cmap=plt.cm.gray)
plt.savefig("hog_input.png")

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

plt.axis('off')
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.savefig("hog_output.png")