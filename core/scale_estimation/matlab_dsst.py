import numpy as np
import cv2
import logging
from PIL import Image
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


class DsstEstimator:

    def __init__(self):

        self.padding = 1.0
        self.nScales = None
        self.scale_step = None
        self.scale_sigma_factor = None
        self.scale_model_max_area = None
        self.target_sz = None
        self.img_files = None
        self.lam = None
        self.learning_rate = None

        self.scale_window = None
        self.pos = None
        self.init_target_sz = None
        self.base_target_sz = None
        self.frame = None
        self.sf_den = None
        self.sf_num = None

    def setup(self, n_scales, scale_step, scale_sigma_factor, img_files, scale_model_max, learning_rate):

        self.nScales = n_scales
        self.scale_step = scale_step
        self.scale_sigma_factor = scale_sigma_factor
        self.img_files = img_files
        self.lam = 1e-2
        self.scale_model_max_area = scale_model_max
        self.learning_rate = learning_rate

    def execute_scale_estimation(self, frame):

        self.frame = frame
        im = self.img_files[frame.number]
        #TODO this should be here rigth?
        self.base_target_sz = [frame.predicted_position.width, frame.predicted_position.height]
        self.currentScaleFactor = 1

        if frame.number == 1:
            # target size at scale = 1
            self.base_target_sz = [frame.predicted_position.width, frame.predicted_position.height]
            self.init_target_sz = [frame.predicted_position.width, frame.predicted_position.height]

            sz = np.floor(np.multiply(self.base_target_sz, (1 + self.padding)))

            # desired scale filter output (gaussian shaped), bandwidth proportional to
            # number of scales
            scale_sigma = self.nScales / np.sqrt(self.nScales) * self.scale_sigma_factor
            ss = np.subtract(np.arange(1, self.nScales + 1), np.ceil(33 / 2))
            ys = np.exp(-0.5 * (np.power(ss, 2)) / scale_sigma ** 2)
            self.ysf = np.fft.fft(ys)

            # store pre-computed scale filter cosine window
            # if mod(nScales,2) == 0
            if np.mod(self.nScales, 2) == 0:
                self.scale_window = np.hanning(self.nScales + 1)
                self.scale_window = self.scale_window[2: len(self.scale_window)]
            else:
                self.scale_window = np.hanning(self.nScales)  # correct val

            # scale factors
            ss = np.arange(1, self.nScales + 1)
            self.scaleFactors = np.power(
                self.scale_step,
                np.subtract(np.rint(self.nScales / 2), ss))  # almost correct val? round error?

            # compute the resize dimensions used for feature extraction in the scale
            # estimation
            scale_model_factor = 1
            if np.prod(self.init_target_sz) > self.scale_model_max_area:
                scale_model_factor = np.sqrt(self.scale_model_max_area / np.prod(self.init_target_sz))

            self.scale_model_sz = np.floor(np.multiply(self.init_target_sz, scale_model_factor))

            self.currentScaleFactor = 1

            # find maximum and minimum scales
            im = self.img_files[0]
            self.min_scale_factor = 0.9
            self.max_scale_factor = 1.1

            # TODO current are 0.1 and 15.3, seems wrong
            #self.min_scale_factor = np.power(
            #    self.scale_step,
            #    np.rint(np.log(np.amax(np.divide(5, sz))) / np.log(self.scale_step)))

            #self.max_scale_factor = np.power(
            #    self.scale_step,
            #    np.floor(
            #        np.divide(
            #            np.log(min(np.divide([im.shape[0], im.shape[1]], self.base_target_sz))),
            #            np.log(self.scale_step))))

        if self.frame.number > 1:

            # extract the test sample feature map for the scale filter
            xs = self.get_scale_sample(im,
                                       frame.predicted_position,
                                       self.base_target_sz,
                                       self.currentScaleFactor,
                                       self.scaleFactors,
                                       self.scale_window,
                                       self.scale_model_sz)

            # calculate the correlation response of the scale filter
            xsf = np.fft.fft2(xs)
            scale_response = np.real(np.fft.ifftn(np.divide(np.sum(np.multiply(self.sf_num, xsf), axis=0),
                                                            (self.sf_den + self.lam))))

            # find the maximum scale response
            # recovered_scale = np.nonzero(scale_response == np.amax(scale_response))

            scale_response = np.multiply(scale_response, self.scale_window) #TODO i think this makes sense?

            recovered_scale = np.argmax(scale_response)

            logger.info("factor response {0}".format(self.scaleFactors[recovered_scale]))

            # update the scale
            self.currentScaleFactor = self.currentScaleFactor * self.scaleFactors[recovered_scale]

            if self.currentScaleFactor < self.min_scale_factor:
                self.currentScaleFactor = self.min_scale_factor
            elif self.currentScaleFactor > self.max_scale_factor:
                self.currentScaleFactor = self.max_scale_factor

        # extract the training sample feature map for the scale filter, with the predicted size
        xs = self.get_scale_sample(im,
                                   frame.predicted_position,
                                   self.base_target_sz,
                                   self.currentScaleFactor,
                                   self.scaleFactors,
                                   self.scale_window,
                                   self.scale_model_sz)

        # calculate the scale filter update
        xsf = np.fft.fft2(xs)
        new_sf_num = np.multiply(self.ysf, np.conj(xsf))
        new_sf_den = np.sum(np.multiply(xsf, np.conj(xsf)), axis=0)

        if self.frame.number == 1:
            # first frame, train with a single image
            self.sf_den = new_sf_den
            self.sf_num = new_sf_num
        else:
            # subsequent frames, update the model
            self.sf_den = (1 - self.learning_rate) * self.sf_den + self.learning_rate * new_sf_den
            self.sf_num = (1 - self.learning_rate) * self.sf_num + self.learning_rate * new_sf_num

        logger.info("currentScaleFactor {0}".format(self.currentScaleFactor))
        target_sz = np.floor(np.multiply(self.base_target_sz, self.currentScaleFactor))

        return target_sz

    def get_scale_sample(self, im, pos, base_target_sz, currentScaleFactor, scale_factors, scale_window, scale_model_sz):

        scaleFactors = currentScaleFactor * scale_factors

        # Extracts a sample for the scale filter at the current
        # location and scale.

        nScales = len(scaleFactors)
        first = True

        for s in range(nScales):
            patch_sz = np.floor(np.multiply(base_target_sz, scaleFactors[s]))

            #xs = np.floor(pos.x) + np.arange(1, patch_sz[1] + 1) - np.floor(patch_sz[1] / 2)
            #ys = np.floor(pos.y) + np.arange(1, patch_sz[0] + 1) - np.floor(patch_sz[0] / 2)

            xs = np.floor(pos.x) + patch_sz[1] + 1 - np.floor(patch_sz[1] / 2)
            ys = np.floor(pos.y) + patch_sz[0] + 1 - np.floor(patch_sz[0] / 2)

            # check for out - of - bounds coordinates, and set them to the values at the borders
            if xs < 1:
                xs = 1

            if ys < 1:
                ys = 1

            if xs > np.shape(im)[0]:
                xs = np.shape(im)[0]

            if ys > np.shape(im)[1]:
                ys = np.shape(im)[1]

            # extract image
            # im_patch = (im[ys, xs])  # TODO is this right?
            #im_patch = im[
            #           int(ys - np.floor(patch_sz[0] / 2)):
            #           int(ys - np.floor(patch_sz[0] / 2) + patch_sz[0]),
            #           int(xs - np.floor(patch_sz[1] / 2)):
            #           int(xs - np.floor(patch_sz[1] / 2) + patch_sz[1])
            #           ]

            im_patch = im[
                int(self.frame.predicted_position.center[1] - np.floor(patch_sz[1] / 2)):
                int(self.frame.predicted_position.center[1] + np.ceil(patch_sz[1] / 2)),
                int(self.frame.predicted_position.center[0] - np.floor(patch_sz[0] / 2)):
                int(self.frame.predicted_position.center[0] + np.ceil(patch_sz[0] / 2))
            ]

            # img patch debugging, can be removed later
            if s == 16:
                img = Image.fromarray(im_patch)
                name = 'im_patches/' + str(self.frame.number) + '_' + str(pos.x) + 'd' + str(pos.y) + '.jpeg'
                img.save(name)
                #img.show()


            # resize image to model size
            # im_patch_resized = cv2.resize(im_patch, scale_model_sz)
            # im_patch_resized = cv2.resize(im_patch, (int(scale_model_sz[0]), int(scale_model_sz[1])))
            im_patch_resized = cv2.resize(im_patch, (32, 32))

            # extract scale features
            # winSize = (int(scale_model_sz[0]), int(scale_model_sz[1]))
            winSize = (32, 32)
            blockSize = (16, 16)
            blockStride = (8, 8)
            cellSize = (4, 4)
            nbins = 9
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            #hog = cv2.HOGDescriptor()
            temp_hog = hog.compute(im_patch_resized)
            # temp = temp_hog[:, :, 1: 31]  # ...TODO what?

            if first:
                out = np.zeros((np.size(temp_hog), nScales))
                first = False

            out[:, s] = np.multiply(temp_hog.flatten(), scale_window[s])

        # out becomes a matrix, where each column is the hog feature vector at a different scale lvl
        return out

    def test(self):
        print("Module import works")



