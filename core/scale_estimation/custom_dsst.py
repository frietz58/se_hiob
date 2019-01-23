import numpy as np
import cv2
import logging
from math import gcd
from PIL import Image

logger = logging.getLogger(__name__)


class CustomDsst:

    def __init__(self):
        # configuration
        self.number_scales = None
        self.scale_step = None
        self.scale_sigma_factor = None
        self.img_files = None
        self.lam = None
        self.learning_rate = None
        self.scale_model_size = None
        self.padding = None

        # run time
        self.img_frames = None
        self.scale_factors = None
        self.current_scale_factor = None
        self.min_scale_factor = None
        self.max_scale_factor = None
        self.frame = None
        self.scale_samples = []
        self.scale_window = None
        self.ysf = None
        self.num = None
        self.den = None
        self.frame_history = []
        self.response_history = []
        self.init_target_size = None
        self.base_target_size = None
        self.sz = None
        self.scale_model_factor = None
        self.scale_model_max_area = None

    def configure(self, conf, img_files):
        self.number_scales = conf['number_scales']
        self.scale_step = conf['scale_factor']
        self.scale_sigma_factor = conf['scale_sigma_factor']
        self.img_files = img_files
        self.lam = 1e-2
        self.learning_rate = conf['learning_rate']
        self.static_model_size = conf['scale_model_size']
        self.padding = conf['padding']
        self.scale_model_max_area = conf['scale_model_max']

    def handle_initial_frame(self, frame):
        self.init_target_size = [frame.ground_truth.w, frame.ground_truth.h]
        self.base_target_size = [frame.ground_truth.w, frame.ground_truth.h]

        self.sz = np.floor(np.multiply(self.base_target_size, int(1 + self.padding)))

        # desired scale filter output (gaussian shaped), bandwidth proportional to number of scales
        scale_sigma = self.number_scales / np.sqrt(self.number_scales) * self.scale_sigma_factor
        ss = np.subtract(np.arange(1, self.number_scales + 1), np.ceil(self.number_scales / 2))
        ys = np.exp(-0.5 * (np.power(ss, 2)) / scale_sigma ** 2)
        self.ysf = np.fft.fft(ys)

        # store pre-computed scale filter cosine window
        if np.mod(self.number_scales, 2) == 0:
            self.scale_window = np.hanning(self.number_scales + 1)
            self.scale_window = self.scale_window[2: len(self.scale_window)]
        else:
            self.scale_window = np.hanning(self.number_scales)

        # scale factors
        ss = np.arange(1, self.number_scales + 1)
        self.scale_factors = np.power(self.scale_step, (np.ceil(self.number_scales / 2) - ss))

        # compute resize dimensions for feature extraction
        self.scale_model_factor = 1
        if np.prod(self.init_target_size) > self.scale_model_max_area:
            self.scale_model_factor = np.sqrt(np.divide(self.scale_model_max_area, np.prod(self.init_target_size)))

        # self.scale_model_size = np.floor(np.multiply(self.init_target_size, self.scale_model_factor))

        # find a size that is closest to multiple of 4 (dsst uses that as bin size)
        x_dim = self.get_closest_match(base=4, val=self.init_target_size[0])
        y_dim = self.get_closest_match(base=4, val=self.init_target_size[1])

        self.scale_model_size = [x_dim, y_dim]

        self.current_scale_factor = 1

        # TODO dynamic
        self.min_scale_factor = 0.15
        self.max_scale_factor = 13.6528

    def dsst(self, frame):

        self.frame = frame

        # all frames except initial
        if self.frame.number > 1:

            # extract the test sample for the feature map for the scale filter
            sample = self.extract_scale_sample(frame=self.frame)

            # calculate the correlation response
            f_sample = np.fft.fft2(sample)
            scale_response = np.divide(
                np.sum(np.multiply(self.num, f_sample), axis=0),
                (self.den + self.lam))

            # find the maximum scale response
            real_part = np.real(np.fft.ifftn(scale_response))
            recovered_scale = np.argmax(real_part)

            # update the scale
            self.current_scale_factor = self.scale_factors[recovered_scale]

            if self.current_scale_factor < self.min_scale_factor:
                self.current_scale_factor = self.min_scale_factor
            elif self.current_scale_factor > self.max_scale_factor:
                self.current_scale_factor = self.max_scale_factor

        # extract training sample for current frame, with updated scale
        sample = self.extract_scale_sample(self.frame)

        # calculate scale filter update
        f_sample = np.fft.fft2(sample)
        new_num = np.multiply(self.ysf, np.conj(f_sample))
        new_den = np.sum(np.multiply(f_sample, np.conj(f_sample)), axis=0)

        if frame.number == 1:
            # if initial frame, train on image
            self.num = new_num
            self.den = new_den
        else:
            # update the model
            new_num = np.add(
                np.multiply((1 - self.learning_rate), self.num),
                np.multiply(self.learning_rate, np.multiply(np.conj(self.ysf), f_sample)))

            new_den = np.add(
                np.multiply((1 - self.learning_rate), self.den),
                np.sum(np.multiply(self.learning_rate, np.multiply(np.conj(f_sample), f_sample)), axis=0)
            )

            self.num = new_num
            self.den = new_den

        logger.info("predicted scale: {0}".format(self.current_scale_factor))

        new_target_size = np.floor(np.multiply(self.base_target_size, self.current_scale_factor))

        return new_target_size

    def extract_scale_sample(self, frame):
        global out
        self.frame = frame
        im = self.img_files[frame.number]

        # calculate the current scale factors
        scale_factors = self.scale_factors * self.current_scale_factor

        # create an image patch for every scale lvl
        for i in range(self.number_scales):

            patch_size = np.floor(np.multiply(self.base_target_size, scale_factors[i]))

            # if initial frame use annotated
            if self.frame.number == 1:
                y0 = int(self.frame.ground_truth.center[1] - np.floor(patch_size[1] / 2))
                y1 = int(self.frame.ground_truth.center[1] + np.floor(patch_size[1] / 2))
                x0 = int(self.frame.ground_truth.center[0] - np.floor(patch_size[0] / 2))
                x1 = int(self.frame.ground_truth.center[0] + np.floor(patch_size[0] / 2))
            else:
                y0 = int(self.frame.predicted_position.center[1] - np.floor(patch_size[1] / 2))
                y1 = int(self.frame.predicted_position.center[1] + np.floor(patch_size[1] / 2))
                x0 = int(self.frame.predicted_position.center[0] - np.floor(patch_size[0] / 2))
                x1 = int(self.frame.predicted_position.center[0] + np.floor(patch_size[0] / 2))

            # check for out of bounds
            y0, y1, x0, x1 = self.check_oob(y0=y0, y1=y1, x0=x0, x1=x1, im=im)

            img_patch = im[y0:y1, x0:x1]
            cv2.imshow('test', img_patch)
            # TODO change axis??
            img_patch_resized = cv2.resize(img_patch, (int(self.scale_model_size[0]), int(self.scale_model_size[1])))

            # extract the hog features
            temp_hog = self.hog_vector(img_patch_resized)

            # init output sample, one column for each scale factor
            if i == 0:
                out = np.zeros((np.size(temp_hog), self.number_scales))

            out[:, i] = np.multiply(temp_hog.flatten(), self.scale_window[i])

        return out

    def hog_vector(self, img_patch_resized):
        # win_size = (int(self.scale_model_size[0]), int(self.scale_model_size[1]))
        winSize = (self.static_model_size, self.static_model_size)
        blockSize = (4, 4)  # for illumination: large block = local changes less significant
        blockStride = (2, 2)  # overlap between blocks, typically 50% blocksize
        cellSize = (4, 4)  # defines how big the features are that get extracted
        nbins = 9  # number of bins in histogram
        derivAperture = 1  # shouldn't be relevant
        winSigma = -1.  # shouldn't be relevant
        histogramNormType = 0  # shouldn't be relevant
        L2HysThreshold = 0.2  # shouldn't be relevant
        gammaCorrection = 1  # shouldn't be relevant
        nlevels = 64
        signedGradients = True  # 0 - 360 deg = True, 0 - 180 = false

        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

        temp_hog = hog.compute(img_patch_resized)

        return temp_hog

    @staticmethod
    def check_oob(y0, y1, x0, x1, im):

        if y0 < 0:
            y0 = 0
        if y0 > im.shape[0]:
            y0 = im.shape[0]

        if y1 < 0:
            y1 = 0
        if y1 > im.shape[0]:
            y1 = im.shape[0]

        if x0 < 0:
            x0 = 0
        if x0 > im.shape[1]:
            x0 = im.shape[1]

        if x1 < 0:
            x1 = 0
        if x1 > im.shape[1]:
            x1 = im.shape[1]

        return y0, y1, x0, x1

    @staticmethod
    def get_closest_match(base, val):
        # create a list of multiples
        multiples = []
        for i in range(1, 100):
            multiples.append(base * i)

        return min(multiples, key=lambda x:abs(x-val))
