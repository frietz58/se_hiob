import numpy as np
import cv2
import logging
from ..Rect import Rect

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
        self.d_change_aspect_ratio = None
        self.hog_cell_size = None
        self.hog_block_norm_size = None

        # run time
        self.img_frames = None
        self.scale_factors = None
        self.x_scale_factors = None
        self.y_scale_factors = None
        self.current_scale_factor = None
        self.current_x_scale_factor = None
        self.current_y_scale_factor = None
        self.min_scale_factor = None
        self.max_scale_factor = None
        self.frame = None
        self.scale_samples = []
        self.scale_window = None
        self.ysf = None
        self.num = None
        self.den = None
        self.x_num = None
        self.y_num = None
        self.x_den = None
        self.y_den = None
        self.frame_history = []
        self.response_history = []
        self.init_target_size = None
        self.base_target_size = None
        self.sz = None
        self.scale_model_factor = None
        self.scale_model_max_area = None
        self.initialized_model = False

    def configure(self, conf, img_files):

        # reset values form previous session
        self.img_frames = None
        self.scale_factors = None
        self.x_scale_factors = None
        self.y_scale_factors = None
        self.current_scale_factor = None
        self.current_x_scale_factor = None
        self.current_y_scale_factor = None
        self.min_scale_factor = None
        self.max_scale_factor = None
        self.frame = None
        self.scale_samples = []
        self.scale_window = None
        self.ysf = None
        self.num = None
        self.den = None
        self.x_num = None
        self.y_num = None
        self.x_den = None
        self.y_den = None
        self.frame_history = []
        self.response_history = []
        self.init_target_size = None
        self.base_target_size = None
        self.sz = None
        self.scale_model_factor = None
        self.scale_model_max_area = None
        self.initialized_model = False
        self.hog_cell_size = None

        self.number_scales = conf['dsst_number_scales']
        self.scale_step = conf['scale_factor']
        self.scale_sigma_factor = conf['scale_sigma_factor']
        self.img_files = img_files
        self.lam = 1e-2
        self.learning_rate = conf['learning_rate']
        self.padding = conf['padding']
        self.scale_model_max_area = conf['scale_model_max']
        self.d_change_aspect_ratio = conf['d_change_aspect_ratio']

        hog_cell_size_tuple = (int(conf['hog_cell_size']), int(conf['hog_cell_size']))
        self.hog_cell_size = hog_cell_size_tuple

        hog_block_norm_tuple = (int(conf['hog_block_norm_size']), int(conf['hog_block_norm_size']))
        self.hog_block_norm_size = hog_block_norm_tuple

    def handle_initial_frame(self, frame):
        """
        Does typicall task that we need to do when  process the first frame of a tracking sequence, Like preparing the
        scale filter, calculating the punishment scale window, calculating the baseline scale factors that are adapted
        during tracking and preparing the scale model size considering the configurable hog parameters.
        :param frame: the current (first) frame
        :return: nothing, but sets the values on the dsst class object.
        """
        self.init_target_size = [frame.ground_truth.w, frame.ground_truth.h]
        self.base_target_size = [frame.ground_truth.w, frame.ground_truth.h]

        # padding used for what?
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

        if self.d_change_aspect_ratio:
            self.x_scale_factors = np.power(self.scale_step, (np.ceil(self.number_scales / 2) - ss))
            self.y_scale_factors = np.power(self.scale_step, (np.ceil(self.number_scales / 2) - ss))

        # compute resize dimensions for feature extraction
        self.scale_model_factor = 1
        if np.prod(self.init_target_size) > self.scale_model_max_area:
            self.scale_model_factor = np.sqrt(np.divide(self.scale_model_max_area, np.prod(self.init_target_size)))

        self.scale_model_size = np.floor(np.multiply(self.init_target_size, self.scale_model_factor))

        # find a size that is closest to multiple of the block size, which is the highest
        x_dim = self.get_closest_match(base=self.hog_block_norm_size[0], val=self.scale_model_size[0])
        y_dim = self.get_closest_match(base=self.hog_block_norm_size[0], val=self.scale_model_size[1])

        self.scale_model_size = [x_dim, y_dim]

        self.current_scale_factor = 1

        if self.d_change_aspect_ratio:
            self.current_x_scale_factor = 1
            self.current_y_scale_factor = 1

        self.min_scale_factor = np.power(self.scale_step, np.ceil(np.log(max(np.divide(5, self.sz))) / np.log(self.scale_step)))
        self.max_scale_factor = np.power(self.scale_step, np.floor(np.log(min(np.divide(
            (np.shape(frame.capture_image)[0], np.shape(frame.capture_image)[1]), self.base_target_size)))
                                                              / np.log(self.scale_step)))

    def dsst(self, frame):
        """
        The core dsst algorithm. This calculates the best scale for a given frame, given the predicted position and the
        scale model, which contains information about the past scale development of the object.
        :param frame: the current frame
        :return: a rectangle that corresponds to the best predicted size of the object
        """
        self.frame = frame

        # all frames except initial
        if self.initialized_model:

            # extract the test sample for the feature map for the scale filter
            sample = self.extract_scale_sample(self.frame)
            # print(sample.sum(axis=0))
            # print(np.argmax(sample.sum(axis=0)))

            # calculate the correlation response
            if not self.d_change_aspect_ratio:
                f_sample = np.fft.fft(sample)

                scale_response = np.divide(
                    np.sum(np.multiply(self.num, f_sample), axis=0),
                    (self.den + self.lam))

                # find the maximum scale response
                real_part = np.real(np.fft.ifftn(scale_response))
                recovered_scale = np.argmax(real_part)

                # update the scale
                self.current_scale_factor = self.current_scale_factor * self.scale_factors[recovered_scale]

                if self.current_scale_factor < self.min_scale_factor:
                    self.current_scale_factor = self.min_scale_factor
                elif self.current_scale_factor > self.max_scale_factor:
                    self.current_scale_factor = self.max_scale_factor

            elif self.d_change_aspect_ratio:
                x_f_sample = np.fft.fft(sample['x'])
                y_f_sample = np.fft.fft(sample['y'])

                x_scale_response = np.divide(
                    np.sum(np.multiply(self.x_num, x_f_sample), axis=0),
                    (self.x_den + self.lam))

                y_scale_response = np.divide(
                    np.sum(np.multiply(self.y_num, y_f_sample), axis=0),
                    (self.y_den + self.lam))

                # find the maximum scale responses
                x_real_part = np.real(np.fft.ifftn(x_scale_response))
                x_recovered_scale = np.argmax(x_real_part)

                y_real_part = np.real(np.fft.ifftn(y_scale_response))
                y_recovered_scale = np.argmax(y_real_part)

                # update the scale
                self.current_x_scale_factor = \
                    self.current_x_scale_factor * \
                    self.x_scale_factors[x_recovered_scale]

                self.current_y_scale_factor = \
                    self.current_y_scale_factor * \
                    self.y_scale_factors[y_recovered_scale]

        # extract training sample for current frame, with updated scale
        sample = self.extract_scale_sample(self.frame)

        # calculate scale filter update
        # for static aspect ratio
        if not self.d_change_aspect_ratio:
            f_sample = np.fft.fft(sample)

            new_num = np.multiply(self.ysf, np.conj(f_sample))
            new_den = np.sum(np.real(np.multiply(f_sample, np.conj(f_sample))), axis=0)

        # for dynamic aspect ratio
        elif self.d_change_aspect_ratio:
            x_f_sample = np.fft.fft(sample['x'])
            y_f_sample = np.fft.fft(sample['y'])

            new_x_num = np.multiply(self.ysf, np.conj(x_f_sample))
            new_y_num = np.multiply(self.ysf, np.conj(y_f_sample))

            new_x_den = np.sum(np.real(np.multiply(x_f_sample, np.conj(x_f_sample))), axis=0)
            new_y_den = np.sum(np.real(np.multiply(y_f_sample, np.conj(y_f_sample))), axis=0)

        # if initial frame, train on image
        if not self.initialized_model and not self.d_change_aspect_ratio:
            self.num = new_num
            self.den = new_den
            self.initialized_model = True

        elif not self.initialized_model and self.d_change_aspect_ratio:
            self.x_num = new_x_num
            self.y_num = new_y_num

            self.x_den = new_x_den
            self.y_den = new_y_den
            self.initialized_model = True

        # update the model
        elif self.initialized_model and not self.d_change_aspect_ratio:
            new_num = np.add(
                np.multiply((1 - self.learning_rate), self.num),
                np.multiply(self.learning_rate, new_num))

            new_den = np.add(
                np.multiply((1 - self.learning_rate), self.den),
                np.multiply(self.learning_rate, new_den))

            self.num = new_num
            self.den = new_den

        elif self.initialized_model and self.d_change_aspect_ratio:
            new_x_num = np.add(
                np.multiply((1 - self.learning_rate), self.x_num),
                np.multiply(self.learning_rate, new_x_num))

            new_y_num = np.add(
                np.multiply((1 - self.learning_rate), self.y_num),
                np.multiply(self.learning_rate, new_y_num))

            new_x_den = np.add(
                np.multiply((1 - self.learning_rate), self.x_den),
                np.multiply(self.learning_rate, new_x_den))

            new_y_den = np.add(
                np.multiply((1 - self.learning_rate), self.y_den),
                np.multiply(self.learning_rate, new_y_den))

            self.x_num = new_x_num
            self.y_num = new_y_num
            self.x_den = new_x_den
            self.y_den = new_y_den

        if not self.d_change_aspect_ratio:
            logger.info("predicted scale: {0}".format(self.current_scale_factor))
            new_target_size = np.floor(np.multiply(self.base_target_size, self.current_scale_factor))

        elif self.d_change_aspect_ratio:
            logger.info("predicted scale for x: {0}".format(self.current_x_scale_factor))
            logger.info("predicted scale for y: {0}".format(self.current_y_scale_factor))
            new_target_size = [
                np.floor(np.multiply(self.base_target_size[0], self.current_x_scale_factor)),
                np.floor(np.multiply(self.base_target_size[1], self.current_y_scale_factor))]

        # adjust x and y pos so that the box remains centered when height/width change
        old_x = self.frame.predicted_position.center[0]
        old_y = self.frame.predicted_position.center[1]

        new_x = int(old_x - np.rint(new_target_size[0] / 2))
        new_y = int(old_y - np.rint(new_target_size[1] / 2))

        return Rect(new_x, new_y, new_target_size[0], new_target_size[1])

    def extract_scale_sample(self, frame):
        """
        This method generates a scale sample for a given frame. One scale sample contains representations of the object
        (the predicted object location to be precise) at every scale level.
        :param frame: the current frame
        :return: a resized image patch for every scale level
        """
        global out
        self.frame = frame
        im = self.img_files[frame.number]

        # calculate the current scale factors, this depends on the current scale level and the base scale factors that
        # have been defined
        if not self.d_change_aspect_ratio:
            scale_factors = self.scale_factors * self.current_scale_factor

        elif self.d_change_aspect_ratio:
            x_scale_factors = self.x_scale_factors * self.current_x_scale_factor
            y_scale_factors = self.y_scale_factors * self.current_y_scale_factor

        # create an image patch for every scale lvl
        for i in range(self.number_scales):

            if not self.d_change_aspect_ratio:
                patch_size = np.floor(np.multiply(self.base_target_size, scale_factors[i]))

            elif self.d_change_aspect_ratio:
                unchanged_patch_size = self.base_target_size
                aspect_patch_size = [
                    int(np.floor(np.multiply(self.base_target_size[0], x_scale_factors[i]))),
                    int(np.floor(np.multiply(self.base_target_size[1], y_scale_factors[i])))]

            # if initial frame use annotated and static aspect ratio
            if self.frame.number == 1 and not self.d_change_aspect_ratio:
                xs = np.add(np.floor(self.frame.ground_truth.center[0]),
                            np.arange(1, patch_size[0] + 1)) - np.floor(patch_size[0] / 2)
                ys = np.add(np.floor(self.frame.ground_truth.center[1]),
                            np.arange(1, patch_size[1] + 1)) - np.floor(patch_size[1] / 2)

                # for later indexing, needs to be slices of ints
                xs = xs.astype(int)
                ys = ys.astype(int)

            # if initial frame and dynamic aspect ratio
            elif self.frame.number == 1 and self.d_change_aspect_ratio:
                # get the unchanged im patch for each axis
                x_unchanged = np.add(np.floor(self.frame.ground_truth.center[0]),
                                     np.arange(1, unchanged_patch_size[0] + 1)) - np.floor(unchanged_patch_size[0] / 2)

                y_unchanged = np.add(np.floor(self.frame.ground_truth.center[1]),
                                     np.arange(1, unchanged_patch_size[1] + 1)) - np.floor(unchanged_patch_size[1] / 2)

                # get the changed im patch for each axis
                xs = np.add(np.floor(self.frame.ground_truth.center[0]),
                            np.arange(1, aspect_patch_size[0] + 1)) - np.floor(aspect_patch_size[0] / 2)
                ys = np.add(np.floor(self.frame.ground_truth.center[1]),
                            np.arange(1, aspect_patch_size[1] + 1)) - np.floor(aspect_patch_size[1] / 2)

                # for later indexing, needs to be slices of ints
                x_unchanged = x_unchanged.astype(int)
                y_unchanged = y_unchanged.astype(int)
                xs = xs.astype(int)
                ys = ys.astype(int)

            # keep aspect ratio static
            elif self.frame.number != 1 and not self.d_change_aspect_ratio:
                xs = np.add(np.floor(self.frame.predicted_position.center[0]),
                            np.arange(1, patch_size[0] + 1)) - np.floor(patch_size[0] / 2)
                ys = np.add(np.floor(self.frame.predicted_position.center[1]),
                            np.arange(1, patch_size[1] + 1)) - np.floor(patch_size[1] / 2)

                # for later indexing, needs to be slices of ints
                xs = xs.astype(int)
                ys = ys.astype(int)

            # for dynamic aspect ratio
            elif self.frame.number != 1 and self.d_change_aspect_ratio:
                # get the unchanged im patch for each axis
                x_unchanged = np.add(np.floor(self.frame.predicted_position.center[0]),
                                     np.arange(1, unchanged_patch_size[0] + 1)) - np.floor(unchanged_patch_size[0] / 2)

                y_unchanged = np.add(np.floor(self.frame.predicted_position.center[1]),
                                     np.arange(1, unchanged_patch_size[1] + 1)) - np.floor(unchanged_patch_size[1] / 2)

                # get the changed im patch for each axis
                xs = np.add(np.floor(self.frame.predicted_position.center[0]),
                            np.arange(1, aspect_patch_size[0] + 1)) - np.floor(aspect_patch_size[0] / 2)
                ys = np.add(np.floor(self.frame.predicted_position.center[1]),
                            np.arange(1, aspect_patch_size[1] + 1)) - np.floor(aspect_patch_size[1] / 2)

                # for later indexing, needs to be slices of ints
                x_unchanged = x_unchanged.astype(int)
                y_unchanged = y_unchanged.astype(int)
                xs = xs.astype(int)
                ys = ys.astype(int)

            # check for out of bounds
            xs[xs < 1] = 1
            ys[ys < 1] = 1
            xs[xs > np.shape(im)[1] - 1] = np.shape(im)[1] - 1
            ys[ys > np.shape(im)[0] - 1] = np.shape(im)[0] - 1

            # also check oob of none changed dimensions, even thought those should never be oob
            if self.d_change_aspect_ratio:
                x_unchanged[x_unchanged < 1] = 1
                y_unchanged[y_unchanged < 1] = 1
                x_unchanged[x_unchanged > np.shape(im)[1] - 1] = np.shape(im)[1] - 1
                y_unchanged[y_unchanged > np.shape(im)[0] - 1] = np.shape(im)[0] - 1

            # dont change aspect ratio
            if not self.d_change_aspect_ratio:
                # extract one image patch corresponding to current scale
                img_patch = im[ys, :]
                img_patch = img_patch[:, xs]

                img_patch_resized = cv2.resize(img_patch,
                                               (int(self.scale_model_size[0]), int(self.scale_model_size[1])))

                # extract the hog features
                temp_hog = self.hog_vector(img_patch_resized)

                # init output sample, one column for each scale factor
                if i == 0:
                    out = np.zeros((np.size(temp_hog), self.number_scales))

                # punish each candidate based on its divergence to 1
                out[:, i] = np.multiply(temp_hog.flatten(), self.scale_window[i])

            # dynamic aspect ratio
            elif self.d_change_aspect_ratio:
                # create two image patches per scale level, only changing one axis for each
                x_changed_patch = im[:, xs]
                x_changed_patch = x_changed_patch[y_unchanged, :]

                y_changed_patch = im[ys, :]
                y_changed_patch = y_changed_patch[:, x_unchanged]

                # resize both patches
                x_patch_resized = cv2.resize(x_changed_patch,
                                             (int(self.scale_model_size[0]), int(self.scale_model_size[1])))
                y_patch_resized = cv2.resize(y_changed_patch,
                                             (int(self.scale_model_size[0]), int(self.scale_model_size[1])))

                # get hog representation for both patches
                x_hog = self.hog_vector(x_patch_resized)
                y_hog = self.hog_vector(y_patch_resized)

                if i == 0:
                    out = {'x': np.zeros((np.size(x_hog), self.number_scales)),
                           'y': np.zeros((np.size(y_hog), self.number_scales))}

                # punish each candidate based on its divergence to 1
                out['x'][:, i] = np.multiply(x_hog.flatten(), self.scale_window[i])
                out['y'][:, i] = np.multiply(y_hog.flatten(), self.scale_window[i])

        return out

    def hog_vector(self, img_patch_resized):
        """
        This method calculates the hog feature vector to describe a given image patch. Specifically, this method is used
        to generate the hog feature representation for all the extracted scale samples, after they have all been resized
        to one size
        :param img_patch_resized: the resized patch for which we which to obtain the hog feature vector
        :return: the hog feature vector
        """
        winSize = (int(self.scale_model_size[0]), int(self.scale_model_size[1]))
        blockSize = self.hog_block_norm_size  # for illumination: large block = local changes less significant
        blockStride = (int(self.hog_block_norm_size[0]/2), int(self.hog_block_norm_size[1]/2))  # overlap between blocks, typically 50% blocksize
        cellSize = self.hog_cell_size  # defines how big the features are that get extracted
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
    def get_closest_match(base, val):
        """
        This method is used for finding the closest match we can obtain by multiplying a base value with a factor to a
        second value val. We need to this to compute the size of the scale model, which must be compatible to the hog
        block normalization size
        :param base: the base value to which we try to find a close match with the second value by multiplying with a
        factor
        :param val: the second value
        :return: the value with the smallest absolute difference
        """
        # create a list of multiples
        multiples = []
        for i in range(1, 100):
            multiples.append(base * i)

        return min(multiples, key=lambda x: abs(x - val))


