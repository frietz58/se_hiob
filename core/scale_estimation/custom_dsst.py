import numpy as np
import cv2
import logging
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

    def configure(self, n_scales, scale_step, scale_sigma_factor, img_files, learning_rate):
        self.number_scales = n_scales
        self.scale_step = scale_step
        self.scale_sigma_factor = scale_sigma_factor
        self.img_files = img_files
        self.lam = 1e-2
        self.learning_rate = learning_rate

    def initial_calculations(self):

        self.response_history = np.zeros((33, len(self.img_files)))

        # scale factors
        ss = np.arange(1, self.number_scales + 1)
        self.scale_factors = np.power(self.scale_step, (np.ceil(self.number_scales / 2) - ss))

        self.current_scale_factor = 1

        # TODO dynamic depending on scale step and number scales
        self.min_scale_factor = 0.95
        self.max_scale_factor = 1.05

        # store pre-computed scale filter cosine window
        if np.mod(self.number_scales, 2) == 0:
            self.scale_window = np.hanning(self.number_scales + 1)
            self.scale_window = self.scale_window[2: len(self.scale_window)]
        else:
            self.scale_window = np.hanning(self.number_scales)

        # desired scale filter output (gaussian shaped), bandwidth proportional to number of scales
        scale_sigma = self.number_scales / np.sqrt(self.number_scales) * self.scale_sigma_factor
        ss = np.subtract(np.arange(1, self.number_scales + 1), np.ceil(self.number_scales / 2))
        ys = np.exp(-0.5 * (np.power(ss, 2)) / scale_sigma ** 2)
        self.ysf = np.fft.fft(ys)

    def extract_scale_sample(self, frame, use_gt=False):
        global out
        self.frame = frame
        im = self.img_files[frame.number]


        # if initial frame use annotated
        if True:
            base_target_size = (self.frame.ground_truth.width, self.frame.ground_truth.height)
        else:
            base_target_size = (self.frame.predicted_position.width, self.frame.predicted_position.height)

        # create an image patch for every scale lvl
        for i in range(self.number_scales):

            patch_size = np.rint(np.multiply(base_target_size, self.scale_factors[i]))

            # if initial frame use annotated
            if True:
                y0 = int(self.frame.ground_truth.center[1] - np.rint(patch_size[1] / 2))
                y1 = int(self.frame.ground_truth.center[1] + np.rint(patch_size[1] / 2))
                x0 = int(self.frame.ground_truth.center[0] - np.rint(patch_size[0] / 2))
                x1 = int(self.frame.ground_truth.center[0] + np.rint(patch_size[0] / 2))
            else:
                y0 = int(self.frame.predicted_position.center[1] - np.rint(patch_size[1] / 2))
                y1 = int(self.frame.predicted_position.center[1] + np.rint(patch_size[1] / 2))
                x0 = int(self.frame.predicted_position.center[0] - np.rint(patch_size[0] / 2))
                x1 = int(self.frame.predicted_position.center[0] + np.rint(patch_size[0] / 2))

            # check for out of bounds
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

            img_patch = im[y0:y1, x0:x1]
            img_patch_resized = cv2.resize(img_patch, (16, 16))

            if i == 0 or i == 16 or i == 32:
                patch_img = Image.fromarray(img_patch_resized)
                #patch_img.show()

            # extract the hog features
            temp_hog = self.hog_vector(img_patch_resized)

            # init output sample, one column for each scale factor
            if i == 0:
                out = np.zeros((np.size(temp_hog), self.number_scales))

            out[:, i] = np.multiply(temp_hog.flatten(), self.scale_window[i])
            #out[:, i] = np.multiply(temp_hog[:, 0], self.scale_window[i])

        # if initial frame, calculate num and den here, otherwise in main part of algorithm
        if use_gt:
            f_out = np.fft.fft2(out)
            self.num = np.multiply(self.learning_rate, np.multiply(np.conj(self.ysf), f_out))
            self.den = np.sum(np.multiply(f_out, np.conj(f_out)), axis=0)
            self.frame_history.append(self.frame)

        self.scale_samples.append(out)

        return out

    def dsst(self, frame):
        approach = "dsst"

        self.frame = frame
        self.frame_history.append(self.frame)  # for own idea...
        base_target_size = (self.frame.predicted_position.width, self.frame.predicted_position.height)
        #self.current_scale_factor = 1

        # extract sample for current frame
        sample = self.extract_scale_sample(self.frame)

        # calculate scale factor by comparing current sample to model
        f_sample = np.fft.fft2(sample)

        # extract features from previews frames to compare to
        if approach == "own":
            avg_target = np.zeros(64)
            first = True
            for i in range(0, len(self.frame_history)):

                # ignore the current frame, only build avg model from past frames
                if i == 0:
                    continue

                # only build the avg over the last 5 frames, to stay up to date and avoid computational sink
                if i == 5:
                    break

                # build average target, older samples decaying
                punisher = (1 - self.learning_rate) ** i
                prev_img = self.img_files[self.frame.number - i]
                prev_frame = self.frame_history[-i]
                prev_patch = prev_img[
                             prev_frame.predicted_position.y:
                             prev_frame.predicted_position.y + prev_frame.predicted_position.height,
                             prev_frame.predicted_position.x:
                             prev_frame.predicted_position.x + prev_frame.predicted_position.width
                             ]
                resized_patch = cv2.resize(prev_patch, (32, 32))
                resized_img = Image.fromarray(resized_patch)
                #resized_img.show()
                hog = self.hog_vector(resized_patch)
                punished_hog = np.multiply(hog, punisher)

                if first:
                    avg_target = np.zeros(np.shape(hog))
                    first = False

                avg_target = np.add(avg_target, punished_hog)

            avg_target = np.divide(avg_target, len(self.frame_history))
            f_avg_target = np.fft.fft2(avg_target)

            scale_response = np.sum(np.multiply(f_avg_target, f_sample), axis=0)

        elif approach == "dsst":
            scale_response = np.divide(
                np.sum(np.multiply(self.num, f_sample), axis=0),
                (self.den + self.lam))

        else:
            print("no matching implementation")

        real_part = np.real(np.fft.ifftn(scale_response))
        self.response_history[:, self.frame.number] = real_part

        recovered_scale = np.argmax(real_part)
        scale_change = self.scale_factors[recovered_scale]

        # prevent outliers from distorting the scale
        if scale_change < self.min_scale_factor:
            scale_change = self.min_scale_factor
        elif scale_change > self.max_scale_factor:
            scale_change = self.max_scale_factor

        logger.info("estimated scale {0}".format(scale_change))

        new_target_size = np.rint(np.multiply(base_target_size, scale_change))

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

        if self.frame.number == len(self.img_files) - 2:
            print("look at response history")

        return new_target_size

    @staticmethod
    def hog_vector(img_patch_resized, use_man=False):

        if use_man:
            print("manual_feature extraction")

        else:
            win_size = (16, 16)
            block_size = (4, 4)
            block_stride = (4, 4)
            cell_size = (2, 2)
            n_bins = 9
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
            temp_hog = hog.compute(img_patch_resized)

        return temp_hog

