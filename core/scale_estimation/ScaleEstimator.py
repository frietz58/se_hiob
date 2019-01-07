"""
Created on 2018-11-17

@author: Finn Rietz
"""


import numpy as np
from PIL import Image
import logging
import scipy.stats as st
import cv2
from .matlab_dsst import DsstEstimator
from .custom_dsst import CustomDsst
from .eco import Eco
from ..Rect import Rect

from matplotlib import pyplot as plt
from multiprocessing import Pool
import concurrent

logger = logging.getLogger(__name__)


class ScaleEstimator:

    #TODO make it a state machine

    def __init__(self):

        self.configuration = None
        self.econf = None
        self.dsst = DsstEstimator()
        self.custom_dsst = CustomDsst()

        self.frame = None
        self.tracker = None
        self.initial_size = None
        self.sample = None
        self.conj_G = None
        self.executor = None
        self.worker_count = None
        self.sample = None
        self.filter_history = []
        self.scaled_filters = []
        self.box_history = []
        self.dsst_numerator_a = []
        self.dsst_denominator_b = []

        #configuration
        self.use_scale_estimation = None
        self.number_scales = None
        self.inner_punish_threshold = None
        self.inner_punish_factor = None
        self.outer_punish_threshold = None
        self.scale_factor_range = None
        self.scale_factor = None
        self.learning_rate = None
        self.regularization = None
        self.scale_sigma_factor = None
        self.lam = None
        self.scale_model_max = None

    def setup(self, tracker=None, sample=None):
        self.tracker = tracker
        self.configuration = tracker.configuration
        self.sample = sample

        self.set_up_modules()


    def configure(self, configuration):
        self.econf = configuration['scale_estimator']
        self.use_scale_estimation = self.econf['use_scale_estimation']
        self.approach = self.econf['approach']
        self.inner_punish_threshold = self.econf['inner_punish_threshold']
        self.inner_punish_factor = self.econf['inner_punish_factor']
        self.outer_punish_threshold = self.econf['outer_punish_threshold']
        self.number_scales = self.econf['number_scales']
        self.scale_factor_range = self.econf['scale_factor_range']
        self.scale_factor = self.econf['scale_factor']
        self.learning_rate = self.econf['learning_rate']
        self.regularization = self.econf['reg']
        self.scale_sigma_factor = self.econf['scale_sigma_factor']
        self.scale_model_max = self.econf['scale_model_max']

        # logger is not initialized at this point, hence print statement...
        if self.use_scale_estimation:
            print("Scale Estimator has been configured")

    def set_up_modules(self):
        self.dsst.setup(n_scales=self.number_scales,
                        scale_step=self.scale_factor,
                        scale_sigma_factor=self.scale_sigma_factor,
                        img_files=self.sample.cv2_img_cache,
                        scale_model_max=self.scale_model_max,
                        learning_rate=self.learning_rate)

        self.custom_dsst.configure(n_scales=self.number_scales,
                                   scale_step=self.scale_factor,
                                   scale_sigma_factor=self.scale_sigma_factor,
                                   img_files=self.sample.cv2_img_cache,
                                   learning_rate=self.learning_rate)
        self.custom_dsst.initial_calculations()

    def estimate_scale(self, frame, feature_mask, mask_scale_factor, roi):
        """
        :param frame: the current frame in which the best position has already been calculated
        :param feature_mask: he consolidated feature mask containing pixel values for how likely they belong to the
        object
        :param mask_scale_factor: the factor with which the feature mask has been scaled to correspond to the actual ROI
        size, the cnn output is 48x48
        :param roi: the region of interest
        :return: the best rated candidate
        """

        self.frame = frame
        final_candidate = None

        # If scale estimation has been disabled in configuration, return unscaled bounding box
        if not self.use_scale_estimation:
            logger.critical("Scale Estimation is disabled, returning unchanged prediction")
            return frame.predicted_position

        if self.approach == 'candidates':
            logger.info("starting scale estimation. Approach: Candidate Generation")

            scaled_candidates = self.generate_scaled_candidates(frame)
            final_candidate = self.evaluate_scaled_candidates(scaled_candidates, feature_mask, mask_scale_factor, roi)

            logger.info("finished scale estimation")

        elif self.approach == 'mse':
            logger.info("starting scale estimation. Approach: Correlation Filter")
            scaled_filters = self.generate_scaled_filter()
            final_candidate = frame.predicted_position
            best_filter = self.evaluate_scaled_filters(scaled_filters)
            final_candidate = self.scale_prediction(best_filter, frame)

            logger.info("finished scale estimation")

        elif self.approach == 'dsst':
            logger.info("starting scale estimation. Approach: DSST")

            size = self.dsst.execute_scale_estimation(frame)
            frame.predicted_position = Rect(frame.predicted_position.x,
                                            frame.predicted_position.y,
                                            size[0],
                                            size[1])
            final_candidate = frame.predicted_position
            """
            scaled_samples = self.generate_scaled_patches()
            self.correlation_score_helper(scaled_samples)
            final_candidate = frame.predicted_position
            """
            logger.info("finished scale estimation")

        elif self.approach == "custom_dsst":
            logger.info("starting scale estimation. Approach: DSST")

            size = self.custom_dsst.dsst(frame)
            #TODO for debuggin use GT
            frame.predicted_position = Rect(frame.predicted_position.x,
                                            frame.predicted_position.y,
                                            size[0],
                                            size[1])
            final_candidate = frame.predicted_position

            logger.info("finished scale estimation")

        else:
            logger.critical("No implementation for approach in configuration")

        return final_candidate

    def handle_initial_frame(self, frame, sample):
        """
        :param sample: the current tracking sequence
        :param frame: the 0th frame
        :return:
        """
        self.frame = frame
        self.sample = sample

        if self.approach == 'mse':
            # create patch centered around target object (factor is 1, we know targets exact position and scale)
            scaled_width = int(round(self.frame.predicted_position.width * 1))
            scaled_height = int(round(self.frame.predicted_position.height * 1))

            current_cv2_im = self.sample.cv2_img_cache[self.sample.current_frame_id]
            im_x_coord = int(round(self.frame.predicted_position.centerx - scaled_width / 2))
            im_y_coord = int(round(self.frame.predicted_position.centery - scaled_height / 2))
            patch = current_cv2_im[
                    im_y_coord: im_y_coord + scaled_height,
                    im_x_coord: im_x_coord + scaled_width]
            np.save('initial_patch', patch)

            # create filter output (so that we can later solve for the actual filter)
            snth_filter_output = np.zeros((np.shape(patch)[0], np.shape(patch)[1]))

            # create 2d guassian
            gaussian = self.create_2d_gaussian_kernel(5, 2)

            # put gaussian at center of output/object
            # TODO make sure its actually in the center
            patch_x_coord = (round(np.shape(patch)[0] / 2) - round(np.shape(gaussian)[0] / 2))
            patch_y_coord = (round(np.shape(patch)[1] / 2) - round(np.shape(gaussian)[1] / 2))
            snth_filter_output[
            patch_y_coord: patch_y_coord + np.shape(gaussian)[1],
            patch_x_coord: patch_x_coord + np.shape(gaussian)[0]
            ] = gaussian
            np.save('patch_with_gauss', snth_filter_output)

            # bring both the patch and the filter output into the fourier domain and solve for filter
            patch_in_f = np.fft.fft2(patch)
            output_in_f = np.fft.fft2(snth_filter_output)
            filter = np.divide(output_in_f, patch_in_f)

            self.scaled_filters.append({'factor': 1, 'filter': filter, 'size': (scaled_width, scaled_height)})

        elif self.approach == "custom_dsst":
            self.custom_dsst.extract_scale_sample(frame, use_gt=True)

        elif self.approach == 'dsst':

            """
            current_cv2_im = self.sample.cv2_img_cache[self.sample.current_frame_id]
            self.initial_size = frame.predicted_position

            # create img patch of exact object
            target_patch = current_cv2_im[
                           self.frame.predicted_position.x:
                           self.frame.predicted_position.x + self.frame.predicted_position.width,
                           self.frame.predicted_position.y:
                           self.frame.predicted_position.x + self.frame.predicted_position.height
                           ]

            # get the hog feature vector of the patch, aspect ration needs to be 1:2
            resized_patch = cv2.resize(target_patch, dsize=(64, 128))
            hog = self.extract_hog_features(resized_patch)
            HOG = np.fft.fft(hog)
            conj_Hog = np.conj(HOG)

            # create target 1-d gaussian: TODO wie genau???
            #g = np.random.normal(loc=1, scale=1/16 * self.number_scales, size=(64 * 128))
            g = np.random.normal(loc=1, scale=1 / 16 * self.number_scales, size=10)
            G = np.fft.fft(g) #here
            self.conj_G = np.conj(G)

            # compute numerator and demominator
            a = np.multiply(self.learning_rate, np.multiply(self.conj_G, HOG))
            self.dsst_numerator_a.append(a)

            b = np.multiply(self.learning_rate, np.multiply(conj_Hog, HOG))
            self.dsst_denominator_b.append(b)
            """


        elif self.approach == 'candidates':
            # nothing needs to be done
            logger.info('')

    def generate_scaled_candidates(self, frame):
        """
        :param frame: the current frame in which the best position has already been calculated
        :return: a list of scaled variations of the best predicted position
        """

        self.frame = frame

        current_prediction = frame.predicted_position
        scaled_predictions = []

        # Generate n scaled candidates
        for i in range(self.number_scales):
            scaled_width = np.random.normal(loc=current_prediction.width, scale=1.0)
            scaled_height = np.random.normal(loc=current_prediction.height, scale=1.0)

            scaled_box = Rect(frame.predicted_position.x, frame.predicted_position.y, scaled_width, scaled_height)
            scaled_predictions.append(scaled_box)

        logger.info("created %s scaled candidates", len(scaled_predictions))

        # append current prediction as well, so that the array can be evaluated and that its possible,
        # that no changes in scale are necessary.
        scaled_predictions.append(current_prediction)

        return scaled_predictions

    def evaluate_scaled_candidates(self, scaled_candidates, feature_mask, mask_scale_factor, roi):
        """
        :param scaled_candidates: the candidates based on the best position but scaled in widht and height
        :param feature_mask: the consolidated feature mask containing pixel values for how likely they belong to the
        object
        :param mask_scale_factor: the factor with which the feature mask has been scaled to correspond to the actual ROI 
        size, the cnn output is 48x48
        :param roi: the region of interest
        :return: the best scaled candidate, can also be the original, not scaled candidate
        """

        logger.info("evaluating scaled candidates")

        # Apply the scaled candidates to the feature mask like mask[top:bottom,width:height]
        pixel_values = [feature_mask[
                        round(pos.top / mask_scale_factor[1]):
                        round((pos.bottom - 1) / mask_scale_factor[1]),
                        round(pos.left / mask_scale_factor[0]):
                        round((pos.right - 1) / mask_scale_factor[0])] for pos in scaled_candidates]

        # Sum up the values from the candidates
        candidate_sums = list(map(np.sum, pixel_values))

        # Evaluate each candidate based on its size
        evaluated_candidates = []

        for single_sum, candidate in zip(candidate_sums, scaled_candidates):
            evaluated_candidates.append(self.rate_scaled_candidate(single_sum,
                                                                   candidate,
                                                                   mask_scale_factor,
                                                                   feature_mask))

        best_candidate = np.argmax(evaluated_candidates)

        return Rect(scaled_candidates[best_candidate])

    def rate_scaled_candidate(self, candidate_sum, candidate, mask_scale_factor, feature_mask):
        """
        :param candidate_sum: the summed up pixel values of the candidate
        :param candidate: the current candidate
        :param mask_scale_factor: the factor with which the feature mask has been scaled to correspond to the actual ROI
        size, the cnn output is 48x48
        :param feature_mask: the consolidated feature mask containing pixel values for how likely they belong to the
        object
        :return: the quality of the candidate based on its size
        """

        # Calculate a score that punishes the candidate for containing pixel values < x
        inner_punish = np.where(feature_mask[
                        round(candidate.top / mask_scale_factor[1]):
                        round((candidate.bottom - 1) / mask_scale_factor[1]),
                        round(candidate.left / mask_scale_factor[0]):
                        round((candidate.right - 1) / mask_scale_factor[0])] < self.inner_punish_threshold)

        inner_punish_sum = np.sum(inner_punish) * self.inner_punish_factor

        # Calculate a score that punishes the candidate for not containing pixel values > x
        outer_punish = np.sum([np.where(feature_mask > self.outer_punish_threshold)])
        inner_helper = np.where(feature_mask[
                        round(candidate.top / mask_scale_factor[1]):
                        round((candidate.bottom - 1) / mask_scale_factor[1]),
                        round(candidate.left / mask_scale_factor[0]):
                        round((candidate.right - 1) / mask_scale_factor[0])] > self.outer_punish_threshold)

        inner_helper_sum = np.sum(inner_helper)
        outer_punish_sum = outer_punish - inner_helper_sum
        # TODO make this without the helper (only take the values that are bigger and not in the candidate...)


        # Evaluate the candidate
        quality_of_candidate = candidate_sum - (inner_punish_sum + outer_punish_sum)


        # logger.info("inner_punish_sum: {0}, outer_punish_sum: {1}, quality: {2} ".format(
        # inner_punish_sum,
        # outer_punish_sum,
        # quality_of_candidate))

        return quality_of_candidate

    def create_2d_gaussian_kernel(self, kernlen, nsig):
        """Returns a 2D Gaussian kernel array."""

        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def create_fourier_rep(self, tracker, frame):

        self.frame = frame
        self.sample = self.tracker.current_sample

        # get current frame and bring it into the fourier domain
        current_cv2_im = self.sample.cv2_img_cache[self.sample.current_frame_id]
        cv2_in_f = np.fft.fft2(current_cv2_im)

        #fshift = np.fft.fftshift(cv2_in_f)
        #magnitude_spectrum = 20*np.log(np.abs(fshift))
        #magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
        #np.save('mag_spec', magnitude_spectrum)

        cv2_back = np.fft.ifft2(cv2_in_f)
        cv2_back = np.abs(cv2_back)
        np.save('img_back',  cv2_back)

        # cv2_roi = current_cv2_im[frame.roi.x:frame.roi.x + frame.roi.w, frame.roi.y:frame.roi.y + frame.roi.h]

        # build synthetic img
        snth_filter_output = np.zeros((np.shape(frame.capture_image)[0], np.shape(frame.capture_image)[1]))

        # put 2d guassian where the object is
        gauss = self.create_2d_gaussian_kernel(frame.predicted_position.width/2, 2)
        #gauss = self.create_2d_gaussian_kernel(5, 2)
        x_coord = round(frame.predicted_position.centerx - (np.shape(gauss)[0] / 2))
        y_coord = round(frame.predicted_position.centery - (np.shape(gauss)[1] / 2))
        snth_filter_output[
            y_coord: y_coord + gauss.shape[0],
            x_coord: x_coord + gauss.shape[1]
        ] = gauss

        np.save('output', snth_filter_output)

        # get filter output in the fourier domain
        output_in_f = np.fft.fft2(snth_filter_output)

        #output_back = np.fft.ifft2(output_in_f)
        #output_back = np.abs(output_back)
        #np.save('output_back', output_back)

        # now we can solve for the filter like this: filter = snth_output/input_img
        filter = np.divide(output_in_f, cv2_in_f)
        con_filter = np.conjugate(filter)
        spatial_filter = np.fft.ifft2(con_filter)
        abs_filter = np.abs(spatial_filter)
        np.save('spatial_filter', abs_filter)

        #self.filter_history.append(abs_filter)

        #self.build_avg_filter()

        #x_coord_on_roi = round(frame.predicted_position.x - frame.roi.x + (np.shape(gauss)[0] / 2))
        #y_coord_on_roi = round(frame.predicted_position.y - frame.roi.y + (np.shape(gauss)[1] / 2))
        #snth_filter[x_coord_on_roi:x_coord_on_roi + gauss.shape[0], y_coord_on_roi:y_coord_on_roi + gauss.shape[1]] = gauss



        # create a pil image that can be seen in the gui output
        fshift = np.fft.fftshift(cv2_in_f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        pil_image = Image.fromarray(magnitude_spectrum, mode='RGB')

        return pil_image

    def append_to_history(self, frame):
        """
        Writes the final (scaled or unscaled) box from the current frame to the execution log
        :param frame: the current frame
        """
        if frame.number is not 0:
            self.box_history.append([frame.number,
                                     frame.predicted_position.x,
                                     frame.predicted_position.y,
                                     frame.predicted_position.width,
                                     frame.predicted_position.height])

            logger.info("Box at frame{0}: size: {5}x: {1}, y: {2}, width: {3}, height: {4}".format(
                frame.number,
                frame.predicted_position.x,
                frame.predicted_position.y,
                frame.predicted_position.width,
                frame.predicted_position.height, (
                        frame.predicted_position.width *
                        frame.predicted_position.height)
            ))

        return None

    def build_avg_filter(self):

        if len(self.filter_history) > 10:
            self.avg_filter = np.zeros(np.shape(self.filter_history[0]))
            for filter in self.filter_history:
                self.avg_filter = np.add(self.avg_filter, filter)

            self.avg_filter = np.divide(self.avg_filter, len(self.filter_history))
            self.avg_filter = np.fft.ifft2(self.avg_filter)
            self.avg_filter = np.abs(self.avg_filter)
            np.save('avg_filter', self.avg_filter)

        return None

    def generate_scaled_filter(self):
        """
        generates the image patches with different scale levels centered around the object and calculates the filters
        describing each patch.
        :return: An array with the filters for the different scale levels and the corresponding scale factors.
        """

        self.sample = self.tracker.current_sample

        # number_scales should always be odd, so that an equal amount is smaller and bigger and it contains 1
        scale_factors = np.linspace(self.scale_factor_range[0], self.scale_factor_range[1], num=self.number_scales)
        locale_scaled_factors = []

        for factor in scale_factors:

            # create patch centered around target object
            scaled_width = int(round(self.frame.predicted_position.width * factor))
            scaled_height = int(round(self.frame.predicted_position.height * factor))

            current_cv2_im = self.sample.cv2_img_cache[self.sample.current_frame_id]

            self.extract_hog_features(current_cv2_im)

            im_x_coord = int(round(self.frame.predicted_position.centerx - scaled_width / 2))
            im_y_coord = int(round(self.frame.predicted_position.centery - scaled_height / 2))
            patch = current_cv2_im[
                im_y_coord: im_y_coord + scaled_height,
                im_x_coord: im_x_coord + scaled_width]
            np.save('im_patch', patch)

            #resized_patch = cv2.resize(patch, dsize=(64, 128))
            #self.extract_hog_features(resized_patch)

            # resize the patches so that they are all of the size of the prev prediction
            patch = cv2.resize(patch, dsize=(
                np.shape(self.scaled_filters[self.sample.current_frame_id - 1]['filter'])[1],
                np.shape(self.scaled_filters[self.sample.current_frame_id - 1]['filter'])[0]))

            # create filter output (so that we can later solve for the actual filter)
            snth_filter_output = np.zeros((np.shape(patch)[0], np.shape(patch)[1]))

            #create 2d guassian
            gaussian = self.create_2d_gaussian_kernel(5, 2)

            # put gaussian at center of output/object
            #TODO make sure its actually in the center
            patch_x_coord = (round(np.shape(patch)[0] / 2) - round(np.shape(gaussian)[0] / 2))
            patch_y_coord = (round(np.shape(patch)[1] / 2) - round(np.shape(gaussian)[1] / 2))
            snth_filter_output[
            patch_y_coord: patch_y_coord + np.shape(gaussian)[1],
            patch_x_coord: patch_x_coord + np.shape(gaussian)[0]
            ] = gaussian
            np.save('patch_with_gauss', snth_filter_output)

            # bring both the patch and the filter output into the fourier domain and solve for filter
            patch_in_f = np.fft.fft2(patch)
            output_in_f = np.fft.fft2(snth_filter_output)
            filter = np.divide(output_in_f, patch_in_f)

            locale_scaled_factors.append({'factor': factor, 'filter': filter, 'size': (scaled_width, scaled_height)})

        return locale_scaled_factors

    def mse(self, a, b):
        """
        :param a: array one
        :param b: array two
        :return: the mse between the two array. Arrays must be os same dimensions
        """
        a = np.abs(a)
        b = np.abs(b)
        err = np.sum((a.astype("float") - b.astype("float")) ** 2)
        err /= float(a.shape[0] * b.shape[1])

        return err

    def evaluate_scaled_filters(self, filters):
        prev_best = self.scaled_filters[self.tracker.current_sample.current_frame_id - 1]
        mse_filters = []

        for filter_dict in filters:
            mse = self.mse(filter_dict['filter'], prev_best['filter'])
            mse_filters.append({'mse': mse, 'filter': filter_dict})
            smallest_mse = 1000000
            best_pair = None

        for pair in mse_filters:
            if pair['mse'] < smallest_mse:
                smallest_mse = pair['mse']
                best_pair = pair

        return best_pair['filter']

    def scale_prediction(self, filter, frame):
        scaled_width = int(round(self.frame.predicted_position.width * filter['factor']))
        scaled_height = int(round(self.frame.predicted_position.height * filter['factor']))

        scaled_box = Rect(frame.predicted_position.x, frame.predicted_position.y, scaled_width, scaled_height)
        self.scaled_filters.append(filter)

        return scaled_box

    def extract_hog_features(self, cv2_arr):
        # testing
        winSize = (32, 32)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (4, 4)
        nbins = 9
        #hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(cv2_arr)

        return hog_features

    def generate_scaled_patches(self):

        scale_sample = []
        upper_factor_limit = 1 + ((self.number_scales - 1) / 2) * 0.02 # TODO take this from the configuration
        bottom_factor_limit = 1 - ((self.number_scales - 1) / 2) * 0.02
        factor_range = np.linspace(bottom_factor_limit, upper_factor_limit, self.number_scales)

        for factor in factor_range:

            # create patch centered around target object and resize it to a fixed size
            scaled_width = int(round(self.frame.predicted_position.width * factor))
            scaled_height = int(round(self.frame.predicted_position.height * factor))

            current_cv2_im = self.sample.cv2_img_cache[self.sample.current_frame_id]

            im_x_coord = int(round(self.frame.predicted_position.centerx - scaled_width / 2))
            im_y_coord = int(round(self.frame.predicted_position.centery - scaled_height / 2))

            # make sure the coordinate are within the image
            if im_x_coord > 0 and im_y_coord > 0:
                patch = current_cv2_im[
                        im_y_coord: im_y_coord + scaled_height,
                        im_x_coord: im_x_coord + scaled_width]
                patch = cv2.resize(patch, (32, 128))
                np.save('im_patch', patch)
            else:
                logger.info('skipping patch because invalid position')

            # extract visual HOG features
            feat_vec = self.extract_hog_features(current_cv2_im)

            scale_sample.append({'factor': factor, 'img_patch': patch, 'features': feat_vec})

        return scale_sample

    def correlation_score_helper(self, scale_sample):

        #with concurrent.futures.ProcessPoolExecutor() as executor:
            #executor.map(self.multi_process_calc, scale_sample)

        for prev_a, curr_a in zip(self.dsst_numerator_a[self.frame.number - 1], scale_sample):
            a = np.multiply((1 - self.learning_rate), prev_a)
            step0 = np.multiply(self.conj_G, curr_a['features'])
            step1 = np.multiply(self.learning_rate, step0)
            a = np.add(a, step1)

        for prev_b, curr_b in zip(self.dsst_denominator_b[self.frame.number - 1], scale_sample):
            b = np.multiply((1 - self.learning_rate), prev_b)
            step0 = np.multiply(np.conj(scale_sample['features']), scale_sample['features'])
            step1 = np.multiply(self.learning_rate, step0)
            b = np.add(b, step1)

        self.dsst_numerator_a.append(a),
        self.dsst_denominator_b.append(b)

    def multi_process_calc(self, scale_sample):

        for prev_a, curr_a in zip(self.dsst_numerator_a[self.frame.number - 1], scale_sample):
            a = np.multiply((1 - self.learning_rate), prev_a)
            step0 = np.multiply(self.conj_G, curr_a['features'])
            step1 = np.multiply(self.learning_rate, step0)
            a = np.add(a, step1)

        for prev_b, curr_b in zip(self.dsst_denominator_b[self.frame.number - 1], scale_sample):
            b = np.multiply((1 - self.learning_rate), prev_b)
            step0 = np.multiply(np.conj(scale_sample['features']), scale_sample['features'])
            step1 = np.multiply(self.learning_rate, step0)
            b = np.add(b, step1)

        self.dsst_numerator_a.append(a),
        self.dsst_denominator_b.append(b)

