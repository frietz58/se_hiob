"""
Created on 2018-11-17

@author: Finn Rietz
"""


import numpy as np
from PIL import Image, ImageDraw
import logging
from scipy.ndimage import gaussian_filter
import scipy.stats as st
import cv2
import transitions
from matplotlib import pyplot as plt
import pandas as pd

from ..Rect import Rect

logger = logging.getLogger(__name__)


class ScaleEstimator():

    #TODO make it a state machine

    def __init__(self):

        self.configuration = None
        self.econf = None

        self.frame = None
        self.tracker = None
        self.box_history = []
        self.sample = None

        #configuration
        self.use_scale_estimation = None
        self.number_scales = None
        self.inner_punish_threshold = None
        self.inner_punish_factor = None
        self.outer_punish_threshold = None

    def setup(self, tracker=None):
        self.tracker = tracker
        self.configuration = tracker.configuration

    def configure(self, configuration):
        self.econf = configuration['scale_estimator']
        self.use_scale_estimation = self.econf['use_scale_estimation']
        self.approach = self.econf['approach']
        self.inner_punish_threshold = self.econf['inner_punish_threshold']
        self.inner_punish_factor = self.econf['inner_punish_factor']
        self.outer_punish_threshold = self.econf['outer_punish_threshold']

        # logger is not initialized at this point, hence print statement...
        if self.use_scale_estimation:
            print("Scale Estimator has been configured")

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

        # If scale estimation has been disabled in configuration, return unscaled bounding box
        if not self.use_scale_estimation:
            logger.critical("Scale Estimation is disabled, returning unchanged prediction")
            return frame.predicted_position

        if self.approach == 'candidates':

            logger.info("starting scale estimation. Approach: Candidate Generation")

            scaled_candidates = self.generate_scaled_candidates(frame)
            final_candidate = self.evaluate_scaled_candidates(scaled_candidates, feature_mask, mask_scale_factor, roi)

            logger.info("finished scale estimation")

        elif self.approach == 'correlation':

            logger.info("starting scale estimation. Approach: Correlation Filter")

            self.create_fourier_rep(self.tracker, self.frame)
            final_candidate = frame.predicted_position
            logger.info("finished scale estimation")
        else:
            logger.critical("No implementation for approach in configuration")

        return final_candidate

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

        # append current prediction aswell, so that the array can be evaluated and that its possible,
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
        :param roi: the region of interest
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

        # arr = np.asarray(self.tracker.sroi_generator.generated_sroi.read_value().eval(), dtype=np.uint8)
        # im = Image.fromarray(arr[0])

        self.frame = frame
        self.sample = self.tracker.current_sample

        current_cv2_im = self.sample.cv2_img_cache[self.sample.current_frame_id]
        cv2_roi = current_cv2_im[frame.roi.x:frame.roi.x + frame.roi.w, frame.roi.y:frame.roi.y + frame.roi.h]


        im = np.asarray(self.tracker.sroi_generator.generated_sroi.read_value().eval(), dtype=np.uint8)[0]
        # im = np.asanyarray((self.image_path))
        img = frame.capture_image
        np.save('cap', img)

        # Bring the image into the frequency domain and shift it, so that the zero frequency component (dc compoennt) is
        # at the center of the image, instead of top left
        # Then find the magnitude spectrum ?
        roi_in_fourier = np.fft.fft2(img)
        fshift = np.fft.fftshift(roi_in_fourier)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        np.save('mag_spec', magnitude_spectrum)

        # build synthetic img
        snth_filter = np.zeros((frame.roi.width, frame.roi.height))

        # put 2d guassian at center of object
        gauss = self.create_2d_gaussian_kernel(5, 2)
        #x_coord = round(frame.predicted_position.centerx + np.shape(gauss)[0]/2)
        #y_coord = round(frame.predicted_position.centery + np.shape(gauss)[1]/2)

        x_coord_on_roi = round(frame.predicted_position.x - frame.roi.x + (np.shape(gauss)[0] / 2))
        y_coord_on_roi = round(frame.predicted_position.y - frame.roi.y + (np.shape(gauss)[1] / 2))

        snth_filter[x_coord_on_roi:x_coord_on_roi + gauss.shape[0], y_coord_on_roi:y_coord_on_roi + gauss.shape[1]] = gauss


        #plt.interactive(True)
        #plt.imshow(magnitude_spectrum, cmap='gray')
        #plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        #plt.show(block=True)
        #plt.show()


        # apply filter/do stuff in frequency domain
        #rows = im.shape[0]
        #cols = im.shape[1]
        #crow, ccol = rows / 2, cols / 2
        #fshift[int(crow - 30): int(crow + 30), int(ccol - 30):int(ccol + 30)] = 0

        # bring the image back into the spaial domain
        #f_ishift = np.fft.ifftshift(fshift)
        #img_back = np.fft.ifft2(f_ishift)
        #img_back = np.abs(img_back)

        # create a pil image that can be seen in the gui output
        pil_image = Image.fromarray(magnitude_spectrum, mode='RGB')

        return pil_image

    def append_to_history(self, frame):
        """
        Writes the final (scaled or unscaled) box from the current frame to the execution log
        :param frame: the current frame
        """
        self.box_history.append([frame.number, frame.predicted_position.x, frame.predicted_position.y, frame.predicted_position.width, frame.predicted_position.height])

        logger.info("Box at frame{0}: size: {5}x: {1}, y: {2}, width: {3}, height: {4}".format(
            frame.number,
            frame.predicted_position.x,
            frame.predicted_position.y,
            frame.predicted_position.width,
            frame.predicted_position.height, (
                    frame.predicted_position.width *
                    frame.predicted_position.height)
        ))




