"""
Created on 2018-11-17

@author: Finn Rietz
"""


import numpy as np
from ..Rect import Rect
import cv2
import transitions

import logging
logger = logging.getLogger(__name__)


class ScaleEstimator():

    #TODO make it a state machine

    def __init__(self):

        self.frame = None
        self.number_scales = None
        self.tracker = None
        self.box_history = []
        self.configuration = None


    def setup(self, tracker=None):

        self.tracker = tracker
        logger.info("scale estimator ready")

    def configure(self, conf):
        self.configuration = conf
        logger.info("estimator conf set")
        # TODO why does this never happen?

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

        logger.info("starting scale estimation")

        scaled_candidates = self.generate_scaled_candidates(frame)
        final_candidate = self.evaluate_scaled_candidates(scaled_candidates, feature_mask, mask_scale_factor, roi)

        logger.info("finished scale estimation")

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
        for i in range(33):
            scaled_width = np.random.normal(loc=current_prediction.width, scale=1.0)
            scaled_height = np.random.normal(loc=current_prediction.height, scale=1.0)

            scaled_box = Rect(frame.predicted_position.x, frame.predicted_position.y, scaled_width, scaled_height)
            scaled_predictions.append(scaled_box)

        logger.info("created %s scaled positions", len(scaled_predictions))

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

        logger.info("type: %s", type(mask_scale_factor))

        # Sum up the values from the candidates
        candidate_sums = list(map(np.sum, pixel_values))

        # Evaluate each candidate based on its size
        evaluated_candidates = []

        for single_sum, candidate in zip(candidate_sums, scaled_candidates):
            evaluated_candidates.append(self.rate_scaled_candidate(single_sum,
                                                                   candidate,
                                                                   mask_scale_factor,
                                                                   feature_mask,
                                                                   roi))

        best_candidate = np.argmax(evaluated_candidates)

        return Rect(scaled_candidates[best_candidate])

    def rate_scaled_candidate(self, candidate_sum, candidate, mask_scale_factor, feature_mask, roi):
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

        # Calculate the score of the candidate based on its pixel size
        inner_punish = np.where(feature_mask[
                        round(candidate.top / mask_scale_factor[1]):
                        round((candidate.bottom - 1) / mask_scale_factor[1]),
                        round(candidate.left / mask_scale_factor[0]):
                        round((candidate.right - 1) / mask_scale_factor[0])] < 0.1)

        inner_punish_sum = np.sum(inner_punish)
        # inner_punish_sum = inner_punish.size

        # Calculate the score of of the pixels that are in the the feature mask but not in the candidate
        outer_punish = np.sum([np.where(feature_mask > 0.5)])
        # outer_punish = [np.where(feature_mask > 0.5)].size
        inner_helper = np.where(feature_mask[
                        round(candidate.top / mask_scale_factor[1]):
                        round((candidate.bottom - 1) / mask_scale_factor[1]),
                        round(candidate.left / mask_scale_factor[0]):
                        round((candidate.right - 1) / mask_scale_factor[0])] > 0.5)

        inner_helper_sum = np.sum(inner_helper)
        # inner_helper_sum = inner_helper.size
        outer_punish_sum = outer_punish - inner_helper_sum


        # Evaluate the candidate
        quality_of_candidate = candidate_sum - ((inner_punish_sum * 0.5) + outer_punish_sum)
        # quality_of_candidate = candidate_sum - (outer_punish_sum)
        # logger.info("inner_punish_sum: {0}, outer_punish_sum: {1}, quality: {2} ".format(inner_punish_sum, outer_punish_sum, quality_of_candidate))

        return quality_of_candidate

    def create_fourier_rep(self, frame=None):
        logger.info("creating fourier representation")
        #logger.info("frame.capture_iamge %s", frame.capture_image)
        #img = cv2.imread(frame.capture_image)
        return frame

    def append_to_history(self, frame):
        self.box_history.append([frame.number, frame.predicted_position.x, frame.predicted_position.y, frame.predicted_position.width, frame.predicted_position.height])

        logger.info("Box at frame{0}: size: {5}x: {1}, y: {2}, width: {3}, height: {4}".format(frame.number,
                                                                                      frame.predicted_position.x,
                                                                                      frame.predicted_position.y,
                                                                                      frame.predicted_position.width,
                                                                                      frame.predicted_position.height,
                                                                                        (frame.predicted_position.width * frame.predicted_position.height)))




