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

        # The factor with which the feature mask has been scaled.
        squared_mask_scale_factor = mask_scale_factor[0] * mask_scale_factor[1]

        # Calculate the score of the candidate based on its pixel size
        inner = candidate_sum
        inner_fill = candidate_sum / (candidate.pixel_count() / squared_mask_scale_factor)

        # Calculate the score of of the pixels that are in the the feature mask but not in the candidate
        outer = feature_mask.sum() - inner
        outer_fill = outer / max((roi.pixel_count() - candidate.pixel_count()) / squared_mask_scale_factor, 1)

        # Evaluate the candidate
        #quality_of_candidate = max(inner_fill - outer_fill, 0.0)

        quality_of_candidate = inner
        return quality_of_candidate

    def create_fourier_rep(self, frame=None):
        logger.info("creating fourier representation")
        #logger.info("frame.capture_iamge %s", frame.capture_image)
        #img = cv2.imread(frame.capture_image)
        return frame

    def append_to_history(self, frame):
        self.box_history.append([frame.number, frame.predicted_position.x, frame.predicted_position.y, frame.predicted_position.width, frame.predicted_position.height])

        logger.info("Box at frame{0}: x: {1}, y: {2}, width: {3}, height: {4}".format(frame.number,
                                                                                      frame.predicted_position.x,
                                                                                      frame.predicted_position.y,
                                                                                      frame.predicted_position.width,
                                                                                      frame.predicted_position.height))




