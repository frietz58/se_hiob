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
from ..Rect import Rect
from .candidates import CandidateApproach

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
        self.candidate_approach = CandidateApproach()

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
        self.se_time = 0.0
        self.passed_since_last_se = 0

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
        self.approach = None
        self.scale_model_size = None
        self.padding = None
        self.dyn_min_se_treshold = None
        self.dyn_max_se_treshold = None
        self.update_strategy = None
        self.static_update_val = None

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
        self.outer_punish_threshold = self.econf['outer_punish_threshold']
        self.scale_factor = self.econf['scale_factor']
        self.learning_rate = self.econf['learning_rate']
        self.regularization = self.econf['reg']
        self.scale_sigma_factor = self.econf['scale_sigma_factor']
        self.scale_model_max = self.econf['scale_model_max']
        self.scale_model_size = self.econf['scale_model_size']
        self.padding = self.econf['padding']
        self.dyn_min_se_treshold = self.econf['dyn_min_se_treshold']
        self.dyn_max_se_treshold = self.econf['dyn_max_se_treshold']
        self.update_strategy = self.econf['update_strategy']
        self.static_update_val = self.econf['static_update_val']
        self.passed_since_last_se = self.econf['static_update_val']

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

        self.custom_dsst.configure(self.econf, img_files=self.sample.cv2_img_cache)

        self.candidate_approach.configure(self.econf)

    def estimate_scale(self, frame, feature_mask, mask_scale_factor, prediction_quality, tracking):
        """
        :param frame: the current frame in which the best position has already been calculated
        :param feature_mask: he consolidated feature mask containing pixel values for how likely they belong to the
        object
        :param mask_scale_factor: the factor with which the feature mask has been scaled to correspond to the actual ROI
        size, the cnn output is 48x48
        :param prediction_quality: the quality of the prediction for the current frame.
        :return: the best rated candidate
        """

        self.frame = frame
        final_candidate = None

        # If scale estimation has been disabled in configuration, return unscaled bounding box
        if not self.use_scale_estimation:
            logger.info("Scale Estimation is disabled, returning unchanged prediction")
            return frame.predicted_position

        # update strategies:
        # continuous, update on every frame
        if self.update_strategy == "cont":
            final_candidate = self.execute_se_algorithm(frame, feature_mask, mask_scale_factor, tracking)

        # static, update every x frames
        elif self.update_strategy == "static":
            if self.passed_since_last_se == self.static_update_val:
                final_candidate = self.execute_se_algorithm(frame, feature_mask, mask_scale_factor, tracking)
                self.passed_since_last_se = 0
            else:
                self.passed_since_last_se += 1
                final_candidate = frame.predicted_position

        # dynamic, update every time the quality gets smaller than threshold, indicating change in appearance
        elif self.update_strategy == "dynamic":
            if prediction_quality <= self.dyn_max_se_treshold:
                final_candidate = self.execute_se_algorithm(frame, feature_mask, mask_scale_factor, tracking)
            else:
                final_candidate = frame.predicted_position

        # limited, dynamic but only if quality still good enough
        elif self.update_strategy == "limited":
            if self.dyn_max_se_treshold >= prediction_quality >= self.dyn_min_se_treshold:
                final_candidate = self.execute_se_algorithm(frame, feature_mask, mask_scale_factor, tracking)
            else:

                final_candidate = frame.predicted_position

        # if the quality of the prediction is too low. return unscaled bounding box
        # if prediction_quality < self.min_se_treshold and self.use_update_strategies:
        #     logger.info("frame prediction quality is smaller than scale estimation threshold {0}, not changing"
        #                 " the size".format(self.min_se_treshold))
        #     return frame.predicted_position

        # if quality of prediction is too high no SE needed
        # if prediction_quality > self.max_se_treshold and self.use_update_strategies:
        #    logger.info("frame prediction quality bigger than scale estimation threshold {0}, not changing"
        #                " the size".format(self.max_se_treshold))
        #    return frame.predicted_position

        # final_candidate = self.execute_se_algorithm(frame, feature_mask, mask_scale_factor)

        # if self.approach == 'candidates':
        #     logger.info("starting scale estimation. Approach: Candidate Generation")
        #
        #     scaled_candidates = self.candidate_approach.generate_scaled_candidates(frame)
        #     final_candidate = self.candidate_approach.evaluate_scaled_candidates(scaled_candidates,
        #                                                                          feature_mask,
        #                                                                          mask_scale_factor)
        #
        #     logger.info("finished scale estimation")
        #
        # elif self.approach == "custom_dsst":
        #     logger.info("starting scale estimation. Approach: DSST")
        #
        #     final_candidate = self.custom_dsst.dsst(frame)
        #
        #     logger.info("finished scale estimation")
        #
        # else:
        #     logger.critical("No implementation for approach in configuration")

        return final_candidate

    def handle_initial_frame(self, frame, sample):
        """
        :param sample: the current tracking sequence
        :param frame: the 0th frame
        :return:
        """
        if not self.use_scale_estimation:
            return None

        self.frame = frame
        self.sample = sample

        if self.approach == "custom_dsst":
            self.custom_dsst.handle_initial_frame(frame=frame)

        elif self.approach == 'candidates':
            # nothing needs to be done
            self.candidate_approach.handle_initial_frame(frame)

    def execute_se_algorithm(self, frame, feature_mask, mask_scale_factor, tracking):
        if self.approach == 'candidates':
            logger.info("starting scale estimation. Approach: Candidate Generation")

            scaled_candidates = self.candidate_approach.generate_scaled_candidates(frame, tracking)
            final_candidate = self.candidate_approach.evaluate_scaled_candidates(scaled_candidates,
                                                                                 feature_mask,
                                                                                 mask_scale_factor,
                                                                                 )

            logger.info("finished scale estimation")

        elif self.approach == "custom_dsst":
            logger.info("starting scale estimation. Approach: DSST")

            final_candidate = self.custom_dsst.dsst(frame, tracking)

            logger.info("finished scale estimation")

        else:
            logger.critical("No implementation for approach in configuration")
            final_candidate = None

        return final_candidate
