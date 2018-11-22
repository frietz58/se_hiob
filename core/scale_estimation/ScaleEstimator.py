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

    #TODO do i really want these?

    module_states = [
        'created',
        'estimating_scales',
        'done',
    ]
    transitions = [
        ['created', 'estimating_scales'],
        ['estimating_scales', 'done'],
    ]

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


    def estimate_scale(self, frame):

        self.frame = frame
        logger.info("estimating scale")

        #estimator_conf = self.configuration['scale_estimator']

        current_prediction = frame.predicted_position
        scaled_predictions = []
        #for i in range(estimator_conf['number_candidates']):
        for i in range(33):
            scaled_width = np.random.normal(loc=current_prediction.width, scale=1.0)
            scaled_height = np.random.normal(loc=current_prediction.height, scale=1.0)

            scaled_box = Rect(frame.predicted_position.x, frame.predicted_position.y, scaled_width, scaled_height)
            scaled_predictions.append(scaled_box)

        logger.info("created %s scaled positions", len(scaled_predictions))
        logger.info(scaled_predictions)

        #append current prediction aswell, so that the array can be evaluated and that its possible,
        #that no changes in scale are necessary.
        scaled_predictions.append(current_prediction)

        return scaled_predictions


    def create_fourier_rep(self, frame=None):
        logger.info("creating fourier representation")
        #logger.info("frame.capture_iamge %s", frame.capture_image)
        #img = cv2.imread(frame.capture_image)
        return frame

    def append_to_history(self, frame):
        self.box_history.append([frame.number, frame.predicted_position.x, frame.predicted_position.y, frame.predicted_position.width, frame.predicted_position.height])

        logger.info("Box at frame{0}: x: {1}, y: {2}, width: {3}, height: {4}".format(frame.number, frame.predicted_position.x,
                                                                                      frame.predicted_position.y,
                                                                                      frame.predicted_position.width,
                                                                                      frame.predicted_position.height))




