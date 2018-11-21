"""
Created on 2018-11-17

@author: Finn Rietz
"""

import logging
import cv2
import transitions

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

    def setup(self, tracker=None):
        logger.info("setting up scale estimator")
        self.tracker = tracker


    def estimate_scale(self, frame):
        self.frame = frame

        logger.info("estimating scale")
        logger.info("testing frame %s", frame.predicted_position)

    def create_fourier_rep(self, frame=None):
        logger.info("creating fourier representation")
        #logger.info("frame.capture_iamge %s", frame.capture_image)
        #img = cv2.imread(frame.capture_image)
        #logger.info
        return frame

    def append_to_history(self, frame):
        self.box_history.append([frame.number, frame.predicted_position.__getattr__("width"), frame.predicted_position.__getattr__("height")])
        logger.info("Box at frame{0} has width {1} and height {2}".format(frame.number, frame.predicted_position.__getattr__("width"), frame.predicted_position.__getattr__("height")))




