"""
Created on 2018-11-17

@author: Finn Rietz
"""

import logging
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
        self.sroi = None
        self.number_scales = None
        self.position = None
        self.tracker = None

    def setup(self, tracker=None):
        logger.info("setting up scale estimator")

        self.tracker = tracker

        logger.info("tracker.pursuer from estimator: %s", tracker.pursuer)

    def estimate_scale(self, frame):
        self.frame = frame

        logger.info("estimating scale")
        logger.info("testing frame %s", frame.predicted_position)

    def create_fourier_rep(self, frame=None):
        return frame




