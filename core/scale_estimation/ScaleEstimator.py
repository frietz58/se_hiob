import logging
from .matlab_dsst import DsstEstimator
from .custom_dsst import CustomDsst
from .candidates import CandidateApproach


logger = logging.getLogger(__name__)


class ScaleEstimator:
    """
    This is the main Coordinator, which makes calls to the specefic algorithm (Candidates or DSST) to update the
    scale on each frame.
    """

    def __init__(self):
        self.configuration = None
        self.econf = None
        self.custom_dsst = CustomDsst()
        self.candidate_approach = CandidateApproach()
        self.frame = None
        self.tracker = None
        self.sample = None
        self.passed_since_last_se = 0

        # configuration values
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
        """
        Sets tracker, configuration and sample on main ScaleEstimator
        :param tracker: the tracker object
        :param sample: the current sample
        """
        self.tracker = tracker
        self.configuration = tracker.configuration
        self.sample = sample
        self.set_up_modules()

    def configure(self, configuration):
        """
        Sets values from configuration faile on main ScaleEstimator
        :param configuration:
        :return:
        """
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
        """
        Sets values from configuration on both se algorithms
        """
        self.custom_dsst.configure(self.econf, img_files=self.sample.cv2_img_cache)
        self.candidate_approach.configure(self.econf)

    def estimate_scale(self, frame, feature_mask, mask_scale_factor, tracking):
        """
        This is called on every frame (if scale estimation is enabled) and depending on the configuration uses one of
        implemented approaches to estimate the scale of the object on the current frame.
        :param frame: the current frame in which the best position has already been calculated
        :param feature_mask: he consolidated feature mask containing pixel values for how likely they belong to the
        object
        :param mask_scale_factor: the factor with which the feature mask has been scaled to fit to the actual ROI
        size, the cnn output is configurable
        :return: the best rated candidate
        """

        self.frame = frame
        final_candidate = None

        # If scale estimation has been disabled in configuration, return unscaled bounding box
        if not self.use_scale_estimation:
            logger.info("Scale Estimation is disabled, returning unchanged prediction")
            return frame.predicted_position

        # continuous, update on every frame
        if self.update_strategy == "cont":
            final_candidate = self.execute_se_algorithm(frame, feature_mask, mask_scale_factor, tracking)

        # high_gain / limited combined
        elif self.update_strategy == "limited":
            if self.passed_since_last_se == 20:
                logger.info("20 frames passed without updating the scale, enforcing execution of SE")
                final_candidate = self.execute_se_algorithm(frame, feature_mask, mask_scale_factor, tracking)
                self.passed_since_last_se = 0

            elif (frame.prediction_quality >= self.dyn_min_se_treshold) \
                    and (frame.prediction_quality <= self.dyn_max_se_treshold):
                logger.info(("frame.prediction_quality = {0},"
                             " lies within window, executing SE").format(frame.prediction_quality))

                final_candidate = self.execute_se_algorithm(frame, feature_mask, mask_scale_factor, tracking)
                self.passed_since_last_se = 0

            else:
                logger.info(("frame.prediction_quality  = {0},"
                             " not within window, not executing se").format(frame.prediction_quality))

                final_candidate = frame.predicted_position
                self.passed_since_last_se += 1

        return final_candidate

    def handle_initial_frame(self, frame, sample):
        """
        The initial frame is a special case, hence both the candidates algorithm have dedicated methods
        :param sample: the current tracking sequence
        :param frame: the 0th frame
        """
        if not self.use_scale_estimation:
            return None

        self.frame = frame
        self.sample = sample

        if self.approach == "custom_dsst":
            self.custom_dsst.handle_initial_frame(frame=frame)

        elif self.approach == 'candidates':
            self.candidate_approach.handle_initial_frame(frame)

    def execute_se_algorithm(self, frame, feature_mask, mask_scale_factor, tracking):
        """
        Executes the actual scale estimation algorithm of one of the implemented algorithms
        :param frame: the current frame
        :param feature_mask: the CNN feature mask, used by the Candidates approach
        :param mask_scale_factor: the factor with which the feature mask has been scaled to fit to the actual ROI
        size, the cnn output is configurable
        :param tracking: the tracking object
        :return:
        """
        if self.approach == 'candidates':
            logger.info("starting scale estimation. Approach: Candidate Generation")
            scaled_candidates = self.candidate_approach.generate_scaled_candidates(frame, tracking)
            final_candidate = self.candidate_approach.evaluate_scaled_candidates(scaled_candidates,
                                                                                 feature_mask,
                                                                                 mask_scale_factor)

        elif self.approach == "custom_dsst":
            logger.info("starting scale estimation. Approach: DSST")
            final_candidate = self.custom_dsst.dsst(frame)

        else:
            logger.critical("No implementation for approach in configuration")
            final_candidate = None

        logger.info("finished scale estimation")

        return final_candidate
