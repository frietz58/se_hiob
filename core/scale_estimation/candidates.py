import logging
import numpy as np
from ..Rect import Rect

logger = logging.getLogger(__name__)


class CandidateApproach:

    def __init__(self):
        # configuration
        self.number_scales = None
        self.inner_punish_threshold = None
        self.inner_punish_factor = None
        self.outer_punish_threshold = None

        # run time
        self.frame = None

    def configure(self, configuration):
        self.number_scales = configuration['number_scales']
        self.inner_punish_threshold = configuration['inner_punish_threshold']
        self.inner_punish_factor = configuration['inner_punish_factor']
        self.outer_punish_threshold = configuration['outer_punish_threshold']

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

    def evaluate_scaled_candidates(self, scaled_candidates, feature_mask, mask_scale_factor):
        """
        :param scaled_candidates: the candidates based on the best position but scaled in widht and height
        :param feature_mask: the consolidated feature mask containing pixel values for how likely they belong to the
        object
        :param mask_scale_factor: the factor with which the feature mask has been scaled to correspond to the actual ROI
        size, the cnn output is 48x48
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
