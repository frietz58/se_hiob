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
        self.scale_step = None
        self.max_scale_change = None

        # run time
        self.frame = None
        self.scale_factors = None
        self.unique_candidates = []
        self.unique_factors = []

    def configure(self, configuration):
        """
        reads the values from the configuration
        :param configuration: the configuration file
        """

        self.number_scales = configuration['number_scales']
        self.inner_punish_threshold = configuration['inner_punish_threshold']
        self.inner_punish_factor = configuration['inner_punish_factor']
        self.outer_punish_threshold = configuration['outer_punish_threshold']
        self.scale_step = configuration['scale_factor']
        self.max_scale_change = configuration['max_scale_difference']

    def generate_scaled_candidates(self, frame):
        """
        :param frame: the current frame in which the best position has already been calculated
        :return: a list of scaled variations of the best predicted position
        """

        self.frame = frame
        scaled_predictions = []

        ss = np.arange(1, self.number_scales + 1)
        self.scale_factors = np.power(self.scale_step, (np.ceil(self.number_scales / 2) - ss))

        # Generate n scaled candidates
        for i in range(self.number_scales):

            scale_factor = self.scale_factors[i]
            scaled_box = Rect(
                frame.predicted_position.x,
                frame.predicted_position.y,
                frame.predicted_position.w * scale_factor,
                frame.predicted_position.h * scale_factor)

            scaled_predictions.append(scaled_box)

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

        # TODO maybe include scale window here? so none 1 factor candidates values get punished?

        # find unique candidates and calculate corresponding average scale factors (because low resolution of feature
        # mask, different different scale lvls will have same result, causing np.argmax to return the first match,
        # distorting the scale prediction)
        self.get_unique_candidates(evaluated_candidates)

        # recover the scale change factor
        scale_change = self.unique_factors[np.argmax(self.unique_candidates)]

        # make sure the area doesnt change too much
        limited_factor = self.limit_scale_change(scale_change, keep_original=True)

        # return Rect(scaled_candidates[scale_change])
        new_w = round(self.frame.predicted_position.w * limited_factor)
        new_h = round(self.frame.predicted_position.h * limited_factor)

        return Rect(self.frame.predicted_position.x, self.frame.predicted_position.y, new_w, new_h)

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

        return quality_of_candidate

    def limit_scale_change(self, factor, keep_original=False):
        """
        limits the scale change between two frames to a configured threshold
        :param factor: the predicted scale change factor for the target size at factor 1
        :param keep_original: for debugging, can be passed with the function call so that the predicted factor will
            always be used
        :return: if factors out of threshold, threshold factor, otherwise normal factor
        """

        # make sure area didn't change too much, correcting scale factor
        if factor > 1 + self.max_scale_change:
            output_factor = 1 + self.max_scale_change
            logger.info("predicted scale change was {0}, reduced it to {1}".format(factor, output_factor))
        elif factor < 1 - self.max_scale_change:
            output_factor = 1 - self.max_scale_change
            logger.info("predicted scale change was {0}, increased it  to {1}".format(factor, output_factor))
        else:
            output_factor = factor
            logger.info("predicted scale change was {0}".format(factor))

        # for debugging
        if keep_original:
            logger.info("final output factor: {0}".format(factor))
            output_factor = factor

        return output_factor

    def get_unique_candidates(self, candidates):
        """
        because the feature map has a low resolution, candidates at different scale factors might end up with the exact
        same quality score, which is calculated with the pixel values on the feature map. Therefor it is necessary to
        find the distinct candidates and calculate the corresponding scale factors, otherwise working with np.argmax
        will return the first match, which would in almost every case return a scale factor that is actually to high
        :param candidates: the non-distinct candidate list
        :return: sets distinct candidates corresponding scale factors
        """

        # reset class variables
        self.unique_candidates = []
        self.unique_factors = []

        # find the candidates with different pixel values and qualities
        unique_candidates = np.unique(candidates)

        # find the indices of the same candidates and get average scale factor for the unique candidate
        for candidate in unique_candidates:
            indices = np.where(candidates == candidate)

            avg_scale_factor = np.sum(self.scale_factors[indices]) / indices[0].__len__()
            self.unique_candidates.append(candidate)
            self.unique_factors.append(avg_scale_factor)
        # TODO doesn't contain factor 1 now...
        logger.info(self.unique_factors)
