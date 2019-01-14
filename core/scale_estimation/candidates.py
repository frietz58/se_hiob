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
        if np.mod(self.number_scales, 2) == 0:
            raise ValueError("Number of Scales needs to be odd!")
        self.inner_punish_threshold = configuration['inner_punish_threshold']
        self.inner_punish_factor = configuration['inner_punish_factor']
        self.outer_punish_threshold = configuration['outer_punish_threshold']
        self.scale_step = configuration['scale_factor']
        self.max_scale_change = configuration['max_scale_difference']

    def generate_scaled_candidates(self, frame):
        """
        generates the candidates based on the predicted position but at different scale levels
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
        evaluates the candidates of different sizes. this is done by getting the values on the feature mask for each
        candidate and punishing the rating of the candidate for not containing high values (aka strong probability
        of feature belonging to target) and for containing small values.  If the overall prediction has low values, the
        scale wont be changed. Each candidate also gets punished more for having a factor further away from 1.
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
            try:
                evaluated_candidates.append(self.rate_scaled_candidate(
                    single_sum,
                    candidate,
                    mask_scale_factor,
                    feature_mask))
            except ValueError:
                # handcraft the results, so that the scale will not change when the prediction is bad
                # number scales is always odd, therefor we now that factor 1 is in the middle
                if evaluated_candidates.__len__() == (self.number_scales - 1) / 2:
                    evaluated_candidates.append(1)
                # in every other case when we are not at factor 1, append 0. Like this, when multiplied with the scale
                # window, 1 will be the best factor and scale wont be changed.
                else:
                    evaluated_candidates.append(0)

        # hanning scale window to punish punish factors there further they are away from  1
        scale_window = np.hanning(self.number_scales)
        punished_candidates = np.multiply(evaluated_candidates, scale_window)

        # find unique candidates and calculate corresponding average scale factors (because low resolution of feature
        # mask, different different scale lvls will have same result, causing np.argmax to return the first match,
        # distorting the scale prediction)
        # self.get_unique_candidates(evaluated_candidates)

        # recover the scale change factor
        scale_change = self.scale_factors[np.argmax(punished_candidates)]

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

        # check highest value on feature map, calculate threshold values accordingly
        # (its possible that every prediction value is smaller than value for the inner threshold, in which case every
        # rating becomes 0, when everything in the quality output is 0, np.argmax will return 0 aswell, thus the
        # scale prediction fails
        max_val = np.amax(feature_mask)
        if max_val < self.inner_punish_threshold:
            raise ValueError('Highest probability is smaller than threshold')

        # punish candidate for containing values smaller than threshold
        # get the candidate on representation on the feature mask
        candidate_on_mask = feature_mask[
                        round(candidate.top / mask_scale_factor[1]):
                        round((candidate.bottom - 1) / mask_scale_factor[1]),
                        round(candidate.left / mask_scale_factor[0]):
                        round((candidate.right - 1) / mask_scale_factor[0])]

        # get the filter that checks where the condition applies
        mask_filter = candidate_on_mask < self.inner_punish_threshold

        # apply filter to get feature values
        feature_values = candidate_on_mask[mask_filter]

        # sum values up to get inner punish score
        inner_punish_sum = np.sum(feature_values)

        # calculate a score that punishes the candidate for not containing values bigger than threshold
        # filter for all values that are bigger than threshold
        outer_mask_filter = feature_mask > self.outer_punish_threshold

        # get the values of the filter
        outer_values = feature_mask[outer_mask_filter]

        # find the values that are bigger but within the candidate (we dont want to punish those)
        on_candidate_filter = candidate_on_mask > self.outer_punish_threshold

        # get the values
        on_candidate_values = candidate_on_mask[on_candidate_filter]

        # sum both values up and subtract the values that are bigger but within the candidate
        outer_punish_sum = np.sum(outer_values) - np.sum(on_candidate_values)

        # Evaluate the candidate
        quality_of_candidate = candidate_sum - (inner_punish_sum + outer_punish_sum)

        if quality_of_candidate == 0:
            raise ValueError("Quality of candidate is 0, this should not happen")

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
