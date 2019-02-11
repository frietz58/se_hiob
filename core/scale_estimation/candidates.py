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
        self.scale_window_step_size = None
        self.change_aspect_ration = None

        # run time
        self.frame = None
        self.scale_factors = None
        self.manual_scale_window = ()
        self.hanning_scale_window = None
        self.base_target_size = None
        self.current_scale_factor = 1
        self.current_width_factor = 1
        self.current_height_factor = 1
        self.width_scale_factors = []
        self.height_scale_factors = []
        self.limited_growth_times = []
        self.limited_shrink_times = []

    def configure(self, configuration):
        """
        reads the values from the configuration
        :param configuration: the configuration file
        """

        self.number_scales = configuration['c_number_scales']
        if np.mod(self.number_scales, 2) == 0:
            raise ValueError("Number of Scales needs to be odd!")
        self.inner_punish_threshold = configuration['inner_punish_threshold']
        self.inner_punish_factor = configuration['inner_punish_factor']
        self.outer_punish_threshold = configuration['outer_punish_threshold']
        self.scale_step = configuration['scale_factor']
        self.max_scale_change = configuration['max_scale_difference']
        self.scale_window_step_size = configuration['scale_window_step_size']
        self.change_aspect_ration = configuration['c_change_aspect_ratio']

        self.calc_manual_scale_window(step_size=self.scale_window_step_size)
        self.hanning_scale_window = np.hanning(self.number_scales)

    def handle_initial_frame(self, frame):
        self.frame = frame
        self.base_target_size = [self.frame.ground_truth.w, self.frame.ground_truth.h]
        self.current_scale_factor = 1

        ss = np.arange(1, self.number_scales + 1)
        self.scale_factors = np.power(self.scale_step, (np.ceil(self.number_scales / 2) - ss))

    def generate_scaled_candidates(self, frame):
        """
        generates the candidates based on the predicted position but at different scale levels. Depending on the
        settings in the configuration, either candidates will only be scaled, or will be scaled along x or y axis
        separately. In the later case the best values for x and y axis will be found, resulting in a prediction that
        can adapt to changes in the aspect ration.
        :param frame: the current frame in which the best position has already been calculated
        :return: a list of scaled variations of the best predicted position
        """

        self.frame = frame

        # only change scale, keep aspect ration through tracking
        if not self.change_aspect_ration:
            # calculate the current scale factors
            scale_factors = np.multiply(self.scale_factors, self.current_scale_factor)

            # init scaled predictions 2d array. 0 for scaled width, 1 for scaled height
            scaled_predictions = []

            # Generate n scaled candidates
            for i in range(self.number_scales):
                scale_factor = scale_factors[i]
                scaled_box = Rect(
                    frame.predicted_position.x,
                    frame.predicted_position.y,
                    np.floor(self.base_target_size[0] * scale_factor),
                    np.floor(self.base_target_size[1] * scale_factor))

                scaled_predictions.append(scaled_box)

        # create patches with flexible aspect ratio
        elif self.change_aspect_ration:

            # calculate the current scale factors
            self.width_scale_factors = np.multiply(self.scale_factors, self.current_width_factor)
            self.height_scale_factors = np.multiply(self.scale_factors, self.current_height_factor)

            # init scaled predictions 2d array. 0 for scaled width, 1 for scaled height
            scaled_predictions = [[], []]

            # Generate 2 * n scaled candidates
            for i in range(self.number_scales):
                width_scale_factor = self.width_scale_factors[i]
                height_scale_factor = self.height_scale_factors[i]

                # calc new width and height
                new_w = round(self.base_target_size[0] * width_scale_factor)
                new_h = round(self.base_target_size[1] * height_scale_factor)

                # adjust x and y pos so that the box remains centered when height/width change
                old_x = self.frame.predicted_position.center[0]
                old_y = self.frame.predicted_position.center[1]

                # TODO check oob
                new_x = int(old_x - np.rint(new_w / 2))
                new_y = int(old_y - np.rint(new_h / 2))

                scaled_width_box = Rect(
                    new_x,
                    old_y,
                    new_w,
                    self.frame.predicted_position.h)
                scaled_predictions[0].append(scaled_width_box)

                scaled_height_box = Rect(
                    old_x,
                    new_y,
                    self.frame.predicted_position.w,
                    new_h)
                scaled_predictions[1].append(scaled_height_box)

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
        # if changes aspect ration
        if self.change_aspect_ration:

            # Evaluate each candidate based on its size
            evaluated_candidates = [[], []]

            # get pixel value of candidates at scaled lvl 1
            base_candidate_rect = scaled_candidates[0][16]
            base_candidate_val = np.sum([feature_mask[
                                         round(base_candidate_rect.top / mask_scale_factor[1]):
                                         round((base_candidate_rect.bottom - 1) / mask_scale_factor[1]),
                                         round(base_candidate_rect.left / mask_scale_factor[0]):
                                         round((base_candidate_rect.right - 1) / mask_scale_factor[0])]])

            for width_candidate in scaled_candidates[0]:
                # TODO dont have this code block three times...
                try:
                    evaluated_candidates[0].append(self.rate_scaled_candidate(
                        candidate=width_candidate,
                        feature_mask=feature_mask,
                        mask_scale_factor=mask_scale_factor))
                # should not happen because when only get here when the quality of the frame is higher than threshold
                except ValueError:
                    # handcraft the results, so that the scale will not change when the prediction is bad
                    # number scales is always odd, therefor we now that factor 1 is in the middle
                    if evaluated_candidates[0].__len__() == (self.number_scales - 1) / 2:
                        evaluated_candidates[0].append(0)
                        logger.info("Probability Values on Heatmap too low, returning unchanged size")
                    else:
                        evaluated_candidates[0].append(1)

            for height_candidate in scaled_candidates[1]:
                try:
                    evaluated_candidates[1].append(self.rate_scaled_candidate(
                        candidate=height_candidate,
                        feature_mask=feature_mask,
                        mask_scale_factor=mask_scale_factor))
                # should not happen because when only get here when the quality of the frame is higher than threshold
                except ValueError:
                    # handcraft the results, so that the scale will not change when the prediction is bad
                    # number scales is always odd, therefor we now that factor 1 is in the middle
                    if evaluated_candidates[1].__len__() == (self.number_scales - 1) / 2:
                        evaluated_candidates[1].append(0)
                        logger.info("Probability Values on Heatmap too low, returning unchanged size")
                    else:
                        evaluated_candidates[1].append(1)

        # if keep aspect ration the same
        else:
            base_candidate_rect = scaled_candidates[16]
            base_candidate_val = np.sum([feature_mask[
                                  round(base_candidate_rect.top / mask_scale_factor[1]):
                                  round((base_candidate_rect.bottom - 1) / mask_scale_factor[1]),
                                  round(base_candidate_rect.left / mask_scale_factor[0]):
                                  round((base_candidate_rect.right - 1) / mask_scale_factor[0])]])

            # Evaluate each candidate based on its size
            evaluated_candidates = []

            for candidate in scaled_candidates:
                try:
                    evaluated_candidates.append(self.rate_scaled_candidate(
                        candidate=candidate,
                        mask_scale_factor=mask_scale_factor,
                        feature_mask=feature_mask))
                # should not happen because when only get here when the quality of the frame is higher than threshold
                except ValueError:
                    # handcraft the results, so that the scale will not change when the prediction is bad
                    # number scales is always odd, therefor we now that factor 1 is in the middle
                    if evaluated_candidates.__len__() == (self.number_scales - 1) / 2:
                        evaluated_candidates.append(0)
                        logger.info("Probability Values on Heatmap too low, returning unchanged size")
                    else:
                        evaluated_candidates.append(1)

        # use either hanning or manual scale window to punish the candidates depending of their factor
        # IMPORTANT also makes them unique, which is import for the np.argmax later
        if self.change_aspect_ration:
            punished_width_candidate_scores = np.multiply(evaluated_candidates[0], self.manual_scale_window)
            punished_height_candidate_scores = np.multiply(evaluated_candidates[1], self.manual_scale_window)

            limited_width_factor = self.limit_scale_change(
                old=self.current_width_factor,
                new=self.current_width_factor * self.scale_factors[np.argmin(punished_width_candidate_scores)],
                max_change_percentage=self.max_scale_change,
                axis='width'
            )

            limited_height_factor = self.limit_scale_change(
                old=self.current_height_factor,
                new=self.current_height_factor * self.scale_factors[np.argmin(punished_height_candidate_scores)],
                max_change_percentage=self.max_scale_change,
                axis='height'
            )

            # update the aspect raio
            self.current_width_factor = limited_width_factor
            self.current_height_factor = limited_height_factor

            # calc new width and height
            new_w = round(self.base_target_size[0] * self.current_width_factor)
            new_h = round(self.base_target_size[1] * self.current_height_factor)

        else:

            punished_candidates = np.multiply(evaluated_candidates, self.manual_scale_window)

            # correct scale factor if it changed too much
            limited_factor = self.limit_scale_change(
                old=self.current_scale_factor,
                new=self.current_scale_factor * self.scale_factors[np.argmin(punished_candidates)],
                max_change_percentage=self.max_scale_change,
                axis='scale factor'
            )

            # update the scale
            self.current_scale_factor = limited_factor

            # calc new width and height
            new_w = round(self.base_target_size[0] * self.current_scale_factor)
            new_h = round(self.base_target_size[1] * self.current_scale_factor)

        # adjust x and y pos so that the box remains centered when height/width change
        old_x = self.frame.predicted_position.center[0]
        old_y = self.frame.predicted_position.center[1]

        new_x = int(old_x - np.rint(new_w/2))
        new_y = int(old_y - np.rint(new_h/2))

        return Rect(new_x, new_y, new_w, new_h)

    def rate_scaled_candidate(self, candidate, mask_scale_factor, feature_mask):
        """
        :param candidate: the current candidate
        :param mask_scale_factor: the factor with which the feature mask has been scaled to correspond to the actual ROI
        size, the cnn output is 48x48
        :param feature_mask: the consolidated feature mask containing pixel values for how likely they belong to the
        :return: the quality of the candidate based on its size
        """

        # check highest value on feature map, calculate threshold values accordingly
        # (its possible that every prediction value is smaller than value for the inner threshold, in which case every
        # rating becomes 0, when everything in the quality output is 0, np.argmax will return 0 aswell, thus the
        # scale prediction fails
        #TODO also make this use the max of base candidate not entire feature mask...
        max_val = np.amax(feature_mask)
        if max_val < self.inner_punish_threshold:
            # TODO make custom error class
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
        quality_of_candidate = inner_punish_sum + outer_punish_sum

        return quality_of_candidate

    def limit_scale_change(self, old, new, max_change_percentage, axis=None):
        """
        limits the scale change between two frames to a configured threshold
        :param old: the current scale factor still from the previous frames
        :param new: the new predicted scale factor, which is to be limited
        :param max_change_percentage: the percentage value of which the scale is allowed to change each frame
        :param axis: only for logger message: height width of both
        :return: if factors out of threshold, threshold factor, otherwise normal factor
        """

        # see if the factor has been limited 3 times in a row, indicating a currently strong change
        if len(self.limited_growth_times) > 2:
            self.max_scale_change += 0.02
            logger.info("scale_factor has been reduced too often, new limit is {0}".format(self.max_scale_change))
        elif len(self.limited_shrink_times) > 2:
            self.max_scale_change += 0.02
            logger.info("scale_factor has been increased too often, new limit is {0}".format(self.max_scale_change))

        # make sure area didn't change too much, correcting scale factor
        if new > old + max_change_percentage:
            output_factor = old + max_change_percentage
            self.limited_growth_times.append(new)
            logger.info("predicted scale for {0} was {1}, reduced it to {2}".format(axis, new, output_factor))
        elif new < old - max_change_percentage:
            output_factor = old - max_change_percentage
            self.limited_growth_times.append(new)
            logger.info("predicted scale for {0} was {1}, increased it  to {2}".format(axis, new, output_factor))
        else:
            self.limited_growth_times = []
            self.limited_shrink_times = []
            output_factor = new
            logger.info("predicted scale for {0} is {1}".format(axis, output_factor))

        return output_factor

    def get_unique_candidates(self, candidates):
        """
        because the feature map has a low resolution, candidates at different scale factors might end up with the exact
        same quality score, which is calculated with the pixel values on the feature map. Therefor it is necessary to
        find the distinct candidates and calculate the corresponding scale factors, otherwise working with np.argmax
        will return the first match, which would in almost every case return a scale factor that is to high
        :param candidates: the non-distinct candidate list
        :return: sets distinct candidates corresponding scale factors
        """

        # find the candidates with different pixel values and qualities
        unique_candidates = []

        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)

        avg_factors = []

        # find the indices of the same candidates and get average scale factor for the unique candidate
        for candidate in unique_candidates:
            indices = np.where(candidates == candidate)

            avg_scale_factor = np.sum(self.scale_factors[indices]) / np.shape(indices)[1]
            avg_factors.append(avg_scale_factor)

        return unique_candidates, avg_factors

    def calc_manual_scale_window(self, step_size):
        """
        create a hanning like window except it only decrease from 1 at the center by the step_size in each direction
        :param step_size: the step to decrease from one per iteration
        :return: a list containing the hanning like curve
        """

        # initialize with zeros
        curve = [0] * self.number_scales

        if np.mod(len(curve), 2) == 0:
            raise ValueError('Number of Candidate (from configuration) needs to be odd!')

        # place 1 at the  (number scales is always odd)
        center = int((len(curve) - 1) / 2)
        curve[center] = 1

        # calculate the curve
        for i in range(1, int((len(curve) - 1) / 2) + 1):

            val = np.around(1 + (step_size * i), decimals=3)

            pos_loc = center + i
            neg_loc = center - i

            curve[pos_loc] = val
            curve[neg_loc] = val

        self.manual_scale_window = curve
