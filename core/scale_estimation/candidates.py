import logging
import numpy as np
from ..Rect import Rect
import os
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class CandidateApproach:

    def __init__(self):
        # configuration
        self.number_scales = None
        self.inner_punish_threshold = None
        self.outer_punish_threshold = None
        self.scale_step = None
        self.max_scale_change = None
        self.scale_window_step_size = None
        self.change_aspect_ration = None
        self.adjust_max_scale_diff = None
        self.adjust_max_scale_diff_after = None
        self.update_strategy = None

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
        :param configuration: the configuration dict
        """

        self.number_scales = configuration['c_number_scales']
        if np.mod(self.number_scales, 2) == 0:
            raise ValueError("Number of Scales needs to be odd!")
        self.inner_punish_threshold = configuration['inner_punish_threshold']
        self.outer_punish_threshold = configuration['outer_punish_threshold']
        self.scale_step = configuration['c_scale_factor']
        self.max_scale_change = configuration['max_scale_difference']
        self.scale_window_step_size = configuration['scale_window_step_size']
        self.change_aspect_ration = configuration['c_change_aspect_ratio']
        self.adjust_max_scale_diff = configuration['adjust_max_scale_diff']
        self.adjust_max_scale_diff_after = configuration['adjust_max_scale_diff_after']
        self.update_strategy = configuration['update_strategy']

        self.calc_manual_scale_window(step_size=self.scale_window_step_size)
        self.hanning_scale_window = np.hanning(self.number_scales)

    def handle_initial_frame(self, frame):
        """
        Special case where we set some additional values that are not available at the time when we load the
        configuration. Those values must be set so that we can actually process the oth frame like any other frame.
        :param frame:
        :return:
        """

        # reset values from previous sequence
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

        self.frame = frame
        self.base_target_size = [self.frame.ground_truth.w, self.frame.ground_truth.h]
        self.current_scale_factor = 1

        ss = np.arange(1, self.number_scales + 1)
        self.scale_factors = np.power(self.scale_step, (np.ceil(self.number_scales / 2) - ss))

        self.calc_manual_scale_window(step_size=self.scale_window_step_size)
        self.hanning_scale_window = np.hanning(self.number_scales)

    def generate_scaled_candidates(self, frame, tracking):
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

            scaled_predictions = []

            # Generate n scaled candidates
            for i in range(self.number_scales):
                scale_factor = scale_factors[i]

                # calc new width and height
                new_w = round(self.base_target_size[0] * scale_factor)
                new_h = round(self.base_target_size[1] * scale_factor)

                # adjust x and y pos so that the box remains centered when height/width change
                old_x = self.frame.predicted_position.center[0]
                old_y = self.frame.predicted_position.center[1]
                new_x = int(old_x - np.rint(new_w / 2))
                new_y = int(old_y - np.rint(new_h / 2))

                scaled_box = Rect(
                    new_x,
                    new_y,
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

                new_x = int(old_x - np.rint(new_w / 2))
                new_y = int(old_y - np.rint(new_h / 2))

                # handle out of bounds
                new_x, new_y, new_w, new_h = self.check_oob(new_x, new_y, new_w, new_h)

                scaled_width_box = Rect(new_x,
                                        self.frame.predicted_position.y,
                                        new_w,
                                        self.frame.predicted_position.h)

                scaled_predictions[0].append(scaled_width_box)

                scaled_height_box = Rect(self.frame.predicted_position.x,
                                         new_y,
                                         self.frame.predicted_position.w,
                                         new_h)

                scaled_predictions[1].append(scaled_height_box)

        # for creating images...
        # self.create_image_with_generated_candidates(scaled_predictions, tracking, frame)

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
            evaluated_candidates = [[], []]

            # candidates that have only scaled width
            for width_candidate in scaled_candidates[0]:
                evaluated_candidates[0].append(self.evaluation_helper(candidate=width_candidate,
                                                                      feature_mask=feature_mask,
                                                                      mask_scale_factor=mask_scale_factor,
                                                                      prev_list=evaluated_candidates[0]))

            # candidates that have only scaled height
            for height_candidate in scaled_candidates[1]:
                evaluated_candidates[1].append(self.evaluation_helper(candidate=height_candidate,
                                                                      feature_mask=feature_mask,
                                                                      mask_scale_factor=mask_scale_factor,
                                                                      prev_list=evaluated_candidates[1]))

        # if keep aspect ration the same
        else:
            evaluated_candidates = []

            for candidate in scaled_candidates:
                evaluated_candidates.append(self.evaluation_helper(candidate=candidate,
                                                                   feature_mask=feature_mask,
                                                                   mask_scale_factor=mask_scale_factor,
                                                                   prev_list=evaluated_candidates))

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

        new_x = int(old_x - np.rint(new_w / 2))
        new_y = int(old_y - np.rint(new_h / 2))

        return Rect(new_x, new_y, new_w, new_h)

    def evaluation_helper(self, candidate, feature_mask, mask_scale_factor, prev_list):
        """
        Only used so that we don't have to reuse the code in the body of this method multiply times --> DRY!
        :param candidate: the candidate to evaluate
        :param feature_mask: the CNN feature mask
        :param mask_scale_factor: the factor with which the feature mask has been scaled to correspond to the actual ROI
            size, the cnn output is configurable
        :param prev_list: The list of the previously obtained ratings, needed for the case where we handcraft the
            results
        :return: a rating for the candidate
        """
        try:
            evaluated_candidate = self.rate_scaled_candidate(
                candidate=candidate,
                feature_mask=feature_mask,
                mask_scale_factor=mask_scale_factor)

        # it can happen that all values on the feature map are < than the punish thresholds, in which case the
        # calculation fails (for every candidate). Thus, we dont want to change the scale. Thus, every candidate gets
        # a punishment score of 2, except the candidate at scale 1, which gets a score of 1 and as a consequence, scale
        # factor 1 will be selected. Theoretically, this shouldn't even be needed anymore, because of the scale window.
        except ValueError:
            if len(prev_list) == (self.number_scales - 1) / 2:
                evaluated_candidate = 1
            else:
                evaluated_candidate = 2

        return evaluated_candidate

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
        # inner_punish_sum = np.sum(feature_values)  # old calc, had been used in thesis

        # this has shown to work better, we use 1-val, because these values are SMALLER than threshold, thus wont
        # produce a big sum and have a small impact if we just sum those up...
        inner_punish_sum = np.sum([1 - feature_val for feature_val in feature_values])

        # calculate a score that punishes the candidate for not containing values bigger than threshold
        # filter for all values that are bigger than threshold
        outer_mask_filter = feature_mask > self.outer_punish_threshold

        # get the values of the filter
        outer_values = feature_mask[outer_mask_filter]

        # find the values that are bigger but within the candidate (we don't want to punish those)
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

        # if we use limited, we dont know how many frame can have passed on limiting doesnt make any sense
        if self.update_strategy == "limited":
            return new

        # see if the factor has been limited x times in a row, indicating a currently strong change
        if self.adjust_max_scale_diff:
            if len(self.limited_growth_times) >= self.adjust_max_scale_diff_after:
                self.max_scale_change += 0.02
                logger.info("scale_factor has been reduced too often, new limit is {0}".format(self.max_scale_change))
            elif len(self.limited_shrink_times) >= self.adjust_max_scale_diff_after:
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
        curve = [0] * int(self.number_scales)

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

    def check_oob(self, new_x, new_y, new_w, new_h):
        """
        Checks whether the generate scale of the object is bigger than the frame, in which case it is set to be as big
        as possible without.
        :param new_x: scaled x valuue
        :param new_y: scaled y valuue
        :param new_w: scaled w valuue
        :param new_h: scaled h valuue
        :return: The if necessary adjusted values
        """

        if new_x <= 1:
            new_x = 1

        if new_y <= 1:
            new_y = 1

        if new_x >= self.frame.size[1]:
            new_x = self.frame.size[1]

        if new_y >= self.frame.size[2]:
            new_y = self.frame.size[2]

        if new_w <= 1:
            new_w = 1

        if new_h <= 1:
            new_h = 1

        if new_w >= self.frame.size[1]:
            new_w = self.frame.size[1]

        if new_h >= self.frame.size[2]:
            new_h = self.frame.size[2]

        return new_x, new_y, new_w, new_h

    def create_image_with_generated_candidates(self, scaled_predictions, tracking, frame):
        """
        Incredibly ugly code but idc, used this to generate some images for my thesis
        :param scaled_predictions: the scaled candidates
        :param tracking: the tracking object
        :param frame: the current frame
        """
        image_dir = os.path.join("images", tracking.sample.name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        conolidator_img1 = tracking.get_frame_consolidation_images(decorations=False)['single']
        conolidator_img2 = tracking.get_frame_consolidation_images(decorations=False)['single']
        conolidator_img3 = tracking.get_frame_consolidation_images(decorations=False)['single']

        sroi_img1 = tracking.get_frame_sroi_image(decorations=False)
        sroi_img2 = tracking.get_frame_sroi_image(decorations=False)

        capture_img1 = tracking.get_frame_capture_image(decorations=False)
        capture_img2 = tracking.get_frame_capture_image(decorations=False)

        consolidator_draw1 = ImageDraw.Draw(conolidator_img1)
        consolidator_draw2 = ImageDraw.Draw(conolidator_img2)
        consolidator_draw3 = ImageDraw.Draw(conolidator_img3)

        sroi_draw1 = ImageDraw.Draw(sroi_img1)
        sroi_draw2 = ImageDraw.Draw(sroi_img2)

        capture_draw1 = ImageDraw.Draw(capture_img1)
        capture_draw2 = ImageDraw.Draw(capture_img2)

        if not self.change_aspect_ration:
            for i, rect in enumerate(scaled_predictions):

                resize = (scaled_predictions[int((len(scaled_predictions) - 1) / 2 + 1)]._w, scaled_predictions[
                    int((len(scaled_predictions) - 1) / 2 + 1)]._h)

                # if i % 5 == 0 or i == 0 or i == 32:
                if i == (len(scaled_predictions) - 1) / 2 + 1:
                    # sroi image
                    sroi_pos = tracking.capture_to_sroi(rect, frame.roi).inner
                    sroi_gt = tracking.capture_to_sroi(frame.ground_truth, frame.roi).inner
                    # sroi_draw1.rectangle(sroi_pos, None, tracking.colours['candidate'])
                    # sroi_draw1.rectangle(sroi_gt, None, tracking.colours['ground_truth'])

                    # capture image
                    cap_pos = rect.inner
                    # capture_draw1.rectangle(cap_pos, None, tracking.colours['candidate'])

                    # consolidator image
                    cons_pos = tracking.capture_to_mask(rect, frame.roi).inner
                    consolidator_draw3.rectangle(cons_pos, None, tracking.colours['roi'])

                    # scaled_patch
                    img_patch = capture_img1.crop((rect._x, rect._y, rect._x + rect._w, rect._y + rect._h))
                    img_patch.save(
                        os.path.join(image_dir,
                                     "{}-17_patch.png".format(tracking.get_current_frame_number())))

                elif i == 0:
                    # consolidator image
                    cons_pos = tracking.capture_to_mask(
                        rect, frame.roi).inner
                    consolidator_draw1.rectangle(cons_pos, None, tracking.colours['roi'])

                    # sroi image
                    sroi_pos = tracking.capture_to_sroi(rect, frame.roi).inner
                    # sroi_draw1.rectangle(sroi_pos, None, tracking.colours['candidate'])

                    # scaled_patch
                    img_patch = capture_img1.crop((rect._x, rect._y, rect._x + rect._w, rect._y + rect._h))
                    img_patch = img_patch.resize(resize)
                    img_patch.save(
                        os.path.join(image_dir,
                                     "{}-0_patch.png".format(tracking.get_current_frame_number())))

                elif i == 32:
                    # consolidator image
                    cons_pos = tracking.capture_to_mask(
                        rect, frame.roi).inner
                    consolidator_draw2.rectangle(cons_pos, None, tracking.colours['roi'])

                    # sroi image
                    sroi_pos = tracking.capture_to_sroi(rect, frame.roi).inner
                    # sroi_draw1.rectangle(sroi_pos, None, tracking.colours['candidate'])

                    # scaled_patch
                    img_patch = capture_img1.crop((rect._x, rect._y, rect._x + rect._w, rect._y + rect._h))
                    img_patch = img_patch.resize(resize)
                    img_patch.save(
                        os.path.join(image_dir,
                                     "{}-32_patch.png".format(tracking.get_current_frame_number())))

                else:
                    # sroi image
                    sroi_pos = tracking.capture_to_sroi(rect, frame.roi).inner
                    # sroi_draw1.rectangle(sroi_pos, None, tracking.colours['candidate'])

            conolidator_img1.save(
                os.path.join(image_dir, "{}-consolidator_0th_cand.png".format(tracking.get_current_frame_number())))
            conolidator_img2.save(
                os.path.join(image_dir, "{}-consolidator_32th_cand.png".format(tracking.get_current_frame_number())))
            conolidator_img3.save(
                os.path.join(image_dir, "{}-consolidator_17th_cand.png".format(tracking.get_current_frame_number())))

            sroi_img1.save(
                os.path.join(image_dir, "{}-sroi_candidates.png".format(tracking.get_current_frame_number())))
            capture_img1.save(
                os.path.join(image_dir, "{}-capture_candidates.png".format(tracking.get_current_frame_number())))

            # if tracking.get_current_frame_number() != 1:
            #     target_mask_img.save(os.path.join(image_dir, "{}-target_mask.png".format(tracking.get_current_frame_number())))

        elif self.change_aspect_ration:
            for i in range(0, np.shape(scaled_predictions)[1]):
                if i % 5 == 0 or i == 0 or i == 32:
                    cap_pos_width = scaled_predictions[0][i].inner
                    capture_draw1.rectangle(cap_pos_width, None, tracking.colours['candidate'])

                    cap_pos_height = scaled_predictions[1][i].inner
                    capture_draw2.rectangle(cap_pos_height, None, tracking.colours['candidate'])

                    sroi_pos_width = tracking.capture_to_sroi(scaled_predictions[0][i], frame.roi).inner
                    sroi_draw1.rectangle(sroi_pos_width, None, tracking.colours['candidate'])

                    sroi_pos_width = tracking.capture_to_sroi(scaled_predictions[1][i], frame.roi).inner
                    sroi_draw2.rectangle(sroi_pos_width, None, tracking.colours['candidate'])

            capture_img1.save(
                os.path.join(image_dir, "{}-capture_width_candidates.png".format(tracking.get_current_frame_number())))
            capture_img2.save(
                os.path.join(image_dir, "{}-capture_height_candidates.png".format(tracking.get_current_frame_number())))

            sroi_img1.save(
                os.path.join(image_dir, "{}-sroi_width_candidates.png".format(tracking.get_current_frame_number())))
            sroi_img2.save(
                os.path.join(image_dir, "{}-sroi_height_candidates.png".format(tracking.get_current_frame_number())))
