import asyncio
import logging
from collections import OrderedDict
from datetime import datetime
import time

import matplotlib.cm
import matplotlib.pyplot as plt

import numpy as np
import transitions
from PIL import Image, ImageDraw

from .Frame import Frame
from .Rect import Rect
from .gauss import gen_gauss_mask
from .graph import figure_to_image
from . import evaluation


logger = logging.getLogger(__name__)

loop = asyncio.get_event_loop()


class TrackerModuleStates(object):
    """
    Place to store state of tracker modules within tracking.

    This class is basically just a wrapper around the persistent information
    for the individual modules of the tracker in between the individual frames.
    Every persistent data the tracking modules have to store must be put here,
    because the Tracker instance shall not store any information about an
    individual tracking process (that's what Tracking is for). The Tracker
    object should be able to process multiple trackings at the same time.
    """

    def __init__(self):
        self.roi_calculator = {}
        self.feature_extractor = {}
        self.feature_selector = {}
        self.consolidator = {}
        self.pursuer = {}


class Tracking(object):
    _LAST_SERIAL = 0

    states = [
        'created',
        'loading_sample',
        'sample_loaded',
        'selecting_features',
        'features_selected',
        'training_consolidator',
        'consolidator_trained',
        'begin_tracking',
        'loading_next_frame',
        'no_more_frames',
        'next_frame_loaded',
        'tracking_frame',
        'frame_tracked',
        'evaluating_frame',
        'frame_evaluated',
        'evaluating_tracking',
        'tracking_evaluated',
    ]

    transitions = [
        ['commence_load_sample', 'created', 'loading_sample'],
        ['complete_load_sample', 'loading_sample', 'sample_loaded'],
        ['commence_feature_selection', 'sample_loaded', 'selecting_features'],
        ['complete_feature_selection',
            'selecting_features', 'features_selected'],
        ['commence_consolidator_training',
            'features_selected', 'training_consolidator'],
        ['complete_consolidator_training',
            'training_consolidator', 'consolidator_trained'],
        ['commence_tracking', 'consolidator_trained', 'begin_tracking'],
        ['commence_loading_next_frame', [
            'begin_tracking', 'frame_evaluated'], 'loading_next_frame'],
        ['enter_no_more_frames', 'loading_next_frame', 'no_more_frames'],
        ['complete_loading_next_frame',
            'loading_next_frame', 'next_frame_loaded'],
        ['commence_tracking_frame', 'next_frame_loaded', 'tracking_frame'],
        ['complete_tracking_frame', 'tracking_frame', 'frame_tracked'],
        ['commence_evaluating_frame', 'frame_tracked', 'evaluating_frame'],
        ['complete_evaluating_frame', 'evaluating_frame', 'frame_evaluated'],
        ['commence_evaluate_tracking',
            ['no_more_frames', 'frame_evaluated'], 'evaluating_tracking'],
        ['complete_evaluate_tracking',
            'evaluating_tracking', 'tracking_evaluated'],
    ]

    def __init__(self, tracker, session):
        Tracking._LAST_SERIAL += 1
        self.serial = Tracking._LAST_SERIAL
        logger.info("Creating new Tracking#%d", self.serial)

        self.name = "tracking-%04d" % (self.serial)

        self.tracker = tracker
        self.configuration = tracker.configuration
        self.initial_frame = None
        self.current_frame = None
        self.total_frames = 0
        self.sample = None
        self.tracking_log = []

        # colour map function
        self.cmap = matplotlib.cm.get_cmap("hot")
        self.colours = {
            'roi': 'cyan',
            'ground_truth': 'green',
            'prediction': 'yellow',
            'candidate': 'magenta'
        }

        # size of capture image (determined on sample loading)
        self.capture_size = None
        # size of scaled ROI (= feature extractor input size)
        self.sroi_size = tracker.sroi_size
        # prediction/target mask size (=feature extractor / consolidator output
        # size)
        self.mask_size = tracker.mask_size
        #
        self.sroi_to_mask_ratio = (
            self.sroi_size[0] / self.mask_size[0], self.sroi_size[1] / self.mask_size[1])

        # create state machine for this tracking instance:
        self.machine = transitions.Machine(
            model=self,
            states=self.states, initial='created',
            transitions=self.transitions,
            name=self.__repr__(),
        )

        # timestamps
        self.ts_loaded = None
        self.ts_features_selected = None
        self.ts_consolidator_trained = None
        self.ts_tracking_completed = None

        self.pursuing_total_seconds = 0.0
        self.roi_calculation_total_seconds = 0.0
        self.sroi_generation_total_seconds = 0.0
        self.feature_extraction_total_seconds = 0.0
        self.feature_reduction_total_seconds = 0.0
        self.feature_consolidation_total_seconds = 0.0
        self.se_total_seconds = 0.0


        # updates
        self.updates_max_frames = 0
        self.updates_confidence = 0

        # object holding module_states of tracker modules:
        self.module_states = TrackerModuleStates()

        if tracker.configuration['ros_node']:
            from .RosPositionPublisher import RosPositionPublisher
            self.publisher = RosPositionPublisher(tracker.configuration['ros_node'])
        else:
            self.publisher = None
        self.session = session

    def get_total_frames(self):
        if not self.sample:
            return 0
        return self.sample.count_frames_processed()

    # beginning:
    async def load_sample(self, sample):
        self.commence_load_sample()
        # self.sample = hiob.data_set_old.load_sample(
        #    data_set, sample, config=self.tracker.configuration)
        self.sample = sample
        self.total_frames = self.sample.get_actual_frames()
        await self._load_initial_frame()
        self.capture_size = self.initial_frame.size
        # update name:
        self.name = "tracking-%04d-%s-%s" % (self.serial,
                                             sample.set_name, sample.name)
        #
        self.ts_loaded = datetime.now()
        self.complete_load_sample()

    async def _load_initial_frame(self):
        #self.current_frame_number = 1
        self.initial_frame = await self._get_next_sample_frame()
        # store initial position as position of previous frame:
        self.initial_frame.previous_position = self.sample.initial_position
        self.initial_frame.before_previous_position = self.sample.initial_position
        # we know the truth for this frame, use as prediction:
        self.initial_frame.predicted_position = self.sample.initial_position
        # for now, the initial frame is the current frame:
        self.current_frame = self.initial_frame

        self.tracker.pursuer.set_initial_position(self.initial_frame.previous_position)
        self.tracker.roi_calculator.set_initial_position(self.initial_frame.previous_position)
        self.tracker.estimator.handle_initial_frame(self.current_frame, self.sample)

    async def _get_next_sample_frame(self):
        frame = Frame(tracking=self, number=self.sample.current_frame_id + 1, size=None)
        frame.commence_capture()
        # frame.capture_image = self.sample.images[number - 1]
        frame.capture_image, frame.ground_truth = await self.sample.get_next_frame_data()
        frame.size = self.sample.capture_size
        frame.complete_capture()
        return frame

    # = feature selection =

    def start_feature_selection(self):
        self.commence_feature_selection()
        logger.info("calculate roi...")
        self.calculate_frame_roi(self.initial_frame)
        logger.info("generate sroi and extract features")
        self.generate_frame_sroi(self.initial_frame)
        logger.info("generate target mask")
        self.extract_frame_features(self.initial_frame)
        self.generate_frame_target_mask(self.initial_frame)

        logger.info("train selector")
        self.tracker.feature_selector.start_training(
            self.module_states.feature_selector,
            self.initial_frame)
        return
        # load data into selector:
        # self.tracker.feature_selector.load_data_for_selection(
        #     self.module_states.feature_selector, self.initial_frame)
        # self.selection_cost = self.tracker.feature_selector.cost(
        #     self.module_states.feature_selector)
        # self.tracker.feature_selector.

    def feature_selection_step(self):
        assert self.state == 'selecting_features'
        self.tracker.feature_selector.training_step(
            self.module_states.feature_selector)

    def feature_selection_costs_string(self):
        assert self.state == 'selecting_features'
        return self.tracker.feature_selector.training_costs_string(
            self.module_states.feature_selector)

    def feature_selection_done(self):
        assert self.state == 'selecting_features'
        return self.tracker.feature_selector.training_done(
            self.module_states.feature_selector)

    def feature_selection_forward(self):
        """
        Calculate current prediction of feature selector.
        """
        self.tracker.feature_selector.calculate_forward(
            self.module_states.feature_selector)

    def feature_selection_evaluate(self):
        self.tracker.feature_selector.evaluate_selection(
            self.module_states.feature_selector,
            self.current_frame.features,
            self.current_frame.target_mask,
        )
        for name, orders in self.module_states.feature_selector['feature_orders'].items():
            logger.info(
                'Feature orders for %s: %s', name, ', '.join([str(o) for o in orders]))

    def finish_feature_selection(self):
        # find out impact of individual features:
        self.feature_selection_evaluate()
        self.tracker.feature_selector.free_selection_nets(
            self.module_states.feature_selector)
        self.ts_features_selected = datetime.now()
        self.complete_feature_selection()

    def execute_feature_selection(self):
        """
        Do the whole feature selection loop.

        This executes the complete feature selection process. No results in
        between can be displayed (in a GUI). If you need inbetweens, call
        the start/done/step/finish methods directly from your program.
        """
        self.start_feature_selection()
        while not self.feature_selection_done():
            self.feature_selection_step()
        self.finish_feature_selection()

    # = consolidator =
    def start_consolidator_training(self):
        self.commence_consolidator_training()
        if not self.current_frame.did_reduction:
            self.reduce_frame_features()
        self.tracker.consolidator.start_training(
            self.module_states.consolidator,
            self.current_frame,)
        self.tracker.consolidator.store_frame(
            self.module_states.consolidator,
            'initial_frame',
            self.current_frame,
            1.0,
        )

    def consolidator_training_step(self):
        assert self.state == 'training_consolidator'
        self.tracker.consolidator.training_step(
            self.module_states.consolidator,
            self.current_frame,)

    def consolidator_training_done(self):
        assert self.state == 'training_consolidator'
        return self.tracker.consolidator.training_done(
            self.module_states.consolidator,
            self.current_frame,)

    def consolidator_training_cost(self):
        assert self.state == 'training_consolidator'
        return self.tracker.consolidator.training_cost(
            self.module_states.consolidator,
            self.current_frame,)

    def finish_consolidator_training(self):
        self.ts_consolidator_trained = datetime.now()
        self.complete_consolidator_training()

    def execute_consolidator_training(self):
        self.start_consolidator_training()
        while not self.consolidator_training_done():
            self.consolidator_training_step()
        self.finish_consolidator_training()

    # tracking:

    def start_tracking(self):
        self.commence_tracking()
        # log first frame to log (is always tracked perfectly, because gt is
        # known)
        frame = self.current_frame
        self.consolidate_frame_features(frame)
        frame.commence_pursuing()
        frame.prediction_quality = 1.0
        frame.complete_pursuing()
        self.evaluate_frame(frame)
        frame.result['updated'] = frame.updated
        l = {
            'result': frame.result,
            'consolidation_images': self.get_frame_consolidation_images(frame, decorations=False),
            'roi': frame.roi,
        }
        self.tracking_log.append(l)

    async def tracking_next_frame(self):
        self.commence_loading_next_frame()
        if self.frames_left():
            await self.load_next_frame()
            self.complete_loading_next_frame()
        else:
            # out of frames:
            self.enter_no_more_frames()

    def tracking_track_frame(self):
        self.commence_tracking_frame()
        frame = self.current_frame
        # process frame:
        # ps = [time.time()]  # 0
        ts_start = datetime.now()
        self.calculate_frame_roi(frame)
        self.roi_calculation_total_seconds += (datetime.now() - ts_start).total_seconds()
        # ps.append(time.time())  # 1
        ts_start = datetime.now()
        self.generate_frame_sroi(frame)
        self.sroi_generation_total_seconds += (datetime.now() - ts_start).total_seconds()
        # ps.append(time.time())  # 2
        ts_start = datetime.now()
        self.extract_frame_features(frame)
        self.feature_extraction_total_seconds += (datetime.now() - ts_start).total_seconds()
        # ps.append(time.time())  # 3
        ts_start = datetime.now()
        self.reduce_frame_features(frame)
        self.feature_reduction_total_seconds += (datetime.now() - ts_start).total_seconds()
        # ps.append(time.time())  # 4
        ts_start = datetime.now()
        self.consolidate_frame_features(frame, advance=True)
        self.feature_consolidation_total_seconds += (datetime.now() - ts_start).total_seconds()
        # pursue - find the best prediction in frame
        # ps.append(time.time())  # 5
        ts_start = datetime.now()
        self.pursue_frame(frame)
        self.pursuing_total_seconds += (datetime.now() - ts_start).total_seconds()

        se_start_time = datetime.now()
        self.estimate_scale(frame)
        self.se_total_seconds += (datetime.now() - se_start_time).total_seconds()

        # ps.append(time.time())  # 6

        # lost? TODO: make it modular and nice!
#        if frame.prediction_quality <= 0.0:
#            logger.info("LOST OBJECT!!!")
#            hiob.detector.detect(self, frame)
#            exit()

        self.complete_tracking_frame()

        # ts = [p1 - p0 for p0, p1 in zip(ps[:-1], ps[1:])]
        #
        # total = ps[-1] - ps[0]
        # log = "tracking profile"
        # for n, t in enumerate(ts):
        #     log += "; part{}: {:.2} ({:.2})".format(n, t, t / total)
        # print(log)

    def tracking_evaluate_frame(self):
        self.commence_evaluating_frame()
        frame = self.current_frame
        self.evaluate_frame(frame)
        #frame.sroi_image = None
        #frame.capture_image = None
        #frame.features = None
        del frame
        self.complete_evaluating_frame()

    def tracking_log_frame(self):
        frame = self.current_frame
        # this is found out too late, so have to put it here
        frame.result['updated'] = frame.updated
        l = {
            'result': frame.result,
            'consolidation_images': self.get_frame_consolidation_images(frame, decorations=False),
            'roi': frame.roi,
        }
        #print("tracking log:")
        #print(str(l))
        self.tracking_log.append(l)

    def tracking_publish_position(self):
        frame = self.current_frame
        if self.publisher is not None:
            self.publisher.publish(frame.result)

    def tracking_done(self):
        return not self.frames_left() or self.tracker.interrupt_received

    async def tracking_step(self):
        await self.tracking_next_frame()
        self.tracking_track_frame()
        self.tracking_evaluate_frame()
        self.update_consolidator()
        self.tracking_log_frame()
        if self.configuration['ros_mode']:
            self.tracking_publish_position()

    def finish_tracking(self):
        self.initial_frame = None
        self.current_frame = None
        self.commence_evaluate_tracking()
        self.ts_tracking_completed = datetime.now()
        #rospy.signal_shutdown("Tracking terminated.")
        evaluation.do_tracking_evaluation(self)
        self.complete_evaluate_tracking()

    async def execute_tracking(self):
        self.start_tracking()
        while not self.tracking_done():
            await self.tracking_step()
        self.finish_tracking()

    # update consolidator:
    def update_consolidator(self):
        frame = self.current_frame
        consolidator = self.tracker.consolidator
        since = frame.number - \
            self.module_states.consolidator['last_update_frame']
        logger.info("Consolidator update needed? Last update %d frames ago. Confidence: %0.4f, Update threshold %0.4f <= c <= %0.4f",
                    since, frame.prediction_quality,
                    consolidator.update_threshold, consolidator.update_lower_threshold, )
        # find out, if we need update:
        if consolidator.update_min_frames and since < consolidator.update_min_frames:
            logger.info(
                "Min frames without update not reached, update blocked.")
            return
        update_needed = False
        if consolidator.update_max_frames and since > consolidator.update_max_frames:
            # we do have a limit for max frames without update, so check for
            # that:
            logger.info("Max frames without update passed, update forced.")
            update_needed = True
            self.updates_max_frames += 1
            frame.updated = 'f'
        else:
            if frame.prediction_quality < consolidator.update_threshold:
                logger.info("Confidence too low, no update")
            elif frame.prediction_quality > consolidator.update_lower_threshold:
                logger.info("Confidence too high, no update")
            else:
                logger.info("Confidence thresholds, updating consolidator")
                update_needed = True
                self.updates_confidence += 1
                frame.updated = 'c'
        if not update_needed:
            return
        if consolidator.update_use_quality:
            weight = min(
                1.0, frame.prediction_quality * consolidator.update_current_factor)
        else:
            weight = consolidator.update_current_factor
        logger.info(
            "Frame prediction quality %04f, updating consolidator, current weight: %03f", frame.prediction_quality, weight)
        self.generate_frame_target_mask(frame, dev=consolidator.sigma_update)
        consolidator.update(
            self.module_states.consolidator,
            frame,
            weight,
            steps=consolidator.update_max_iterations,
        )
        # store frame for later updates
        consolidator.store_frame(
            self.module_states.consolidator,
            frame.number,
            frame,
            weight,
        )
        # same frame number of last update
        self.module_states.consolidator['last_update_frame'] = frame.number

    async def execute_everything(self):
        self.execute_feature_selection()
        self.execute_consolidator_training()
        await self.execute_tracking()

    # frame processing:

    def calculate_frame_roi(self, frame=None):
        if frame is None:
            frame = self.current_frame
        frame.commence_roi()
        self.tracker.roi_calculator.calculate_roi(frame=frame)
        logger.info("ROI on %s is %s", frame, frame.roi)
        frame.complete_roi()

    def generate_frame_sroi(self, frame=None):
        if frame is None:
            frame = self.current_frame
        frame.commence_sroi()
        self.tracker.sroi_generator.generate_sroi(frame)
        frame.complete_sroi()

    def extract_frame_features(self, frame=None):
        if frame is None:
            frame = self.current_frame
        frame.commence_extraction()
        self.tracker.feature_extractor.extract_features(self, frame)
        frame.complete_extraction()

    def generate_frame_target_mask(self, frame=None, dev=0.6):
        # TODO: this should be external and configurable
        if frame is None:
            frame = self.current_frame
        mask_pos = self.capture_to_mask(frame.predicted_position, frame.roi)
        mask_shape = (1, self.mask_size[0], self.mask_size[1], 1)
        target_mask = gen_gauss_mask(
            self.mask_size, mask_pos, dev).T.reshape(mask_shape)
        frame.target_mask = target_mask

    def reduce_frame_features(self, frame=None):
        if frame is None:
            frame = self.current_frame
        frame.commence_reduction()
        reduced = self.tracker.feature_selector.reduce_features(
            self.module_states.feature_selector,
            frame.features,
        )
        frame.features = reduced
        frame.complete_reduction()

    def consolidate_frame_features(self, frame=None, advance=True):
        if frame is None:
            frame = self.current_frame
        if advance:
            frame.commence_consolidation()
        self.tracker.consolidator.consolidate_features(
            self.module_states.consolidator,
            frame
        )
        if advance:
            frame.complete_consolidation()

    def pursue_frame(self, frame=None):
        if frame is None:
            frame = self.current_frame
        frame.commence_pursuing()
        self.tracker.pursuer.pursue(
            self.module_states.pursuer,
            frame)
        # lost?
        if frame.prediction_quality <= 0.0:
            logger.info("Lost object (1), trying detection 1")
            frame.lost = 1
            self.tracker.pursuer.pursue(
                self.module_states.pursuer,
                frame, lost=1)
            if frame.prediction_quality <= 0.1:
                frame.lost = 3
                logger.info("Lost object (3), using previous position")
                frame.predicted_position = frame.previous_position

        frame.complete_pursuing()

    def estimate_scale(self, frame=None):
        if frame is None:
            frame = self.current_frame

        frame.predicted_position = self.tracker.estimator.estimate_scale(
            frame=frame,
            feature_mask=frame.image_mask,
            mask_scale_factor=frame.mask_scale_factor,
            tracking=self)

    def evaluate_frame(self, frame=None):
        if frame is None:
            frame = self.current_frame
        frame.commence_evaluation()
        result = {
            'predicted_position': frame.predicted_position,
            'prediction_quality': frame.prediction_quality,
            'lost': frame.lost,
        }
        if frame.ground_truth:
            gt = frame.ground_truth
            p = frame.predicted_position
            result['overlap_score'] = gt.overlap_score(p)
            result['center_distance'] = gt.center_distance(p)
            result['relative_center_distance'] = gt.relative_center_distance(p)
            result['adjusted_overlap_score'] = gt.adjusted_overlap_score(p)
            result['gt_size_score'] = gt.width * gt.height * 0.1
            result['size_score'] = frame.predicted_position.width * frame.predicted_position.height * 0.1
        else:
            result['overlap_score'] = None
            result['center_distance'] = None
            result['relative_center_distance'] = None
            result['adjusted_overlap_score'] = None
            result['gt_size_score'] = None
            result['size_score'] = (frame.predicted_position.width * frame.predicted_position.height) * 0.1
        frame.result = result
        frame.complete_evaluation()

    # ==

    async def load_next_frame(self):
        previous_position = self.current_frame.predicted_position
        frame = await self._get_next_sample_frame()
        frame.previous_position = previous_position
        frame.before_previous_position = self.current_frame.previous_position
        self.current_frame = frame

    # information:

    def frames_left(self):
        return self.sample.frames_left()

    def __repr__(self):
        if self.sample:
            return "<Tracking#{} on {}>".format(self.serial, self.sample.full_name)
        else:
            return "<Tracking#{}>".format(self.serial)

    # == coordinate conversions ==

    def capture_to_sroi(self, pos, roi):
        """
        Convert rect in capture to rect in scaled roi.
        """
        rx, ry, rw, rh = roi.tuple
        px, py, pw, ph = pos.tuple
        scale_w = self.sroi_size[0] / rw
        scale_h = self.sroi_size[1] / rh
        ix = round((px - rx) * scale_w)
        iy = round((py - ry) * scale_h)
        iw = scale_w * pw
        ih = scale_h * ph
        return Rect(ix, iy, iw, ih)

    def sroi_to_capture(self, pos, roi):
        """
        Convert rect in scaled roi to rect in capture.
        """
        rx, ry, rw, rh = roi.tuple
        sx, sy, sw, sh = pos.tuple
        scale_w = self.sroi_size[0] / rw
        scale_h = self.sroi_size[1] / rh
        cx = round(sx / scale_w + rx)
        cy = round(sy / scale_h + ry)
        cw = sw / scale_w
        ch = sh / scale_h
        return Rect(cx, cy, cw, ch)

    def sroi_to_mask(self, sroi_pos):
        return Rect(
            int(sroi_pos.left / self.sroi_to_mask_ratio[0]),
            int(sroi_pos.top / self.sroi_to_mask_ratio[1]),
            int(sroi_pos.width / self.sroi_to_mask_ratio[0]),
            int(sroi_pos.height / self.sroi_to_mask_ratio[1]),
        )

    def mask_to_sroi(self, mask_pos):
        return Rect(
            int(mask_pos.left * self.sroi_to_mask_ratio[0]),
            int(mask_pos.top * self.sroi_to_mask_ratio[1]),
            int(mask_pos.width * self.sroi_to_mask_ratio[0]),
            int(mask_pos.height * self.sroi_to_mask_ratio[1]),
        )

    def capture_to_mask(self, pos, roi):
        """
        Convert rect in capture to rect in mask roi.
        """
        rx, ry, rw, rh = roi.tuple
        px, py, pw, ph = pos.tuple
        scale_w = (self.sroi_size[0] / rw) / self.sroi_to_mask_ratio[0]
        scale_h = (self.sroi_size[1] / rh) / self.sroi_to_mask_ratio[1]
        ix = int(round((px - rx) * scale_w))
        iy = int(round((py - ry) * scale_h))
        iw = int(round(scale_w * pw))
        ih = int(round(scale_h * ph))
        return Rect(ix, iy, iw, ih)

    def mask_to_capture(self, pos, roi):
        """
        Convert rect in mask to rect in capture.
        """
        rx, ry, rw, rh = roi.tuple
        sx, sy, sw, sh = pos.tuple
        scale_w = (self.sroi_size[0] / rw) / self.sroi_to_mask_ratio[0]
        scale_h = (self.sroi_size[1] / rh) / self.sroi_to_mask_ratio[1]
        cx = int(round(sx / scale_w + rx))
        cy = int(round(sy / scale_h + ry))
        cw = int(round(sw / scale_w))
        ch = int(round(sh / scale_h))
        return Rect(cx, cy, cw, ch)

    # == images ==

    def get_current_frame_number(self):
        return self.sample.current_frame_id

    def get_frame_capture_image(self, frame=None, decorations=True, width=1):
        if frame is None:
            frame = self.current_frame
        #im = frame.capture_image.copy()
        im = Image.fromarray(np.asarray(frame.capture_image, dtype=np.uint8))

        if decorations:
            draw = ImageDraw.Draw(im)
            if frame.roi:
                pos = frame.roi.inner

                gt_rep = [pos[0], pos[1], pos[2], pos[3]]
                gt_line_points = ((gt_rep[0], gt_rep[1]),
                                  (gt_rep[2], gt_rep[1]),
                                  (gt_rep[2], gt_rep[3]),
                                  (gt_rep[0], gt_rep[3]),
                                  (gt_rep[0], gt_rep[1]))
                draw.line(gt_line_points, self.colours['roi'], width=width)

            if frame.ground_truth:
                pos = frame.ground_truth.inner

                gt_rep = [pos[0], pos[1], pos[2], pos[3]]
                gt_line_points = ((gt_rep[0], gt_rep[1]),
                                  (gt_rep[2], gt_rep[1]),
                                  (gt_rep[2], gt_rep[3]),
                                  (gt_rep[0], gt_rep[3]),
                                  (gt_rep[0], gt_rep[1]))
                draw.line(gt_line_points, self.colours['ground_truth'], width=width)

            if frame.predicted_position:
                pos = frame.predicted_position.inner

                gt_rep = [pos[0], pos[1], pos[2], pos[3]]
                gt_line_points = ((gt_rep[0], gt_rep[1]),
                                  (gt_rep[2], gt_rep[1]),
                                  (gt_rep[2], gt_rep[3]),
                                  (gt_rep[0], gt_rep[3]),
                                  (gt_rep[0], gt_rep[1]))
                draw.line(gt_line_points, self.colours['prediction'], width=width)

        return im

    def get_frame_sroi_image(self, frame=None, decorations=True):
        if frame is None:
            frame = self.current_frame
        im = Image.fromarray(np.asarray(self.tracker.sroi_generator.generated_sroi.read_value().eval(), dtype=np.uint8)[0])
        if decorations:
            draw = ImageDraw.Draw(im)
            if frame.ground_truth and frame.roi:
                pos = self.capture_to_sroi(frame.ground_truth, frame.roi).inner
                draw.rectangle(pos, None, self.colours['ground_truth'])
            if frame.predicted_position and frame.roi:
                pos = self.capture_to_sroi(
                    frame.predicted_position, frame.roi).inner
                draw.rectangle(pos, None, self.colours['prediction'])
        return im

    def get_frame_consolidation_images(self, frame=None, decorations=True):
        if frame is None:
            frame = self.current_frame
        images = OrderedDict()
        for name, f in frame.consolidated_features.items():
            im = Image.fromarray(
                self.cmap(f.reshape(self.mask_size), bytes=True)
            )
            if decorations:
                draw = ImageDraw.Draw(im)
                if frame.ground_truth and frame.roi:
                    pos = self.capture_to_mask(
                        frame.ground_truth, frame.roi).inner
                    draw.rectangle(pos, None, self.colours['ground_truth'])
                if frame.predicted_position and frame.roi:
                    pos = self.capture_to_mask(
                        frame.predicted_position, frame.roi).inner
                    draw.rectangle(pos, None, self.colours['prediction'])
            images[name] = im

        return images

    def get_frame_target_mask_image(self, frame=None, decorations=True):
        if frame is None:
            frame = self.current_frame
        im = Image.fromarray(
            self.cmap(frame.target_mask.reshape(self.mask_size), bytes=True)
        )
        if decorations:
            draw = ImageDraw.Draw(im)
            if frame.ground_truth and frame.roi:
                pos = self.capture_to_mask(frame.ground_truth, frame.roi).inner
                draw.rectangle(pos, None, self.colours['ground_truth'])
        return im

    def get_frame_selection_mask_images(self, frame=None, decorations=True):
        if frame is None:
            frame = self.current_frame

        images = OrderedDict()

        for n, f in self.module_states.feature_selector['forwards'].items():
            im = Image.fromarray(
                self.cmap(
                    f.reshape(self.mask_size), bytes=True)
            )
            if decorations:
                draw = ImageDraw.Draw(im)
                if frame.ground_truth:
                    draw.rectangle(
                        frame.ground_truth.inner, None, self.colours['ground_truth'])
            images[n] = im
        return images



    # tracking log evaluation:
    def get_evaluation_figures(self):
        cd = []
        ov = []
        in20 = 0
        for l in self.tracking_log:
            r = l['result']
            if r['center_distance'] <= 20:
                in20 += 1
            cd.append(r['center_distance'])
            ov.append(r['overlap_score'])

        dim = np.arange(1, len(cd) + 1)

        f = plt.figure(figsize=(4.5, 3))
        plt.xlabel("frame")
        plt.ylabel("center distance")
        plt.axhline(y=20, color='r', linestyle='--')
        plt.plot(dim, cd, 'k', dim, cd, 'bo')
#        center_score = in20 / len(cd)
#        tx = "prec(20) = %0.4f" % center_score
#        plt.text(4, 0.1, tx)
        plt.xlim(1, len(cd))
        cd_im = figure_to_image(f)

        f = plt.figure(figsize=(4.5, 3))
        plt.xlabel("frame")
        plt.ylabel("overlap score")
        plt.plot(dim, ov, 'k', dim, ov, 'bo')
        plt.xlim(1, len(cd))
        plt.ylim(ymin=0.0, ymax=1.0)
        ov_im = figure_to_image(f)
        return {
            'center_distance': cd_im,
            'overlap_score': ov_im,
        }
