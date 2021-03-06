"""
Created on 2016-11-30

@author: Peer Springstübe
"""

import os
import logging
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#from matplotlib import rcParams
#rcParams['font.family'] = 'serif'
#rcParams['font.size'] = 10


def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)
    return f


# calculates the ares between two curves
def area_between_curves(curve1, curve2):
    assert len(curve1) == len(curve2), "Not the same amount of data points in the two curves"
    abc = 0
    for i in range(0, len(curve1)):
        abc += abs(curve1[i] - curve2[i])

    abc = abc / len(curve1)
    return round(abc, 2)


def normalize_size_datapoints(log):

    size_scores = []
    gt_size_scores = []

    # get all size scores
    for line in log:
        size_scores.append(line['result']['size_score'])
        gt_size_scores.append(line['result']['gt_size_score'])

    # normalize each size score
    max_val = max(gt_size_scores)
    min_val = min(gt_size_scores)
    if min_val == max_val:
        min_val = 1

    for line in log:
        line['result']['size_score'] = (line['result']['size_score'] - min_val) / (max_val - min_val + 0.0025) * 100
        line['result']['gt_size_score'] = (line['result']['gt_size_score'] - min_val) / (max_val - min_val + 0.0025) * 100

    return log


def do_tracking_evaluation(tracking):
    tracker = tracking.tracker

    evaluation = OrderedDict()
    evaluation['set_name'] = tracking.sample.set_name
    evaluation['sample_name'] = tracking.sample.name
    evaluation['loaded'] = tracking.ts_loaded
    evaluation['features_selected'] = tracking.ts_features_selected
    evaluation['consolidator_trained'] = tracking.ts_consolidator_trained
    evaluation['tracking_completed'] = tracking.ts_tracking_completed
    evaluation['total_seconds'] = (
        tracking.ts_tracking_completed - tracking.ts_loaded).total_seconds()
    evaluation['preparing_seconds'] = (
        tracking.ts_consolidator_trained - tracking.ts_loaded).total_seconds()
    evaluation['tracking_seconds'] = (
        tracking.ts_tracking_completed - tracking.ts_consolidator_trained).total_seconds()
    evaluation['sample_frames'] = tracking.total_frames
    evaluation['frame_rate'] = tracking.total_frames / \
        evaluation['total_seconds']
    evaluation['tracking_frame_rate'] = tracking.total_frames / evaluation['tracking_seconds']
    evaluation['pursuing_frame_rate'] = tracking.total_frames / tracking.pursuing_total_seconds
    evaluation['roi_calculation_frame_rate'] = tracking.total_frames / tracking.roi_calculation_total_seconds
    evaluation['sroi_generation_frame_rate'] = tracking.total_frames / tracking.sroi_generation_total_seconds
    evaluation['feature_extraction_frame_rate'] = tracking.total_frames / tracking.feature_extraction_total_seconds
    evaluation['feature_reduction_frame_rate'] = tracking.total_frames / tracking.feature_reduction_total_seconds
    evaluation['feature_consolidation_frame_rate'] = tracking.total_frames / tracking.feature_consolidation_total_seconds
    evaluation['se_total_seconds'] = tracking.se_total_seconds
    evaluation['se_frame_rate'] = tracking.total_frames / tracking.se_total_seconds

    tracking_dir = os.path.join(tracker.execution_dir, tracking.name)
    try:
        os.makedirs(tracking_dir)
    except:
        logger.error(
            "Could not create tracking log dir '%s', results will be wasted", tracking_dir)

    log = tracking.tracking_log
    princeton_lines = []
    csv_lines = []
    lost1 = 0
    lost2 = 0
    lost3 = 0
    failures = 0

    log = normalize_size_datapoints(log)

    for n, l in enumerate(log):
        r = l['result']
        pos = r['predicted_position']
        roi = l['roi']
        # princeton as they want it:
        if r['lost'] >= 3:
            # lost object:
            line = "NaN,NaN,NaN,NaN"
        else:
            # found, position:
            line = "{},{},{},{}".format(
                pos.left, pos.top, pos.right, pos.bottom)
        princeton_lines.append(line)
        # my own log line:
        line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            n + 1,
            pos.left, pos.top, pos.width, pos.height,
            r['prediction_quality'],
            roi.left, roi.top, roi.width, roi.height,
            r['center_distance'],
            r['relative_center_distance'],
            r['overlap_score'],
            r['adjusted_overlap_score'],
            r['lost'],
            r['updated'],
            r['gt_size_score'],
            r['size_score']
        )
        csv_lines.append(line)
        if r['lost'] == 1:
            lost1 += 1
        elif r['lost'] == 2:
            lost2 += 1
        elif r['lost'] == 3:
            lost3 += 1
        if r['overlap_score'] == 0.0:
            failures += 1
    evaluation['lost1'] = lost1
    evaluation['lost2'] = lost2
    evaluation['lost3'] = lost3
    evaluation['failures'] = failures
    evaluation['failurePercentage'] = (failures * 100) / len(log)
    evaluation['updates_max_frames'] = tracking.updates_max_frames
    evaluation['updates_confidence'] = tracking.updates_confidence
    evaluation['updates_total'] = tracking.updates_max_frames + \
        tracking.updates_confidence

    princeton_filename = os.path.join(
        tracking_dir, tracking.sample.name.replace("/", "_") + '.txt')
    with open(princeton_filename, 'a') as f:
        f.write("\n".join(princeton_lines))
    csv_filename = os.path.join(tracking_dir, "tracking_log" + '.txt')
    with open(csv_filename, 'w') as f:
        f.write("".join(csv_lines))
    dump_filename = os.path.join(tracking_dir, "tracking_log" + '.p')
    with open(dump_filename, 'wb') as f:
        pickle.dump(log, f)

    # figures:
    cd = np.empty(len(log))
    rcd = np.empty(len(log))
    ov = np.empty(len(log))
    aov = np.empty(len(log))
    cf = np.empty(len(log))
    gt_ss = np.empty(len(log))
    ss = np.empty(len(log))
    in20 = 0
    for n, l in enumerate(log):
        r = l['result']
        if (r['center_distance'] is not None) and (r['center_distance'] <= 20):
            in20 += 1
        cd[n] = r['center_distance']
        rcd[n] = r['relative_center_distance']
        ov[n] = r['overlap_score']
        aov[n] = r['adjusted_overlap_score']
        cf[n] = r['prediction_quality']
        gt_ss[n] = r['gt_size_score']
        ss[n] = r['size_score']

    dim = np.arange(1, len(cd) + 1)

    # distances:
    figure_file2 = os.path.join(tracking_dir, 'center_distance.svg')
    figure_file3 = os.path.join(tracking_dir, 'center_distance.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("center distance")
    plt.axhline(y=20, color='r', linestyle='--')
    plt.plot(dim, cd, 'k', dim, cd, 'bo')
    plt.xlim(1, len(cd))
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    # Size Plot:
    abc = area_between_curves(ss, gt_ss)
    figure_file2 = os.path.join(tracking_dir, 'size_over_time.svg')
    figure_file3 = os.path.join(tracking_dir, 'size_over_time.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("size")
    tx = "abc = {0}".format(abc)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
    plt.text(5.05, 0.05, tx, bbox=bbox_props)
    plt.plot(dim, ss, color='#648fff', label='predicted size')
    plt.plot(dim, gt_ss, color='#ffb000', label='groundtruth size', alpha=0.7)
    plt.fill_between(dim, ss, gt_ss, color="#dc267f")
    plt.axhline(y=ss[0], color='k', linestyle=':', label='initial size')
    plt.legend(loc='best')
    plt.xlim(1, len(ss))
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    evaluation['area_between_size_curves'] = abc
    plt.close()

    # distances:
    figure_file2 = os.path.join(tracking_dir, 'relative_center_distance.svg')
    figure_file3 = os.path.join(tracking_dir, 'relative_center_distance.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("rel. center distance")
    plt.axhline(y=1, color='r', linestyle='--')
    plt.plot(dim, rcd, 'k', dim, rcd, 'bo')
    #plt.ylim(1, 5.0)
    plt.ylim(ymin=0.0, ymax=3.0)
    plt.xlim(5, len(rcd))
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    figure_file2 = os.path.join(tracking_dir, 'overlap_score.svg')
    figure_file3 = os.path.join(tracking_dir, 'overlap_score.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("overlap score")
    plt.plot(dim, ov, 'k', dim, ov, 'bo')
    plt.xlim(1, len(ov))
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    figure_file2 = os.path.join(tracking_dir, 'adjusted_overlap_score.svg')
    figure_file3 = os.path.join(tracking_dir, 'adjusted_overlap_score.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("overlap score")
    plt.plot(dim, aov, 'k', dim, aov, 'bo')
    plt.xlim(1, len(aov))
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    # eval from paper:
    dfun = build_dist_fun(cd)
    rdfun = build_dist_fun(rcd)
    ofun = build_over_fun(ov)
    aofun = build_over_fun(aov)

    figure_file2 = os.path.join(tracking_dir, 'precision_plot.svg')
    figure_file3 = os.path.join(tracking_dir, 'precision_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 50.1, .1)
    y = [dfun(a) for a in x]
    at20 = dfun(20)
    tx = "prec(20) = %0.4f" % at20
    plt.text(5.05, 0.05, tx)
    plt.xlabel("center distance [pixels]")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0, xmax=50)
    plt.ylim(ymin=0.0, ymax=1.0)
    ind = np.where(x == 20.0)[0][0]
    plt.axvline(x=x[ind], color='c', linestyle=':', label='precision at 20px')
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()
    # saving values:
    evaluation['precision_rating'] = at20

    figure_file2 = os.path.join(tracking_dir, 'relative_precision_plot.svg')
    figure_file3 = os.path.join(tracking_dir, 'relative_precision_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 50.1, .1)
    y = [rdfun(a) for a in x]
    at20_1 = rdfun(1)
    tx = "prec(20) = %0.4f" % at20_1
    plt.text(5.05, 0.05, tx)
    plt.xlabel("rel. center distance")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0, xmax=3.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()
    # saving values:
    evaluation['relative_precision_rating'] = at20_1

    figure_file2 = os.path.join(tracking_dir, 'success_plot.svg')
    figure_file3 = os.path.join(tracking_dir, 'success_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 1.001, 0.001)
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    tx = "AUC = %0.4f" % auc
    plt.text(0.05, 0.05, tx)
    plt.xlabel("overlap score")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()
    # saving values:
    evaluation['success_rating'] = auc

    figure_file2 = os.path.join(tracking_dir, 'adjusted_success_plot.svg')
    figure_file3 = os.path.join(tracking_dir, 'adjusted_success_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 1.001, 0.001)
    y = [aofun(a) for a in x]
    auc_1 = np.trapz(y, x)
    tx = "AUC = %0.4f" % auc_1
    plt.text(0.05, 0.05, tx)
    plt.xlabel("adjusted overlap score")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()
    # saving values:
    evaluation['adjusted_success_rating'] = auc

    # plot confidence
    figure_file2 = os.path.join(tracking_dir, 'confidence_plot.svg')
    figure_file3 = os.path.join(tracking_dir, 'confidence_plot.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("confidence")
    # plt.axhline(y=20, color='r', linestyle='--')
    plt.plot(dim, cf, 'k', dim, cf, 'bo')
    plt.xlim(1, len(cf))
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    tracking.evaluation = evaluation
    evaluation_file = os.path.join(tracking_dir, 'evaluation.txt')
    with open(evaluation_file, 'w') as f:
        for k, v in evaluation.items():
            f.write("{}={}\n".format(k, v))


def do_tracker_evaluation(tracker):
    execution_dir = tracker.execution_dir
    trackings_file = os.path.join(execution_dir, 'trackings.txt')
    tracking_sum = 0.0
    pursuing_sum = 0.0
    feature_extraction_sum = 0.0
    roi_calculation_sum = 0.0
    sroi_generation_sum = 0.0
    feature_reduction_sum = 0.0
    feature_consolidation_sum = 0.0
    se_sum = 0.0
    preparing_sum = 0.0
    precision_sum = 0.0
    relative_precision_sum = 0.0
    success_sum = 0.0
    adjusted_success_sum = 0.0
    se_time_total = 0.0
    lost1 = 0
    lost2 = 0
    lost3 = 0
    updates_max_frames = 0
    updates_confidence = 0
    updates_total = 0
    failures = 0
    if len(tracker.tracking_evaluations) == 0:
        logger.info("No evaluation data found.")
        return {}
    with open(trackings_file, 'w') as f:

        line = "#n,set_name,sample_name,sample_frames,precision_rating,relative_precision_rating,success_rating,adjusted_success_rating,loaded,features_selected,consolidator_trained,tracking_completed,total_seconds,preparing_seconds,tracking_seconds,frame_rate,roi_calculation_sum,sroi_generation_sum,feature_extraction_frame_rate,feature_reduction_sum,feature_consolidation_sum,pursuing_frame_rate,lost1,lost2,lost3,updates_max_frames,updates_confidence,update_total,se_total_seconds,se_frame_rate\n"
        f.write(line)
        for n, e in enumerate(tracker.tracking_evaluations):
            line = "{n},{set_name},{sample_name},{sample_frames},{precision_rating},{relative_precision_rating},{success_rating},{adjusted_success_rating},{loaded},{features_selected},{consolidator_trained},{tracking_completed},{total_seconds},{preparing_seconds},{tracking_seconds},{frame_rate},{pursuing_frame_rate},{feature_extraction_frame_rate},{lost1},{lost2},{lost3},{updates_max_frames},{updates_confidence},{updates_total},{se_total_seconds},{se_frame_rate}\n".format(
                n=n + 1,
                **e)
            f.write(line)
            preparing_sum += e['preparing_seconds']
            tracking_sum += e['tracking_seconds']
            pursuing_sum += e['pursuing_frame_rate']
            roi_calculation_sum += e['roi_calculation_frame_rate']
            sroi_generation_sum += e['sroi_generation_frame_rate']
            feature_extraction_sum += e['feature_extraction_frame_rate']
            feature_reduction_sum += e['feature_reduction_frame_rate']
            feature_consolidation_sum += e['feature_consolidation_frame_rate']
            se_sum += e['se_frame_rate']
            precision_sum += e['precision_rating']
            relative_precision_sum += e['relative_precision_rating']
            success_sum += e['success_rating']
            adjusted_success_sum += e['adjusted_success_rating']
            lost1 += e['lost1']
            lost2 += e['lost2']
            lost3 += e['lost3']
            failures += e['failures']
            updates_max_frames += e['updates_max_frames']
            updates_confidence += e['updates_confidence']
            updates_total += e['updates_total']
        roi_calculation_frame_rate = roi_calculation_sum / len(tracker.tracking_evaluations)
        sroi_generation_frame_rate = sroi_generation_sum / len(tracker.tracking_evaluations)
        feature_extraction_frame_rate = feature_extraction_sum / len(tracker.tracking_evaluations)
        feature_reduction_frame_rate = feature_reduction_sum / len(tracker.tracking_evaluations)
        feature_consolidation_frame_rate = feature_consolidation_sum / len(tracker.tracking_evaluations)
        pursuing__frame_rate = pursuing_sum / len(tracker.tracking_evaluations)
        se_frame_rate = se_sum / len(tracker.tracking_evaluations)

    # eval from paper:
    dfun = build_dist_fun(tracker.total_center_distances)
    rdfun = build_dist_fun(tracker.total_relative_center_distances)
    ofun = build_over_fun(tracker.total_overlap_scores)
    aofun = build_over_fun(tracker.total_adjusted_overlap_scores)

    figure_file2 = os.path.join(execution_dir, 'precision_plot.svg')
    figure_file3 = os.path.join(execution_dir, 'precision_plot.pdf')
    plt.figure()
    x = np.arange(0., 50.1, .1)
    y = [dfun(a) for a in x]
    at20 = dfun(20)
    tx = "prec(20) = %0.4f" % at20
    plt.text(5.05, 0.05, tx)
    plt.xlabel("center distance [pixels]")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0, xmax=400)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    figure_file2 = os.path.join(execution_dir, 'relative_precision_plot.svg')
    figure_file3 = os.path.join(execution_dir, 'relative_precision_plot.pdf')
    plt.figure()
    x = np.arange(0., 50.1, .1)
    y = [rdfun(a) for a in x]
    at1 = rdfun(1)
    tx = "prec(20) = %0.4f" % at1
    plt.text(5.05, 0.05, tx)
    plt.xlabel("rel. center distance")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0, xmax=3.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    figure_file2 = os.path.join(execution_dir, 'success_plot.svg')
    figure_file3 = os.path.join(execution_dir, 'success_plot.pdf')
    plt.figure()
    x = np.arange(0., 1.001, 0.001)
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    tx = "AUC = %0.4f" % auc
    plt.text(0.05, 0.05, tx)
    plt.xlabel("overlap score")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    figure_file2 = os.path.join(execution_dir, 'adjusted_success_plot.svg')
    figure_file3 = os.path.join(execution_dir, 'adjusted_success_plot.pdf')
    plt.figure()
    x = np.arange(0., 1.001, 0.001)
    y = [aofun(a) for a in x]
    auc_1 = np.trapz(y, x)
    tx = "AUC = %0.4f" % auc_1
    plt.text(0.05, 0.05, tx)
    plt.xlabel("overlap score")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    ev = OrderedDict()
    ev['execution_name'] = tracker.execution_name
    ev['execution_id'] = tracker.execution_id
    ev['execution_host'] = tracker.execution_host
    ev['execution_dir'] = tracker.execution_dir
    ev['environment_name'] = tracker.environment_name
    ev['git_revision'] = tracker.git_revision
    ev['git_dirty'] = tracker.git_dirty
    ev['random_seed'] = tracker.py_seed
    ev['started'] = tracker.ts_created
    ev['finished'] = tracker.ts_done
    ev['total_samples'] = len(tracker.tracking_evaluations)
    ev['total_frames'] = len(tracker.total_center_distances)
    ev['total_seconds'] = (
        tracker.ts_done - tracker.ts_created).total_seconds()
    ev['tracking_seconds'] = tracking_sum
    ev['average_seconds_per_sample'] = ev[
        'total_seconds'] / ev['total_samples']
    ev['frame_rate'] = ev['total_frames'] / ev['total_seconds']
    ev['tracking_frame_rate'] = ev['total_frames'] / ev['tracking_seconds']
    ev['roi_calculation_frame_rate'] = roi_calculation_frame_rate
    ev['sroi_generation_frame_rate'] = sroi_generation_frame_rate
    ev['feature_extraction_frame_rate'] = feature_extraction_frame_rate
    ev['feature_reduction_frame_rate'] = feature_reduction_frame_rate
    ev['feature_consolidation_frame_rate'] = feature_consolidation_frame_rate
    ev['pursuing_frame_rate'] = pursuing__frame_rate
    ev['se_frame_rate'] = se_frame_rate
    ev['preparing_seconds'] = preparing_sum
    apr = precision_sum / ev['total_samples']
    ev['average_precision_rating'] = apr
    asr = success_sum / ev['total_samples']
    ev['average_success_rating'] = asr
    ev['average_score'] = (apr + asr) / 2.0
    ev['total_precision_rating'] = at20
    ev['total_relative_precision_rating'] = at1
    ev['total_success_rating'] = auc
    ev['total_adjusted_success_rating'] = auc_1
    ev['total_score'] = (at20 + auc) / 2.0
    ev['total_adjusted_score'] = (at1 + auc_1) / 2.0
    ev['probe_score'] = (apr + asr + at20 + auc) / 4.0
    ev['lost1'] = lost1
    ev['lost2'] = lost2
    ev['lost3'] = lost3
    ev['failures'] = failures
    ev['updates_max_frames'] = updates_max_frames
    ev['updates_confidence'] = updates_confidence
    ev['updates_total'] = updates_total
    evaluation_file = os.path.join(execution_dir, 'evaluation.txt')
    with open(evaluation_file, 'w') as f:
        for k, v in ev.items():
            f.write("{}={}\n".format(k, v))
    return ev


def print_tracking_evaluation(evaluation, log_context):
    with log_context(logger):

        logger.info("Tracking complete for '{}/{}'.".format(
            evaluation['set_name'], evaluation['sample_name']))
        for key in ["total_seconds",
                    "preparing_seconds",
                    "sample_frames",
                    "frame_rate",
                    "tracking_frame_rate",
                    "pursuing_frame_rate",
                    "roi_calculation_frame_rate",
                    "sroi_generation_frame_rate",
                    "feature_extraction_frame_rate",
                    "feature_reduction_frame_rate",
                    "feature_consolidation_frame_rate",
                    "se_frame_rate",
                    "lost1",
                    "lost2",
                    "lost3",
                    "failures",
                    "failurePercentage"]:
            logger.info("{}: {}".format(key, evaluation[key]))
