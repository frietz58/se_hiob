import os
from core.Rect import Rect
import argparse
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import csv

parser = argparse.ArgumentParser(description="Evaluates an experiment from the matlab dsst reference implementation. "
                                             "Every value the script requires needs to be saved in the matlab workspace"
                                             " which gets loaded in.")

parser.add_argument("-ptr", "--pathresults", help="Absolute path to the folder which contains saved matlab workspaces "
                                                  "of each tracked sequence")

args = parser.parse_args()

results_path = args.pathresults


# get all workplace files
def get_saved_workplaces(result_dir):
    files_in_dir = os.listdir(result_dir)

    for i, file in enumerate(files_in_dir):
        if not ".mat" in file:
            del files_in_dir[i]

    return files_in_dir


# get the rects of predictions and gt from one workplace
def get_rects_from_workplace(workplace):
    results = scipy.io.loadmat(os.path.join(results_path, workplace))

    # Rect: x y w h
    # matlab result: y x h w
    preds = []
    gts = []
    for i in range(results['ground_truth'].shape[0]):
        prediction_rect = Rect(results['positions'][i][1],
                               results['positions'][i][0],
                               results['positions'][i][3],
                               results['positions'][i][2])
        gt_rect = Rect(results['ground_truth'][i][1],
                       results['ground_truth'][i][0],
                       results['ground_truth'][i][3],
                       results['ground_truth'][i][2])

        preds.append(prediction_rect)
        gts.append(gt_rect)

    return preds, gts


# get all rects of predictions and gt from each workplace in result dir
def get_all_rects(result_dir):
    workplaces = get_saved_workplaces(result_dir)

    pred_gt_rects = {"preds": [], "gts": []}
    for i, workplace in enumerate(workplaces):
        preds, gts = get_rects_from_workplace(workplace)
        pred_gt_rects["preds"].append(preds)
        pred_gt_rects["gts"].append(gts)

    all_preds = []
    for sequence_preds in pred_gt_rects["preds"]:
        for pred_rect in sequence_preds:
            all_preds.append(pred_rect)

    all_gts = []
    for sequence_gts in pred_gt_rects["gts"]:
        for gt_rect in sequence_gts:
            all_gts.append(gt_rect)

    return all_preds, all_gts


# get metric from rects
def get_metrics_from_rects(result_folder):
    all_preds, all_gts = get_all_rects(result_folder)

    center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)

    # overlap_scores = np.empty(len(all_preds))
    # center_distances = np.empty(len(all_preds))
    # relative_center_distance = np.empty(len(all_preds))
    # adjusted_overlap_score = np.empty(len(all_preds))
    # gt_size_scores = np.empty(len(all_preds))
    # size_scores = np.empty(len(all_preds))
    # frames = 0
    #
    # for i in range(len(all_preds)):
    #     p = all_preds[i]
    #     gt = all_gts[i]
    #
    #     overlap_scores[i] = gt.overlap_score(p)
    #     center_distances[i] = gt.center_distance(p)
    #     relative_center_distance[i] = gt.relative_center_distance(p)
    #     adjusted_overlap_score[i] = gt.adjusted_overlap_score(p)
    #     gt_size_scores[i] = (gt[2] * gt[3]) * 0.1
    #     size_scores[i] = (p[2] * p[3]) * 0.1
    #     frames += 1
    #
    # center_distances = np.asarray(center_distances)
    # overlap_scores = np.asarray(overlap_scores)
    # gt_size_scores = np.asarray(gt_size_scores)
    # size_scores = np.asarray(size_scores)

    # calculate the metrics based on the results for each frame of the sequences in the collection
    scores_for_rects = {}

    workplaces = get_saved_workplaces(result_folder)

    scores_for_rects["Samples"] = len(workplaces)
    scores_for_rects["Frames"] = frames

    dfun = build_dist_fun(center_distances)
    at20 = dfun(20)
    scores_for_rects["Avg. Precision"] = at20

    ofun = build_over_fun(overlap_scores)
    x = np.arange(0., 1.001, 0.001)
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    scores_for_rects["Avg. Success"] = auc

    normalized_size_score, normalized_gt_size_scores = normalize_size_datapoints(size_scores, gt_size_scores)
    abc = area_between_curves(normalized_gt_size_scores, normalized_size_score)
    scores_for_rects["Size Score"] = abc

    return scores_for_rects


# create a csv for the scores
def create_score_csv(results_folder):
    score_dict = get_metrics_from_rects(results_folder)
    scores = score_dict.keys()

    if not os.path.isdir(os.path.join(results_folder, "evaluation")):
        os.mkdir(os.path.join(results_folder, "evaluation"))

    out_csv = os.path.join(results_path, "evaluation/scores.csv")
    with open(out_csv, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=scores)
        writer.writeheader()

        writer.writerow(score_dict)


# create the graphs from rects
def create_graphs_from_rects(result_folder):
    all_preds, all_gts = get_all_rects(result_folder)

    center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)

    if not os.path.isdir(os.path.join(result_folder, "evaluation")):
        os.mkdir(os.path.join(result_folder, "evaluation"))

    # precision plot
    dfun = build_dist_fun(center_distances)
    figure_file2 = os.path.join(result_folder, 'evaluation/precision_plot.svg')
    figure_file3 = os.path.join(result_folder, 'evaluation/precision_plot.pdf')
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
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    # success plot
    ofun = build_over_fun(overlap_scores)
    figure_file2 = os.path.join(result_folder, 'evaluation/success_plot.svg')
    figure_file3 = os.path.join(result_folder, 'evaluation/success_plot.pdf')
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

    # Size Plot:
    normalized_size_score, normalized_gt_size_scores = normalize_size_datapoints(size_scores, gt_size_scores)
    abc = area_between_curves(normalized_size_score, normalized_gt_size_scores)
    dim = np.arange(1, len(normalized_size_score) + 1)
    figure_file2 = os.path.join(result_folder, 'evaluation/size_over_time.svg')
    figure_file3 = os.path.join(result_folder, 'evaluation/size_over_time.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("size")
    plt.text(x=10, y=10, s="abc={0}".format(abc))
    plt.plot(dim, normalized_size_score, 'r-', label='predicted size')
    plt.plot(dim, normalized_gt_size_scores, 'g-', label='groundtruth size', alpha=0.7)
    plt.fill_between(dim, normalized_size_score, normalized_gt_size_scores, color="y")
    plt.axhline(y=normalized_size_score[0], color='c', linestyle=':', label='initial size')
    plt.legend(loc='best')
    plt.xlim(1, len(normalized_size_score))
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()


# ================================= HELPER FUNCTIONS =================================
# helper function for metric
def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)

    return f


# helper function for metric
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


def normalize_size_datapoints(pred_size_scores, gt_size_scores):
    max_val = max(gt_size_scores)
    min_val = min(gt_size_scores)

    # normalize each size score based on min max
    for i in range(len(pred_size_scores)):
        pred_size_scores[i] = (pred_size_scores[i] - min_val) / (max_val - min_val + 0.0025) * 100
        gt_size_scores[i] = (gt_size_scores[i] - min_val) / (max_val - min_val + 0.0025) * 100

    return pred_size_scores, gt_size_scores


# function to get basic scores from rects
def get_scores_from_rects(preds, gts):
    overlap_scores = np.empty(len(preds))
    center_distances = np.empty(len(preds))
    relative_center_distance = np.empty(len(preds))
    adjusted_overlap_score = np.empty(len(preds))
    gt_size_scores = np.empty(len(preds))
    size_scores = np.empty(len(preds))
    frames = 0

    for i in range(len(preds)):
        p = preds[i]
        gt = gts[i]

        overlap_scores[i] = gt.overlap_score(p)
        center_distances[i] = gt.center_distance(p)
        relative_center_distance[i] = gt.relative_center_distance(p)
        adjusted_overlap_score[i] = gt.adjusted_overlap_score(p)
        gt_size_scores[i] = (gt[2] * gt[3]) * 0.1
        size_scores[i] = (p[2] * p[3]) * 0.1
        frames += 1

    center_distances = np.asarray(center_distances)
    overlap_scores = np.asarray(overlap_scores)
    gt_size_scores = np.asarray(gt_size_scores)
    size_scores = np.asarray(size_scores)

    return center_distances, overlap_scores, gt_size_scores, size_scores, frames


if __name__ == "__main__":
    create_score_csv(results_path)
    create_graphs_from_rects(results_path)
