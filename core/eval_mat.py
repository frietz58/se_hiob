import os
import scipy.io
from Rect import Rect
import argparse
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description="Evaluates matlab dsst results. the matlab workspace needs to be exported"
                                             "after tracking, the variables, ie the predicted positions and size, can"
                                             " than be accessed from the mat file. Mostly copy paste from the "
                                             "evaluation module from hiobs core. Example usage: 'python3 eval_mat.py "
                                             "-pts /home/finn/PycharmProjects/code-git/HIOB/core/Walking_dsst_results.mat'")

parser.add_argument("-pts", "--path", help="Absolute path to the .mat file from the workspace at the end of tracking")
args = parser.parse_args()


def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)
    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)
    return f


# get the path from the commandline argument
path = args.path

# read in .mat result file
results = scipy.io.loadmat(path)

rect_result_gt_arr = []

# read results from the mat file
for i in range(results['ground_truth'].shape[0]):
    # make a rect for prediction and gt
    # Rect: x y w h
    # matlab result: y x h w
    prediction_rect = Rect(results['positions'][i][1],
                           results['positions'][i][0],
                           results['positions'][i][3],
                           results['positions'][i][2])
    gt_rect = Rect(results['ground_truth'][i][1],
                   results['ground_truth'][i][0],
                   results['ground_truth'][i][3],
                   results['ground_truth'][i][2])

    rect_result_gt_arr.append([prediction_rect, gt_rect])

# calculate metrics
nested_metric_dict = {}
overlap_score = np.empty(len(rect_result_gt_arr))
center_distance = np.empty(len(rect_result_gt_arr))
relative_center_distance = np.empty(len(rect_result_gt_arr))
adjusted_overlap_score = np.empty(len(rect_result_gt_arr))
size_score = np.empty(len(rect_result_gt_arr))

for i in range(len(rect_result_gt_arr)):
    p = rect_result_gt_arr[i][0]
    gt = rect_result_gt_arr[i][1]
    nested_metric_dict[i] = {}
    nested_metric_dict[i]['overlap_score'] = gt.overlap_score(p)
    nested_metric_dict[i]['center_distance'] = gt.center_distance(p)
    nested_metric_dict[i]['relative_center_distance'] = gt.relative_center_distance(p)
    nested_metric_dict[i]['adjusted_overlap_score'] = gt.adjusted_overlap_score(p)
    nested_metric_dict[i]['size_score'] = (p[2] * p[3]) * 0.1

    overlap_score[i] = gt.overlap_score(p)
    center_distance[i] = gt.center_distance(p)
    relative_center_distance[i] = gt.relative_center_distance(p)
    adjusted_overlap_score[i] = gt.adjusted_overlap_score(p)
    size_score[i] = (p[2] * p[3]) * 0.1

# make new result folder, delete if old exists
words_in_path = path.split('/')
words_in_path[-1] = words_in_path[-1].replace('.mat', '')
eval_path = '/'.join(words_in_path)
if not os.path.exists(eval_path):
    os.mkdir(eval_path)

# draw and save plots
dim = np.arange(1, len(center_distance) + 1)

# distances:
figure_file2 = os.path.join(eval_path, 'center_distance.svg')
figure_file3 = os.path.join(eval_path, 'center_distance.pdf')
f = plt.figure()
plt.xlabel("frame")
plt.ylabel("center distance")
plt.axhline(y=20, color='r', linestyle='--')
plt.plot(dim, center_distance, 'k', dim, center_distance, 'b-')
plt.xlim(1, len(center_distance))

if os.path.isfile(figure_file2):
    os.remove(figure_file2)
if os.path.isfile(figure_file3):
    os.remove(figure_file3)

plt.savefig(figure_file2)
plt.savefig(figure_file3)
plt.close()


# distances:
figure_file2 = os.path.join(eval_path, 'relative_center_distance.svg')
figure_file3 = os.path.join(eval_path, 'relative_center_distance.pdf')
f = plt.figure()
plt.xlabel("frame")
plt.ylabel("rel. center distance")
plt.axhline(y=1, color='r', linestyle='--')
plt.plot(dim, relative_center_distance, 'k', dim, relative_center_distance, 'b-')
plt.ylim(ymin=0.0, ymax=3.0)
plt.xlim(5, len(relative_center_distance))

if os.path.isfile(figure_file2):
    os.remove(figure_file2)
if os.path.isfile(figure_file3):
    os.remove(figure_file3)

plt.savefig(figure_file2)
plt.savefig(figure_file3)
plt.close()


# size Plot:
figure_file2 = os.path.join(eval_path, 'size_over_time.svg')
figure_file3 = os.path.join(eval_path, 'size_over_time.pdf')
f = plt.figure()
plt.xlabel("frame")
plt.ylabel("size")
plt.axhline(y=size_score[0], color='r', linestyle='--')
plt.plot(dim, size_score, 'k', dim, size_score, 'b-')
plt.xlim(1, len(size_score))

if os.path.isfile(figure_file2):
    os.remove(figure_file2)
if os.path.isfile(figure_file3):
    os.remove(figure_file3)

plt.savefig(figure_file2)
plt.savefig(figure_file3)
plt.close()


# overlap
figure_file2 = os.path.join(eval_path, 'overlap_score.svg')
figure_file3 = os.path.join(eval_path, 'overlap_score.pdf')
f = plt.figure()
plt.xlabel("frame")
plt.ylabel("overlap score")
plt.plot(dim, overlap_score, 'k', dim, overlap_score, 'b-')
plt.xlim(1, len(overlap_score))
plt.ylim(ymin=0.0, ymax=1.0)

if os.path.isfile(figure_file2):
    os.remove(figure_file2)
if os.path.isfile(figure_file3):
    os.remove(figure_file3)

plt.savefig(figure_file2)
plt.savefig(figure_file3)
plt.close()


figure_file2 = os.path.join(eval_path, 'adjusted_overlap_score.svg')
figure_file3 = os.path.join(eval_path, 'adjusted_overlap_score.pdf')
f = plt.figure()
plt.xlabel("frame")
plt.ylabel("overlap score")
plt.plot(dim, adjusted_overlap_score, 'k', dim, adjusted_overlap_score, 'b-')
plt.xlim(1, len(adjusted_overlap_score))
plt.ylim(ymin=0.0, ymax=1.0)

if os.path.isfile(figure_file2):
    os.remove(figure_file2)
if os.path.isfile(figure_file3):
    os.remove(figure_file3)

plt.savefig(figure_file2)
plt.savefig(figure_file3)
plt.close()


# eval from paper:
dfun = build_dist_fun(center_distance)
rdfun = build_dist_fun(relative_center_distance)
ofun = build_over_fun(overlap_score)
aofun = build_over_fun(adjusted_overlap_score)

figure_file2 = os.path.join(eval_path, 'precision_plot.svg')
figure_file3 = os.path.join(eval_path, 'precision_plot.pdf')
f = plt.figure()
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

figure_file2 = os.path.join(eval_path, 'relative_precision_plot.svg')
figure_file3 = os.path.join(eval_path, 'relative_precision_plot.pdf')
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

figure_file2 = os.path.join(eval_path, 'success_plot.svg')
figure_file3 = os.path.join(eval_path, 'success_plot.pdf')
f = plt.figure()
x = np.arange(0., 1.001, 0.001)
y = [ofun(a) for a in x]
auc = np.trapz(y, x)
tx = "AUC = {0}".format(auc)
plt.text(0.05, 0.05, tx)
plt.xlabel("overlap score")
plt.ylabel("occurrence")
plt.xlim(xmin=0.0, xmax=1.0)
plt.ylim(ymin=0.0, ymax=1.0)
plt.plot(x, y)
plt.savefig(figure_file2)
plt.savefig(figure_file3)
plt.close()


figure_file2 = os.path.join(eval_path, 'adjusted_success_plot.svg')
figure_file3 = os.path.join(eval_path, 'adjusted_success_plot.pdf')
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


