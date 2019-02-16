import os
from Rect import Rect
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Evaluates the results of a HIOB tracking experiment. Precision and "
                                             "success will be calculated for each individual tracking sequence as well "
                                             "as for the entire collection of sequences which have been analyzed by "
                                             "the tracker. Additionally, precision and success will be calculated for "
                                             "the groups of attributes from the tb100")

parser.add_argument("-ptr", "--pathresults", help="Absolute path to the folder which contains the tracking logs of "
                                                  "the experiment.")

args = parser.parse_args()

# get the path from the commandline argument
results_path = args.pathresults

# get all subfolders of tracked sequences
sequences = [x[0] for x in os.walk(results_path)]
del sequences[0]
for i, folder in enumerate(sequences):
    if 'evaluation' in folder or not 'tracking' in folder:
        del sequences[i]


def get_single_sequence_results(sequence_dir):
    # read the tracking evaluation from the evaluation file
    with open(os.path.join(sequence_dir, str("evaluation.txt")), mode="r") as f:
        tracking_evaluation = f.readlines()

    # parse list into dict
    evaluation_dict = {}
    for line in tracking_evaluation:
        key, value = line.split("=")
        value = value.replace("\n", "")
        evaluation_dict[key] = value

    # read the tracking results from the log file
    with open(os.path.join(sequence_dir, str("tracking_log.txt")), mode="r") as f:
        tracking_results = f.readlines()

    # parse list into dict
    # lines are build in evaluation.py @ 92
    # indices:
    # 0:  frame_number,
    # 1: predicted_position_left, 2: predicted_position_top, 3: predicted_position_width, 4: predicted_position_height,
    # 5: prediction_quality,
    # 6: roi_pos_left, 7: roi_pos_top, 8: roi_pos_width, 9: roi_pos_height,
    # 10: center_distance, 11: relative_center_distance, 12: overlap_score, 13: adjusted_overlap_score, 14: lost,
    # 15: updated, 16: size_score
    results = []
    for index, line in enumerate(tracking_results):
        working_line = line.replace("\n", "").split(",")
        print(working_line)
        line_dict = {}
        for i in range(0, len(working_line)):
            try:
                working_line[i] = float(working_line[i])
            except ValueError:
                pass

            if i == 0:
                line_dict["frame_number"] = working_line[i]
            elif i == 1:
                line_dict["predicted_position"] = Rect(working_line[i],
                                                       working_line[i + 1],
                                                       working_line[i + 2],
                                                       working_line[i + 3])
                i += 3
            elif i == 5:
                line_dict["prediction_quality"] = working_line[i]
            elif i == 6:
                line_dict["roi_position"] = Rect(working_line[i],
                                                 working_line[i + 1],
                                                 working_line[i + 2],
                                                 working_line[i + 3])
                i += 3
            elif i == 10:
                line_dict["center_distance"] = working_line[i]
            elif i == 11:
                line_dict["relative_center_distance"] = working_line[i]
            elif i == 12:
                line_dict["overlap_score"] = working_line[i]
            elif i == 13:
                line_dict["adjusted_overlap_score"] = working_line[i]
            elif i == 14:
                line_dict["lost"] = working_line[i]
            elif i == 15:
                line_dict["updated"] = working_line[i]
            elif i == 16:
                line_dict["gt_size_score"] = working_line[i]
            elif i == 17:
                line_dict["size_score"] = working_line[i]

        results.append(line_dict)

    return results, evaluation_dict


def build_dist_fun(dists):
    def f(thresh):
        return (dists <= thresh).sum() / len(dists)

    return f


def build_over_fun(overs):
    def f(thresh):
        return (overs >= thresh).sum() / len(overs)

    return f


def create_graphs_for_sequence(single_sequence_result, sequence_folder):
    center_distances = np.empty(len(single_sequence_result))
    overlap_score = np.empty(len(single_sequence_result))

    for n, line in enumerate(single_sequence_result):
        center_distances[n] = line["center_distance"]
        overlap_score[n] = line["overlap_score"]

    # precision plot
    dfun = build_dist_fun(center_distances)
    figure_file2 = os.path.join(sequence_folder, 'evaluation/precision_plot.svg')
    figure_file3 = os.path.join(sequence_folder, 'evaluation/precision_plot.pdf')
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
    ofun = build_over_fun(overlap_score)
    figure_file2 = os.path.join(sequence_folder, 'evaluation/success_plot.svg')
    figure_file3 = os.path.join(sequence_folder, 'evaluation/success_plot.pdf')
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


def eval_sequences_only(tracked_sequences):
    # traverse all sub folders and do evaluation for each sequence
    for sequence in tracked_sequences:
        if not os.path.exists(sequence + "/evaluation"):
            os.mkdir(os.path.join(sequence, "evaluation"))
        sequence_result, tracking_evaluation = get_single_sequence_results(sequence)
        create_graphs_for_sequence(sequence_result, sequence)

def eval_tracking(tracking_dir):
    # traverse all sub folder, eval each sequence and get results for each sequence
    sequences_results = []
    for sequence in tracking_dir:
        if not os.path.exists(sequence + "/evaluation"):
            os.mkdir(os.path.join(sequence, "evaluation"))
        sequence_result, sequence_evaluation = get_single_sequence_results(sequence)
        sequences_results.extend(sequence_result)
        print(np.shape(sequences_results))
        create_graphs_for_sequence(sequence_result, sequence)

    if not os.path.exists(results_path + "/evaluation"):
        os.mkdir(os.path.join(results_path, "evaluation"))

    # treat collection of results for tracking like one sequence
    create_graphs_for_sequence(sequences_results, results_path)


if __name__ == '__main__':
    # eval_sequences_only(sequences)
    eval_tracking(sequences)

