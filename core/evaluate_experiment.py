import os
from Rect import Rect
import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml
import csv

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


# get the different folder representing different tracking executions
def get_tracking_folders(experiment_folder):
    sub_folders = [x[0] for x in os.walk(experiment_folder)]
    only_tracking_dirs = []
    for i, folder in enumerate(sub_folders):
        print(folder)
        if 'hiob-execution' not in folder:
            continue
        elif 'tracking' in folder:
            continue
        elif 'evaluation' in folder:
            continue
        else:
            only_tracking_dirs.append(folder)

    return only_tracking_dirs


# get all subfolders of tracked sequences
def get_tracked_sequences(tracking_dir):
    sequences = [x[0] for x in os.walk(tracking_dir)]
    only_sequence_folders = []
    del sequences[0]
    for i, folder in enumerate(sequences):
        if 'evaluation' in folder:
            continue
        else:
            only_sequence_folders.append(folder)

    return only_sequence_folders


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


def area_between_curves(curve1, curve2):
    assert len(curve1) == len(curve2), "Not the same amount of data points in the two curves"
    abc = 0
    for i in range(0, len(curve1)):
        abc += abs(curve1[i] - curve2[i])

    return abc


def create_graphs_for_sequence(single_sequence_result, sequence_folder):
    center_distances = np.empty(len(single_sequence_result))
    overlap_score = np.empty(len(single_sequence_result))
    gt_ss = np.empty(len(single_sequence_result))
    ss = np.empty(len(single_sequence_result))

    for n, line in enumerate(single_sequence_result):
        center_distances[n] = line["center_distance"]
        overlap_score[n] = line["overlap_score"]
        gt_ss[n] = line['gt_size_score']
        ss[n] = line['size_score']

    score_dict = {}

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
    score_dict["prec(20)"] = tx

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
    score_dict["success"] = tx

    # Size Plot:
    abc = area_between_curves(ss, gt_ss)
    dim = np.arange(1, len(ss) + 1)
    figure_file2 = os.path.join(sequence_folder, 'evaluation/size_over_time.svg')
    figure_file3 = os.path.join(sequence_folder, 'evaluation/size_over_time.pdf')
    f = plt.figure()
    plt.xlabel("frame")
    plt.ylabel("size")
    plt.text(x=10, y=10, s="abc={0}".format(abc))
    plt.plot(dim, ss, 'r-', label='predicted size')
    plt.plot(dim, gt_ss, 'g-', label='groundtruth size', alpha=0.7)
    plt.fill_between(dim, ss, gt_ss, color="y")
    plt.axhline(y=ss[0], color='c', linestyle=':', label='initial size')
    plt.legend(loc='best')
    plt.xlim(1, len(ss))
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    return score_dict


# read the informations from the tracker.yaml file to reconstruct which tracking algorithm has been used
def get_approach_from_yaml(tracking_dir):
    with open(tracking_dir + "/tracker.yaml", "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
            scale_estimator_conf = configuration["scale_estimator"]
        except yaml.YAMLError as exc:
            print(exc)

        algorithm = None

        if not scale_estimator_conf["use_scale_estimation"]:
            algorithm = "Baseline"
        elif scale_estimator_conf["use_scale_estimation"] and scale_estimator_conf["approach"] == "custom_dsst":
            algorithm = "DSST"
        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "candidates" \
                and scale_estimator_conf["c_change_aspect_ratio"]:
            algorithm = "Candidates dynamic"
        else:
            algorithm = "Candidates static"

        return algorithm


def eval_sequences(tracked_sequences):
    # traverse all sub folders and do evaluation for each sequence
    for sequence in tracked_sequences:
        if not os.path.exists(sequence + "/evaluation"):
            os.mkdir(os.path.join(sequence, "evaluation"))
        sequence_result, tracking_evaluation = get_single_sequence_results(sequence)
        create_graphs_for_sequence(sequence_result, sequence)


def create_tracking_results_csv(tracking_dir):

    sequnce_dirs = get_tracked_sequences(tracking_dir)

    approach = get_approach_from_yaml(tracking_dir)

    # get the results for every sequence into on csv
    csv_name = tracking_dir + '/evaluation/' + approach + '_sequence_results.csv'
    with open(csv_name, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=["Sample", "Total Frames", "Framerate", "Precision", "Success"])
        writer.writeheader()

        for sequence in sequnce_dirs:
            with open(sequence + "/evaluation.txt", "r") as evaltxt:
                lines = evaltxt.readlines()
                sample_name = lines[1].replace("\n", "").split("=")[1]
                number_frames = lines[9].replace("\n", "").split("=")[1]
                frame_rate = lines[10].replace("\n", "").split("=")[1]
                precision = lines[26].replace("\n", "").split("=")[1]
                success = lines[28].replace("\n", "").split("=")[1]
                writer.writerow({'Sample': sample_name,
                                 'Total Frames': number_frames,
                                 'Framerate': frame_rate,
                                 'Precision': precision,
                                 'Success': success})


def eval_tracking(tracking_dir):
    # traverse all sub folder, eval each sequence and get results for each sequence
    sequences_results = []

    # first handle all the sequences
    sequnce_dirs = get_tracked_sequences(tracking_dir)
    for sequence in sequnce_dirs:
        if not os.path.exists(sequence + "/evaluation"):
            os.mkdir(os.path.join(sequence, "evaluation"))
        sequence_result, sequence_evaluation = get_single_sequence_results(sequence)
        sequences_results.extend(sequence_result)
        create_graphs_for_sequence(sequence_result, sequence)

    # make dir for tracking evaluation
    if not os.path.exists(tracking_dir + "/evaluation"):
        os.mkdir(os.path.join(tracking_dir, "evaluation"))

    # treat collection of results for tracking like one sequence
    scores = create_graphs_for_sequence(sequences_results, tracking_dir)
    approach = get_approach_from_yaml(tracking_dir)

    # get the results for every sequence into on csv
    create_tracking_results_csv(tracking_dir)



def eval_all_trackings():
    tracking_folders = get_tracking_folders(results_path)
    for tracking_folder in tracking_folders:
        eval_tracking(tracking_folder)



if __name__ == '__main__':
    eval_all_trackings()


