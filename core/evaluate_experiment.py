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

parser.add_argument("-pta", "--attributes", help="Absolute path to the TB100 collection file, which contains the "
                                                 "attributes for each sequence.")

args = parser.parse_args()

# get the path from the commandline argument
results_path = args.pathresults
attributes_path = args.attributes


# get the different folder representing different tracking executions
def get_tracking_folders(experiment_folder):
    sub_folders = [x[0] for x in os.walk(experiment_folder)]
    only_tracking_dirs = []
    for i, folder in enumerate(sub_folders):
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


# read the tracking log file for one sequence and extract the values from the file
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

    return abc


# creates the graphs for a single sequence
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


# read the information from the tracker.yaml file to reconstruct which tracking algorithm has been used
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


# calculates the metrics for a given collection of sequences
def get_metrics_for_collections(sequences):

    center_distances = []
    overlap_scores = []
    ground_truth_size = []
    predicted_size = []
    frames = 0

    # for every sequence in the collection get the results on each frame
    for sequence in sequences:
        sequence_result, sequence_evaluation = get_single_sequence_results(sequence)

        for line in sequence_result:
            center_distances.append(line["center_distance"])
            overlap_scores.append(line["overlap_score"])
            ground_truth_size.append(line["gt_size_score"])
            predicted_size.append(line["size_score"])
        frames += len(sequence_result)

    # if the current collection is empty (probably only in test cases)
    if frames == 0:
        scores_for_collection = {}

        scores_for_collection["Samples"] = 0
        scores_for_collection["Frames"] = 0
        scores_for_collection["Precision"] = 'None'
        scores_for_collection["Success"] = 'None'
        scores_for_collection["Size Diff"] = 'None'

        return scores_for_collection

    # values need to be array for calculations to work, not lists
    center_distances = np.asarray(center_distances)
    overlap_scores = np.asarray(overlap_scores)
    ground_truth_size = np.asarray(ground_truth_size)
    predicted_size = np.asarray(predicted_size)

    # calculate the metrics based on the results for each frame of the sequences in the collection
    scores_for_collection = {}

    scores_for_collection["Samples"] = len(sequences)
    scores_for_collection["Frames"] = frames

    dfun = build_dist_fun(center_distances)
    at20 = dfun(20)
    scores_for_collection["Precision"] = at20

    ofun = build_over_fun(overlap_scores)
    x = np.arange(0., 1.001, 0.001)
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    scores_for_collection["Success"] = auc

    abc = area_between_curves(ground_truth_size, predicted_size)
    scores_for_collection["Size Diff"] = abc

    return scores_for_collection


# build a collection of sequences for each attribute, based on the sequences in tracking dir
def get_attribute_collections(tracking_dir):
    sequnce_dirs = get_tracked_sequences(tracking_dir)

    with open(attributes_path, "r") as stream:
        try:
            tb100 = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    attribute_collection = {"IV": [],
                            "SV": [],
                            "OCC": [],
                            "DEF": [],
                            "MB": [],
                            "FM": [],
                            "IPR": [],
                            "OPR": [],
                            "OV": [],
                            "BC": [],
                            "LR": []}

    # find the attributes of each sequence and extend the collection for that attribute by the name of the sequence
    for sequnce in sequnce_dirs:
        sequence_name = sequnce.split("/")[-1].split("-")[-1]
        for sample in tb100["samples"]:
            if sample["name"] == sequence_name:
                for attribute in sample["attributes"]:
                    attribute_collection[attribute].append(sequnce)

    return attribute_collection


# creates a csv file with the scores for each attribute of the tb100
def create_attribute_results_csv(tracking_dir):

    attribute_collection = get_attribute_collections(tracking_dir)
    attribute_scores = {}

    # get the metric scores for each collection
    for attribute in attribute_collection:
        metrics = get_metrics_for_collections(sequences=attribute_collection[attribute])
        attribute_scores[attribute] = metrics

    approach = get_approach_from_yaml(tracking_dir)

    # get the results for every sequence into on csv
    csv_name = tracking_dir + '/evaluation/' + approach + '_attribute_results.csv'
    with open(csv_name, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=["Attribute",
                                                    "Samples",
                                                    "Frames",
                                                    "Precision",
                                                    "Success",
                                                    "Size Difference"])
        writer.writeheader()

        for attribute in attribute_scores:
            sample_name = attribute
            samples = attribute_scores[attribute]["Samples"]
            frames = attribute_scores[attribute]["Frames"]
            precision = attribute_scores[attribute]["Precision"]
            success = attribute_scores[attribute]["Success"]
            abc = attribute_scores[attribute]["Size Diff"]

            writer.writerow({"Attribute": sample_name,
                             "Samples": samples,
                             "Frames": frames,
                             "Precision": precision,
                             "Success": success,
                             "Size Difference": abc})


# creates a csv file with every score for every sequence within one tracking folder
def create_tracking_results_csv(tracking_dir):
    sequnce_dirs = get_tracked_sequences(tracking_dir)
    approach = get_approach_from_yaml(tracking_dir)

    # get the results for every sequence into on csv
    csv_name = tracking_dir + '/evaluation/' + approach + '_sequence_results.csv'
    with open(csv_name, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=["Sample", "Frames", "Framerate", "Precision", "Success", "Size Diff"])
        writer.writeheader()

        for sequence in sequnce_dirs:
            with open(sequence + "/evaluation.txt", "r") as evaltxt:
                lines = evaltxt.readlines()
                sample_name = lines[1].replace("\n", "").split("=")[1]
                number_frames = lines[9].replace("\n", "").split("=")[1]
                frame_rate = lines[10].replace("\n", "").split("=")[1]
                abc = lines[26].replace("\n", "").split("=")[1]
                precision = lines[27].replace("\n", "").split("=")[1]
                success = lines[29].replace("\n", "").split("=")[1]
                writer.writerow({'Sample': sample_name,
                                 'Frames': number_frames,
                                 'Framerate': frame_rate,
                                 'Precision': precision,
                                 'Success': success,
                                 'Size Diff': abc})


# only creates graphs for the sequences in one tracking folder
def eval_sequences(tracked_sequences):
    # traverse all sub folders and do evaluation for each sequence
    for sequence in tracked_sequences:
        if not os.path.exists(sequence + "/evaluation"):
            os.mkdir(os.path.join(sequence, "evaluation"))
        sequence_result, tracking_evaluation = get_single_sequence_results(sequence)
        create_graphs_for_sequence(sequence_result, sequence)


# first evaluate each sequence, based on that calc metrics for tracking approach
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

    # get the results for every sequence into one csv
    create_tracking_results_csv(tracking_dir)

    # get the results for each attribute into one csv
    create_attribute_results_csv(tracking_dir)


# evaluates every tracking approach in the experiments folder
def eval_all_trackings():
    tracking_folders = get_tracking_folders(results_path)
    for tracking_folder in tracking_folders:
        eval_tracking(tracking_folder)


# create a csv with precision and success for each tracking in experiment folder
def create_opt_csv(experiment_folder):
    trackings = get_tracking_folders(experiment_folder)

    csv_name = experiment_folder + "/parameter_comparision.csv"
    with open(csv_name, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv,
                                fieldnames=["Avg. Success",
                                            "Avg. Precision",
                                            "Total Secs",
                                            "adjust_max_scale_diff",
                                            "adjust_max_scale_diff_after",
                                            "inner_punish_threshold",
                                            "outer_punish_threshold",
                                            "c_number_scales",
                                            "max_scale_difference",
                                            "scale_window_step_size"])
        writer.writeheader()

        for tracking_dir in trackings:
            with open(tracking_dir + "/evaluation.txt", "r") as evaltxt:
                lines = evaltxt.readlines()
                avg_succ = lines[25].replace("\n", "").split("=")[1]
                avg_prec = lines[24].replace("\n", "").split("=")[1]
                total_secs = lines[12].replace("\n", "").split("=")[1]

            if "tracker.yaml" in os.listdir(tracking_dir):
                with open(tracking_dir + "/tracker.yaml", "r") as stream:
                    try:
                        configuration = yaml.safe_load(stream)
                        scale_estimator_conf = configuration["scale_estimator"]
                        adj_max_dif = scale_estimator_conf["adjust_max_scale_diff"]
                        adj_max_dif_after = scale_estimator_conf["adjust_max_scale_diff_after"]
                        inner_thresh = scale_estimator_conf["inner_punish_threshold"]
                        outer_thresh = scale_estimator_conf["outer_punish_threshold"]
                        number_c = scale_estimator_conf["c_number_scales"]
                        max_diff = scale_estimator_conf["max_scale_difference"]
                        window_step_size = scale_estimator_conf["scale_window_step_size"]

                    except yaml.YAMLError as exc:
                        print(exc)

            elif "tracker_2.yaml" in os.listdir(tracking_dir):
                with open(tracking_dir + "/tracker_2.yaml", "r") as stream:
                    try:
                        configuration = yaml.safe_load(stream)
                        scale_estimator_conf = configuration["scale_estimator"]
                        adj_max_dif = scale_estimator_conf["adjust_max_scale_diff"]
                        adj_max_dif_after = scale_estimator_conf["adjust_max_scale_diff_after"]
                        inner_thresh = scale_estimator_conf["inner_punish_threshold"]
                        outer_thresh = scale_estimator_conf["outer_punish_threshold"]
                        number_c = scale_estimator_conf["c_number_scales"]
                        max_diff = scale_estimator_conf["max_scale_difference"]
                        window_step_size = scale_estimator_conf["scale_window_step_size"]

                    except yaml.YAMLError as exc:
                        print(exc)
            else:
                print("no tracker.yaml configuration found")

            writer.writerow({'Avg. Success': avg_succ,
                             'Avg. Precision': avg_prec,
                             'Total Secs': total_secs,
                             'adjust_max_scale_diff': adj_max_dif,
                             'adjust_max_scale_diff_after': adj_max_dif_after,
                             'inner_punish_threshold': inner_thresh,
                             'outer_punish_threshold': outer_thresh,
                             'c_number_scales': number_c,
                             'max_scale_difference': max_diff,
                             'scale_window_step_size': window_step_size})


if __name__ == '__main__':
    # eval_all_trackings()
    create_opt_csv(results_path)
