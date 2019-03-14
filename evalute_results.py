import os
from core.Rect import Rect
import argparse
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import csv
import zipfile36
import yaml

parser = argparse.ArgumentParser(description="Evaluates tracking results. Can be used on either a matlab result folder,"
                                             " a hiob sequence folder, a hiob tracking folder, or a hiob experiment "
                                             "folder containing multiple tracking folders")

parser.add_argument("-ptr", "--pathresults", help="Absolute path to the folder that contains the tracking results, the "
                                                  "script determines what can of folder it is handling and gets the "
                                                  "results accordingly")

parser.add_argument("-ptgt", "--pathgt", help=" Absolute path to the folder containing the sequence zip files, so that "
                                              "the ground truths can be obtained")

parser.add_argument("-pta", "--attributes", help="Absolute path to the TB100 collection file, which contains the "
                                                 "attributes for each sequence. Those are needed to create CSVs based "
                                                 "the attributes of the sequences")

args = parser.parse_args()

results_path = args.pathresults
tb100_gt_path = args.pathgt
tb100_attributes_path = args.attributes


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


# get the gt rects from a tb100 tip file
def get_tb100_gt_rects_from_zip(sequence):
    zip = zipfile36.ZipFile(os.path.join(tb100_gt_path, sequence + ".zip"))
    items_in_archive = zipfile36.ZipFile.namelist(zip)
    gts = []

    if os.path.join(sequence, "groundtruth_rect.txt") in items_in_archive:
        f = zip.open(os.path.join(sequence.split(".")[0], "groundtruth_rect.txt"))
        zip_contents = f.read()
        zip_contents = zip_contents.decode()
        list_of_strings = zip_contents.split("\n")

        for i in range(0, len(list_of_strings)):
            # because the split on \n, last element ist empty string...
            if list_of_strings[i] == '':
                continue

            coord_string = list_of_strings[i]
            if "\t" in coord_string:
                coord_string = coord_string.replace("\t", ",")

            if "\r" in coord_string:
                coord_string = coord_string.replace("\r", "")

            coord_list = [pos_int_form_string(number) for number in coord_string.split(",")]
            try:
                gt_rect = Rect(int(coord_list[0]), int(coord_list[1]), int(coord_list[2]), int(coord_list[3]))
            except ValueError:
                print("here")
            gts.append(gt_rect)

        f.close()

    return gts


# get the prediction rects of a hiob sequence folder
def get_pred_rects_from_sequence(sequence_folder):
    # read the tracking results from the log file
    with open(os.path.join(sequence_folder, str("tracking_log.txt")), mode="r") as f:
        tracking_results = f.readlines()

    preds = []
    for index, line in enumerate(tracking_results):
        working_line = line.replace("\n", "").split(",")
        for i in range(0, len(working_line)):
            try:
                working_line[i] = float(working_line[i])
            except ValueError:
                pass

            if i == 1:
                pred = Rect(working_line[i], working_line[i + 1], working_line[i + 2], working_line[i + 3])
                preds.append(pred)

    return preds


# get all sequence from one tracking folder
def get_sequences(tracking_dir):
    if os.path.isdir(tracking_dir):
        files_in_dir = os.listdir(tracking_dir)
    else:
        # matlab result_file...
        return tracking_dir

    sequences = []
    folder_type = determine_folder_type(tracking_dir)

    for file in files_in_dir:

        if folder_type == "hiob_tracking_folder":
            if os.path.isdir(os.path.join(tracking_dir, file)) and "tracking" in file:
                sequences.append(file)
        elif folder_type == "matlab_tracking_folder":
            if "results.mat" in file:
                sequences.append(file)
        elif folder_type == "hiob_sequence_folder" or folder_type == "matlab_sequence_file":
            sequences.append([tracking_dir])
            break

    return sequences


# get all rects of predictions and gt from each workplace in result dir
def get_all_rects(result_dir):
    current_folder_type = determine_folder_type(result_dir)
    pred_gt_rects = {"preds": [], "gts": []}

    if current_folder_type == "matlab_tracking_folder":
        workplaces = get_saved_workplaces(result_dir)

        for i, workplace in enumerate(workplaces):
            preds, gts = get_rects_from_workplace(workplace)
            pred_gt_rects["preds"].append(preds)
            pred_gt_rects["gts"].append(gts)

    elif current_folder_type == "hiob_sequence_folder":
        sequence_name = result_dir.split("/")[-1].split("-")[-1]
        preds = get_pred_rects_from_sequence(result_dir)
        gts = get_tb100_gt_rects_from_zip(sequence_name)
        pred_gt_rects["preds"].append(preds)
        pred_gt_rects["gts"].append(gts)

    elif current_folder_type == "hiob_tracking_folder":
        sequence_folders = get_sequences(result_dir)

        for sequence_folder in sequence_folders:
            sequence_name = sequence_folder.split("/")[-1].split("-")[-1]
            preds = get_pred_rects_from_sequence(os.path.join(result_dir, sequence_folder))
            gts = get_tb100_gt_rects_from_zip(sequence_name)
            pred_gt_rects["preds"].append(preds)
            pred_gt_rects["gts"].append(gts)

    elif current_folder_type == "matlab_sequnce_file":
        preds, gts = get_rects_from_workplace(result_dir)
        pred_gt_rects["preds"].append(preds)
        pred_gt_rects["gts"].append(gts)

    # to get it into two proper lists, instead of two lists of lists
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
def get_metrics_from_rects(result_folder, all_preds=None, all_gts=None):
    print("getting metrics for {0}".format(result_folder))

    if all_preds is None and all_gts is None:
        all_preds, all_gts = get_all_rects(result_folder)

    if result_folder != "no_folder_needed":
        folder_type = determine_folder_type(result_folder)
    center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)

    # calculate the metrics based on the results for each frame of the sequences in the collection
    scores_for_rects = {}

    if result_folder != "no_folder_needed":
        sequences = get_sequences(result_folder)
        scores_for_rects["Samples"] = len(sequences)
    else:
        scores_for_rects["Samples"] = "value wont be used"

    scores_for_rects["Frames"] = frames

    dfun = build_dist_fun(center_distances)
    at20 = dfun(20)
    scores_for_rects["Total Precision"] = at20

    ofun = build_over_fun(overlap_scores)
    x = np.arange(0., 1.001, 0.001)
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    scores_for_rects["Total Success"] = auc

    if result_folder != "no_folder_needed":
        if folder_type == "hiob_tracking_folder" or folder_type == "matlab_tracking_folder":
            prec_sum = 0
            succ_sum = 0

            for sequence in sequences:
                sequence_metrics = get_metrics_from_rects(os.path.join(result_folder, sequence))
                prec_sum += sequence_metrics["Total Precision"]
                succ_sum += sequence_metrics["Total Success"]

            scores_for_rects["Avg. Precision"] = prec_sum / scores_for_rects["Samples"]
            scores_for_rects["Avg. Success"] = succ_sum / scores_for_rects["Samples"]

    normalized_size_score, normalized_gt_size_scores = normalize_size_datapoints(size_scores, gt_size_scores)
    abc = area_between_curves(normalized_gt_size_scores, normalized_size_score)
    scores_for_rects["Size Score"] = abc

    return scores_for_rects


# create a csv for the scores
def create_avg_score_csv(results_folder, eval_folder):
    score_dict = get_metrics_from_rects(results_folder)
    scores = score_dict.keys()

    eval_path = os.path.join(results_folder, "evaluation")
    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)

    full_eval_path = os.path.join(eval_path, eval_folder)
    if not os.path.isdir(full_eval_path):
        os.mkdir(full_eval_path)

    out_csv = os.path.join(full_eval_path, eval_folder + ".csv")
    with open(out_csv, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=scores)
        writer.writeheader()

        writer.writerow(score_dict)


# create a csv for the scores on each sequence
def create_sequence_score_csv(result_folder, eval_folder):
    sequences = get_sequences(result_folder)

    eval_path = os.path.join(result_folder, "evaluation")
    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)

    full_eval_path = os.path.join(eval_path, eval_folder)
    if not os.path.isdir(full_eval_path):
        os.mkdir(full_eval_path)

    csv_fields = ["Sample", "Frames", "Precision", "Success", "Size Score"]

    out_csv = os.path.join(full_eval_path, eval_folder + ".csv")
    with open(out_csv, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=csv_fields)
        writer.writeheader()

        for sequence in sequences:
            score_dict = get_metrics_from_rects(os.path.join(results_path, sequence))
            sequence_name = sequence.split("-")[-1]
            writer.writerow({
                 "Sample": sequence_name,
                 "Frames": score_dict["Frames"],
                 "Precision": score_dict["Total Precision"],
                 "Success": score_dict["Total Success"],
                 "Size Score": score_dict["Size Score"]
            })


# create a csv for the attribute collections from one tracking
def create_attribute_score_csv(result_folder, eval_folder):
    sequences = get_sequences(result_folder)

    eval_path = os.path.join(result_folder, "evaluation")
    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)

    full_eval_path = os.path.join(eval_path, eval_folder)
    if not os.path.isdir(full_eval_path):
        os.mkdir(full_eval_path)

    csv_fields = ["Attribute", "Samples", "Frames", "Precision", "Success", "Size Score"]

    out_csv = os.path.join(full_eval_path, eval_folder + ".csv")
    with open(out_csv, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=csv_fields)
        writer.writeheader()

        attribute_collection = get_attribute_collections(result_folder)

        for attribute in attribute_collection:
            attribute_sequences = attribute_collection[attribute]
            if attribute_sequences != []:
                all_preds = []
                all_gts = []
                for attribute_sequence in attribute_sequences:
                    sequence_preds, sequence_gts = get_all_rects(os.path.join(result_folder, attribute_sequence))
                    all_preds.append(sequence_preds)
                    all_gts.append(sequence_gts)

                # flatten lists
                flat_preds = []
                flat_gts = []
                for sublist in all_preds:
                    for item in sublist:
                        flat_preds.append(item)

                for sublist in all_gts:
                    for item in sublist:
                        flat_gts.append(item)

                score_dict = get_metrics_from_rects("no_folder_needed",
                                                    all_preds=flat_preds, all_gts=flat_gts)

                writer.writerow({
                    "Attribute": attribute,
                    "Samples": len(attribute_sequences),
                    "Frames": score_dict["Frames"],
                    "Precision": score_dict["Total Precision"],
                    "Success": score_dict["Total Success"],
                    "Size Score": score_dict["Size Score"]
                })


# create a csv comparring the trackings in one csv folder (opt)
def create_opt_csv(experiment_folder, eval_folder):

    filename = experiment_folder.split("/")[-1] + "_opt.csv"
    csv_name = os.path.join(experiment_folder, filename)

    trackings = get_tracking_folder(experiment_folder)

    approach = get_approach_from_yaml(trackings[0])

    # create head with parameters
    if approach == "Candidates dynamic" or approach == "Candidates static":
        with open(csv_name, 'w', newline='') as outcsv:
            writer = csv.DictWriter(outcsv,
                                    fieldnames=["Avg. Success",
                                                "Avg. Precision",
                                                "Framerate",
                                                "SE Framerate"
                                                "adjust_max_scale_diff",
                                                "adjust_max_scale_diff_after",
                                                "inner_punish_threshold",
                                                "outer_punish_threshold",
                                                "c_number_scales",
                                                "max_scale_difference",
                                                "scale_window_step_size",
                                                "c_scale_factor"])
            writer.writeheader()

    elif approach == "DSST dynamic" or approach == "DSST static":
        with open(csv_name, 'w', newline='') as outcsv:
            writer = csv.DictWriter(outcsv,
                                    fieldnames=["Avg. Success",
                                                "Avg. Precision",
                                                "Framerate",
                                                "SE Framerate",
                                                "dsst_number_scales",
                                                "learning_rate",
                                                "d_scale_factor",
                                                "scale_model_max",
                                                "scale_model_size",
                                                "scale_sigma_factor"])

            writer.writeheader()

            for tracking_dir in trackings:
                with open(tracking_dir + "/evaluation.txt", "r") as evaltxt:
                    lines = evaltxt.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        key_val = line.split("=")
                        if key_val[0] == "average_precision_rating":
                            avg_prec = key_val[1]
                        elif key_val[0] == "average_success_rating":
                            avg_succ = key_val[1]
                        elif key_val[0] == "frame_rate":
                            frame_rate = key_val[1]
                        elif key_val[0] == "se_frame_rate":
                            se_framerate = key_val[1]

                if "tracker.yaml" in os.listdir(tracking_dir):
                    with open(tracking_dir + "/tracker.yaml", "r") as stream:
                        if approach == "Candidates dynamic" or approach == "Candidates static":
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
                                c_scale_factor = scale_estimator_conf["c_scale_factor"]

                            except yaml.YAMLError as exc:
                                print(exc)

                            writer.writerow({'Avg. Success': avg_succ,
                                             'Avg. Precision': avg_prec,
                                             'Framerate': frame_rate,
                                             'SE Framerate': se_framerate,
                                             'adjust_max_scale_diff': adj_max_dif,
                                             'adjust_max_scale_diff_after': adj_max_dif_after,
                                             'inner_punish_threshold': inner_thresh,
                                             'outer_punish_threshold': outer_thresh,
                                             'c_number_scales': number_c,
                                             'max_scale_difference': max_diff,
                                             'scale_window_step_size': window_step_size,
                                             'c_scale_factor': c_scale_factor})

                        elif approach == "DSST dynamic" or approach == "DSST static":
                            try:
                                configuration = yaml.safe_load(stream)
                                scale_estimator_conf = configuration["scale_estimator"]
                                dsst_number_scales = scale_estimator_conf["dsst_number_scales"]
                                learning_rate = scale_estimator_conf["learning_rate"]
                                d_scale_factor = scale_estimator_conf["scale_factor"]
                                scale_model_max = scale_estimator_conf["scale_model_max"]
                                scale_model_size = scale_estimator_conf["scale_model_size"]
                                scale_sigma_factor = scale_estimator_conf["scale_sigma_factor"]

                            except yaml.YAMLError as exc:
                                print(exc)

                            writer.writerow({'Avg. Success': avg_succ,
                                             'Avg. Precision': avg_prec,
                                             'Framerate': frame_rate,
                                             'SE Framerate': se_framerate,
                                             'dsst_number_scales': dsst_number_scales,
                                             'learning_rate': learning_rate,
                                             'd_scale_factor': d_scale_factor,
                                             'scale_model_max': scale_model_max,
                                             'scale_model_size': scale_model_size,
                                             'scale_sigma_factor': scale_sigma_factor})


# create the graphs from rects
def create_graphs_from_rects(result_folder, eval_folder):
    all_preds, all_gts = get_all_rects(result_folder)

    center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)

    eval_path = os.path.join(result_folder, "evaluation")
    eval_path = os.path.join(eval_path, eval_folder)

    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)

    # precision plot
    dfun = build_dist_fun(center_distances)
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
    plt.xlim(xmin=0, xmax=50)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    # success plot
    ofun = build_over_fun(overlap_scores)
    figure_file2 = os.path.join(eval_path, 'success_plot.svg')
    figure_file3 = os.path.join(eval_path, 'success_plot.pdf')
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
    figure_file2 = os.path.join(eval_path, 'size_over_time.svg')
    figure_file3 = os.path.join(eval_path, 'size_over_time.pdf')
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


# create graphs and metrics for a specific set of results
def create_graphs_metrics_for_set(set_of_results, set_name):
    create_avg_score_csv(set_of_results, set_name)
    create_graphs_from_rects(set_of_results, set_name)


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

    if min_val == max_val:
        min_val = 1

    # normalize each size score based on min max
    for i in range(len(pred_size_scores)):
        pred_size_scores[i] = (pred_size_scores[i] - min_val) / (max_val - min_val + 0.0025) * 100
        gt_size_scores[i] = (gt_size_scores[i] - min_val) / (max_val - min_val + 0.0025) * 100

    return pred_size_scores, gt_size_scores


# get basic scores from rects
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


# determine the kind of a folder, eg. hiob sequence, hiob tracking with multiple sequences, or matlab results folder
def determine_folder_type(folder):
    found_mat_files = False
    found_tracker_config = False
    found_tracking_folder = False
    found_positions_txt = False
    not_a_folder = False
    found_hiob_execution = False

    if os.path.isdir(folder):
        files_in_dir = os.listdir(folder)
    else:
        not_a_folder = True

    if os.path.isdir(folder):
        for file in files_in_dir:
            if ".mat" in file:
                found_mat_files = True
            elif file == "tracker.yaml":
                found_tracker_config = True
            elif "tracking" in file and os.path.isdir(os.path.join(folder, file)):
                found_tracking_folder = True
            elif folder.split("/")[-1].split("-")[-1] in file and ".txt" in file:
                found_positions_txt = True
            elif "hiob-execution" in file:
                found_hiob_execution = True


    if found_positions_txt and not (found_mat_files or found_tracker_config or found_tracking_folder or found_hiob_execution):
        return "hiob_sequence_folder"
    elif found_tracker_config and found_tracking_folder and not (found_mat_files or found_positions_txt or found_hiob_execution):
        return "hiob_tracking_folder"
    elif found_mat_files and not (found_positions_txt or found_tracking_folder or found_tracker_config or found_hiob_execution):
        return "matlab_tracking_folder"
    elif not_a_folder and "results.mat" in folder:
        return "matlab_sequnce_file"
    elif found_hiob_execution and not (found_positions_txt or found_tracker_config or found_tracking_folder or found_mat_files):
        return "multiple_hiob_executions"


# return a positive int from a negative one (HIOB handles negative gts like positive ones...)
def pos_int_form_string(string_number):
    if int(string_number) <= 0:
        return int(string_number) * -1
    else:
        return int(string_number)


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

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "custom_dsst" \
                and not scale_estimator_conf["d_change_aspect_ratio"]:
            algorithm = "DSST static"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "custom_dsst"\
                and scale_estimator_conf["d_change_aspect_ratio"]:
            algorithm = "DSST dynamic"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "candidates" \
                and scale_estimator_conf["c_change_aspect_ratio"]:
            algorithm = "Candidates dynamic"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "candidates" \
                and not scale_estimator_conf["c_change_aspect_ratio"]:
            algorithm = "Candidates static"

        return algorithm


# get attribute collections for a tracking
def get_attribute_collections(tracking_dir):
    sequences = get_sequences(tracking_dir)

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

    with open(tb100_attributes_path, "r") as stream:
        try:
            tb100 = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

        for sequence in sequences:
            if ".mat" in sequence:
                sequence_name = sequence.split("_")[0]
            else:
                sequence_name = sequence.split("/")[-1].split("-")[-1]
            for sample in tb100["samples"]:
                if sample["name"] == sequence_name:
                    for attribute in sample["attributes"]:
                        attribute_collection[attribute].append(sequence)

    return attribute_collection


# get the tracking folders of one experiment
def get_tracking_folder(experiment_dir):
    if os.path.isdir(experiment_dir):
        files_in_dir = os.listdir(experiment_dir)
    else:
        # matlab result_file...
        return experiment_dir

    trackings = []

    for item in files_in_dir:
        if os.path.isdir(os.path.join(experiment_dir, item)):
            tracking_items = os.listdir(os.path.join(experiment_dir, item))

            if "tracker.yaml"in tracking_items and "trackings.txt" in tracking_items \
                    and "evaluation.txt" in tracking_items:
                trackings.append(os.path.join(experiment_dir, item))

    return trackings


if __name__ == "__main__":
    folder_type = determine_folder_type(results_path)
    # just one sequence folder from one tracking folder
    if folder_type == "hiob_sequence_folder":
        create_graphs_metrics_for_set(results_path, "avg_full_set")

    # one hiob execution containing multiple sequence folders
    elif folder_type == "hiob_tracking_folder":
        create_graphs_metrics_for_set(results_path, "avg_full_set")
        create_sequence_score_csv(results_path, "sequence_results")
        create_attribute_score_csv(results_path, "attribute_results")

    # matlab tracking folder, containing the saved workplaces from each sequence
    elif folder_type == "matlab_tracking_folder":
        create_graphs_metrics_for_set(results_path, "avg_full_set")
        create_sequence_score_csv(results_path, "sequence_results")
        create_attribute_score_csv(results_path, "attribute_results")

    # experiment folder containing multiple hiob executions, h_opt for example
    elif folder_type == "multiple_hiob_executions":
        create_opt_csv(results_path, "opt")
