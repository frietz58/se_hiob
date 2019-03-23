import os
from core.Rect import Rect
import argparse
import scipy.io
import matplotlib
matplotlib.use('Agg')  # for plot when display is undefined
import matplotlib.pyplot as plt
import numpy as np
import csv
import zipfile36
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

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

parser.add_argument("-m", "--mode", help="To differentiate between an opt folder with multiple hiob executions and an "
                                         "experiment folder with multiple hiob executions")

args = parser.parse_args()

results_path = args.pathresults
tb100_gt_path = args.pathgt
tb100_attributes_path = args.attributes
mode = args.mode

# ======= csv names =======
csv_avg_success = "Avg. Success"
csv_avg_precision = "Avg. Precision"
csv_avg_fps = "Avg. Frame rate"
csv_avg_se_fps = "Avg. SE Frame rate"
csv_avg_ss = "Avg. Size Score"
csv_total_precision = "Total Precision"
csv_total_success = "Total Success"


# ================================= GET FUNCTIONS =================================
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
def get_sequences(tracking_dir, came_from_experiment=False):
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
                if came_from_experiment:
                    sequences.append(os.path.join(tracking_dir, file))
                else:
                    sequences.append(file)
        elif folder_type == "matlab_tracking_folder":
            if "results.mat" in file:
                sequences.append(file)
        elif folder_type == "hiob_sequence_folder" or folder_type == "matlab_sequence_file":
            sequences.append([tracking_dir])
            break
        elif folder_type == "multiple_hiob_executions":
            if os.path.isdir(os.path.join(tracking_dir, file)) and "hiob-execution" in file:
                sequences.append(get_sequences(os.path.join(tracking_dir, file), came_from_experiment=True))

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

    elif current_folder_type == "multiple_hiob_executions":
        hiob_executions = get_tracking_folders(result_dir)

        for hiob_execution in hiob_executions:
            sequence_folders = get_sequences(hiob_execution)
            for sequence in sequence_folders:
                sequence_folder = os.path.join(hiob_execution, sequence)
                sequence_name = os.path.basename(sequence_folder).split("-")[-1]
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


# get the evaluation values from each tracking in an excperiment
def get_avg_results_from_experiment(experiment_folder):
    hiob_executions = get_tracking_folders(experiment_folder)
    # get the raw values from evaluation.txt
    experiment_results = {}
    for hiob_execution in hiob_executions:
        execution_results = {}
        sequence_folders = get_sequences(hiob_execution)

        for sequence in sequence_folders:
            sequence_folder = os.path.join(hiob_execution, sequence)
            sequence_name = os.path.basename(sequence_folder)
            with open(os.path.join(sequence_folder, "evaluation.txt"), "r") as evaltxt:
                sequence_results = {}
                lines = evaltxt.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    key_val = line.split("=")
                    sequence_results[str(key_val[0])] = key_val[1]

            execution_results[sequence_name] = sequence_results
        experiment_results[hiob_execution] = execution_results

    # calculate average metrics each sequence sequence
    sequence_result_collection = {}
    for execution in experiment_results.values():
        for sequence in execution.values():
            if not str(sequence["sample_name"]) in sequence_result_collection.keys():
                sequence_result_collection[str(sequence["sample_name"])] = []
            sequence_result_collection[str(sequence["sample_name"])].append(
                {"frame_rate": sequence["frame_rate"],
                 "frames": sequence["sample_frames"],
                 "se_frame_rate": sequence["se_frame_rate"],
                 "failure_percentage": sequence["failurePercentage"],
                 "size_score": sequence["area_between_size_curves"],
                 "success": sequence["success_rating"],
                 "precision": sequence["precision_rating"],
                 "updates": sequence["updates_total"]})

    # create csv with the average values for each sequence
    print("creating average over sequences scv")
    eval_folder = "evaluation"
    eval_path = os.path.join(experiment_folder, eval_folder)
    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)

    out_csv = os.path.join(eval_path, "sequence_averages.csv")
    with open(out_csv, 'w', newline='') as outcsv:
        csv_fields = ["Sample", "Frames", "Precision", "Success", "Size Score", "Fail %", "Updates"]
        writer = csv.DictWriter(outcsv, fieldnames=csv_fields)
        writer.writeheader()

        rows = []
        for sequence_result in sequence_result_collection:
            prec_ratings = []
            succ_ratings = []
            ss_ratings = []
            fail_percentages = []
            update_totals = []
            sample = str(sequence_result)
            frames = sequence_result_collection[sequence_result][0]["frames"]
            for result in sequence_result_collection[sequence_result]:
                prec_ratings.append(float(result["precision"]))
                succ_ratings.append(float(result["success"]))
                ss_ratings.append(float(result["size_score"]))
                fail_percentages.append(float(result["failure_percentage"]))
                update_totals.append(float(result["updates"]))
            avg_prec = np.around(sum(prec_ratings) / len(sequence_result_collection[sequence_result]), decimals=3)
            avg_succ = np.around(sum(succ_ratings) / len(sequence_result_collection[sequence_result]), decimals=3)
            avg_ss_rating = np.around(sum(ss_ratings) / len(sequence_result_collection[sequence_result]), decimals=3)
            avg_fail_percentages = np.around(sum(fail_percentages) / len(sequence_result_collection[sequence_result]),
                                             decimals=3)
            avg_updates = np.around(sum(update_totals) / len(sequence_result_collection[sequence_result]), decimals=3)

            row_dict = {
                "Sample": sample,
                "Frames": frames,
                "Precision": avg_prec,
                "Success": avg_succ,
                "Size Score": avg_ss_rating,
                "Fail %": avg_fail_percentages,
                "Updates": avg_updates}

            writer.writerow(row_dict)
            rows.append(row_dict)

    # create csv with the average values for each attribute
    print("creating average over attributes csv")
    out_csv = os.path.join(eval_path, "attribute_averages.csv")
    csv_fields = ["Attribute", "Samples", "Frames", csv_avg_precision, csv_avg_success, csv_avg_ss]
    with open(out_csv, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=csv_fields)
        writer.writeheader()
        attribute_sequences = get_attribute_collections(experiment_folder)
        unique_samples = []
        for attribute in attribute_sequences:
            print("attribute: {0}".format(attribute))
            print("loading all rects for sequences with same attribute")
            specific_attribute_sequences = attribute_sequences[attribute]
            if attribute_sequences != []:
                all_preds = []
                all_gts = []

            prec_sum = 0
            succ_sum = 0
            size_score_sum = 0

            for sequence in specific_attribute_sequences:
                if sequence.split("/")[-1].split("-")[-1] not in unique_samples:
                    unique_samples.append(sequence.split("/")[-1].split("-")[-1])

                sequence_preds, sequence_gts = get_all_rects(sequence)
                sequence_results = get_metrics_from_rects("attribute", all_preds=sequence_preds, all_gts=sequence_gts)
                prec_sum += sequence_results["Total Precision"]
                succ_sum += sequence_results["Total Success"]
                size_score_sum += sequence_results["Size Score"]

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

            score_dict = get_metrics_from_rects("attribute", all_preds=flat_preds, all_gts=flat_gts)

            writer.writerow({
                "Attribute": attribute,
                "Samples": len(attribute_sequences[attribute]),
                "Frames": score_dict["Frames"],
                csv_avg_precision: np.around(score_dict["Total Precision"], decimals=3),
                csv_avg_success: np.around(score_dict["Total Success"], decimals=3),
                csv_avg_ss: score_dict["Size Score"]
            })

            # writer.writerow({
            #     "Attribute": attribute,
            #     "Samples": len(specific_attribute_sequences),
            #     "Frames": score_dict["Frames"],
            #     csv_avg_precision: np.around(prec_sum / len(specific_attribute_sequences), decimals=3),
            #     csv_avg_success: np.around(succ_sum / len(specific_attribute_sequences), decimals=3),
            #     csv_avg_ss: np.around(size_score_sum / len(specific_attribute_sequences), decimals=3)
            # })



        # get the average values of the rows:
        # final_precs = []
        # final_succs = []
        # final_ss = []
        # final_fails = []
        # final_updates = []
        # samples = 0
        # frames = 0
        # for row in rows:
        #     final_precs.append(row[csv_avg_precision])
        #     final_succs.append(row[csv_avg_success])
        #     final_ss.append(row["Avg. Size Score"])
        #     final_fails.append(row["Avg. Failure Percentage"])
        #     final_updates.append(row["Avg. Updates"])
        #     samples += 1
        #     frames += int(row["Frames"])
        # final_avg_precs = np.around(np.sum(final_precs) / samples, decimals=3)
        # final_avg_succs = np.around(np.sum(final_succs) / samples, decimals=3)
        # final_avg_ss = np.around(np.sum(final_ss) / samples, decimals=3)
        # final_avg_fails = np.around(np.sum(final_fails) / samples, decimals=3)
        # final_avg_updates = np.around(np.sum(final_updates) / samples, decimals=3)
        #
        # summarizing_row = {
        #     "Sample": samples,
        #     "Frames": frames,
        #     "Precision": final_avg_precs,
        #     "Success": final_avg_succs,
        #     "Size Score": final_avg_ss,
        #     "Fail %": final_avg_fails,
        #     "Updates": final_avg_updates}
        #
        # writer.writerow(summarizing_row)

    return None


# get metric from rects
def get_metrics_from_rects(result_folder, all_preds=None, all_gts=None, sequences=None):
    print("getting metrics for {0}".format(result_folder))

    if all_preds is None and all_gts is None:
        all_preds, all_gts = get_all_rects(result_folder)

    if result_folder != "attribute":
        folder_type = determine_folder_type(result_folder)
    center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)

    # calculate the metrics based on the results for each frame of the sequences in the collection
    scores_for_rects = {}

    if result_folder != "attribute":
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

    if result_folder != "attribute":
        if folder_type == "hiob_tracking_folder" or folder_type == "matlab_tracking_folder":
            prec_sum = 0
            succ_sum = 0

            for sequence in sequences:
                sequence_metrics = get_metrics_from_rects(os.path.join(result_folder, sequence))
                prec_sum += sequence_metrics["Total Precision"]
                succ_sum += sequence_metrics["Total Success"]

            scores_for_rects[csv_avg_precision] = prec_sum / scores_for_rects["Samples"]
            scores_for_rects[csv_avg_success] = succ_sum / scores_for_rects["Samples"]

    normalized_size_score, normalized_gt_size_scores = normalize_size_datapoints(size_scores, gt_size_scores)
    abc = area_between_curves(normalized_gt_size_scores, normalized_size_score)
    scores_for_rects["Size Score"] = abc

    return scores_for_rects


# ================================= CREATE FUNCTIONS =================================
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
                "Precision": np.around(score_dict["Total Precision"], decimals=3),
                "Success": np.around(score_dict["Total Success"], decimals=3),
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

    csv_fields = ["Attribute", "Samples", "Frames", csv_total_precision, csv_total_success, csv_avg_ss]

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

                score_dict = get_metrics_from_rects("attribute",
                                                    all_preds=flat_preds, all_gts=flat_gts)

                writer.writerow({
                    "Attribute": attribute,
                    "Samples": len(attribute_sequences),
                    "Frames": score_dict["Frames"],
                    csv_total_precision: np.around(score_dict["Total Precision"], decimals=3),
                    csv_total_success: np.around(score_dict["Total Success"], decimals=3),
                    csv_avg_ss: score_dict["Size Score"]
                })


# create a csv comparing the trackings in one csv folder (opt)
def create_opt_csv(experiment_folder, eval_folder):
    filename = os.path.basename(experiment_folder)
    csv_name = os.path.join(experiment_folder, filename + "_opt.csv")

    trackings = get_tracking_folders(experiment_folder)

    approach = get_approach_from_yaml(trackings[0])

    # create head with parameters
    if approach == "Candidates dynamic" or approach == "Candidates static":
        # find dirs with same parameter value
        same_parameter_value_collection = get_same_parameter_values(trackings)
        changing_parameter = same_parameter_value_collection[list(same_parameter_value_collection.keys())[0]][0][
            "parameter"]

        with open(csv_name, "w", newline='') as outcsv:
            fieldnames = [csv_avg_success,
                          csv_avg_precision,
                          csv_avg_fps,
                          csv_avg_se_fps,
                          changing_parameter]

            writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
            writer.writeheader()

            # get the average values for the same parameter value for a row in the opt csv
            avg_succs = []
            avg_precs = []
            se_framerates = []
            framerates = []
            for value in same_parameter_value_collection.values():
                # get value for one row
                for single_tracking in value:
                    with open(single_tracking["tracking"] + "/evaluation.txt", "r") as evaltxt:
                        lines = evaltxt.readlines()
                        for line in lines:
                            line = line.replace("\n", "")
                            key_val = line.split("=")
                            if key_val[0] == "average_precision_rating":
                                avg_precs.append(float(key_val[1]))
                            elif key_val[0] == "average_success_rating":
                                avg_succs.append(float(key_val[1]))
                            elif key_val[0] == "frame_rate":
                                framerates.append(float(key_val[1]))
                            elif key_val[0] == "se_frame_rate":
                                se_framerates.append(float(key_val[1]))

                final_avg_succ = np.around(sum(avg_succs) / len(avg_succs), decimals=3)
                final_avg_precc = np.around(sum(avg_precs) / len(avg_precs), decimals=3)
                final_framerate = np.around(sum(framerates) / len(framerates), decimals=3)
                final_se_framerate = np.around(sum(se_framerates) / len(se_framerates), decimals=3)

                writer.writerow({csv_avg_success: final_avg_succ,
                                 csv_avg_precision: final_avg_precc,
                                 csv_avg_fps: final_framerate,
                                 csv_avg_se_fps: final_se_framerate,
                                 changing_parameter: value[0]["value"]})

    elif approach == "DSST dynamic" or approach == "DSST static":

        # find dirs with same parameter value
        same_parameter_value_collection = get_same_parameter_values(trackings)
        changing_parameter = same_parameter_value_collection[list(same_parameter_value_collection.keys())[0]][0][
            "parameter"]

        with open(csv_name, "w", newline='') as outcsv:
            fieldnames = [csv_avg_success,
                          csv_avg_precision,
                          csv_avg_fps,
                          csv_avg_se_fps,
                          changing_parameter]

            writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
            writer.writeheader()

            # get the average values for the same parameter value for a row in the opt csv
            avg_succs = []
            avg_precs = []
            se_framerates = []
            framerates = []
            for value in same_parameter_value_collection.values():
                # get value for one row
                for single_tracking in value:
                    with open(single_tracking["tracking"] + "/evaluation.txt", "r") as evaltxt:
                        lines = evaltxt.readlines()
                        for line in lines:
                            line = line.replace("\n", "")
                            key_val = line.split("=")
                            if key_val[0] == "average_precision_rating":
                                avg_precs.append(float(key_val[1]))
                            elif key_val[0] == "average_success_rating":
                                avg_succs.append(float(key_val[1]))
                            elif key_val[0] == "frame_rate":
                                framerates.append(float(key_val[1]))
                            elif key_val[0] == "se_frame_rate":
                                se_framerates.append(float(key_val[1]))

                final_avg_succ = np.around(sum(avg_succs) / len(avg_succs), decimals=3)
                final_avg_precc = np.around(sum(avg_precs) / len(avg_precs), decimals=3)
                final_framerate = np.around(sum(framerates) / len(framerates), decimals=3)
                final_se_framerate = np.around(sum(se_framerates) / len(se_framerates), decimals=3)

                writer.writerow({csv_avg_success: final_avg_succ,
                                 csv_avg_precision: final_avg_precc,
                                 csv_avg_fps: final_framerate,
                                 csv_avg_se_fps: final_se_framerate,
                                 changing_parameter: value[0]["value"]})

    # sort created csv by parameter and override old
    df = pd.read_csv(csv_name)
    cols = list(df)
    sorted = df.sort_values(cols[-1])
    sorted.to_csv(csv_name, index=False)

    return None


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


# create the graphs for the opt plotting parameter value vs x
def create_graphs_from_opt_csv(obt_folder):
    # obt folder are named like the parameter that is obtimized, this:
    parameter_name = obt_folder.split("/")[-1]

    if os.path.isdir(obt_folder):
        files_in_dir = os.listdir(obt_folder)

    csv_name = parameter_name + "_opt.csv"

    if csv_name in files_in_dir:
        df = pd.read_csv(os.path.join(obt_folder, csv_name))
        sorted_df = df.sort_values(parameter_name)

        # ================ PARAMETER vs METRIC ================
        success = sorted_df[csv_avg_success]
        precision = sorted_df[csv_avg_precision]


        ind = np.arange(len(success))  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()

        rects1 = ax.bar(ind, success, width, color='#648fff')
        rects2 = ax.bar(ind + width, precision, width, color='#ffb000')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Scores')
        ax.set_title('Avg. Precision and Success over parameter values')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(sorted_df[parameter_name])

        ax.legend((rects1[0], rects2[0]), ('Avg. Success', 'Avg. Precision'))
        ax.set_ylim(0, 1)

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        #autolabel(rects1)
        #autolabel(rects2)

        # plt.show()
        figure_file3 = os.path.join(obt_folder, 'parameter_vs_metrics.pdf')
        plt.savefig(figure_file3)

        # ================ PARAMETER vs FRAMERATE ================
        framerate = sorted_df[csv_avg_fps]
        se_framerate = sorted_df[csv_avg_se_fps]

        ind = np.arange(len(framerate))  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()

        rects1 = ax.bar(ind, framerate, width, color='#785ef0')
        rects2 = ax.bar(ind + width, se_framerate, width, color='#fe6100')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Frame-rate')
        ax.set_title('Avg. FPS hiob and avg. FPS  of SE module over parameter values')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(sorted_df[parameter_name])

        ax.legend((rects1[0], rects2[0]), ('Avg. Frame-rate', 'Avg. SE Frame-rate'))
        ax.set_ylim(0, max(se_framerate) + 5)

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        # autolabel(rects1)
        # autolabel(rects2)

        # plt.show()
        figure_file3 = os.path.join(obt_folder, 'parameter_vs_framerate.pdf')
        plt.savefig(figure_file3)



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

    if found_positions_txt and not (
            found_mat_files or found_tracker_config or found_tracking_folder or found_hiob_execution):
        return "hiob_sequence_folder"
    elif found_tracker_config and found_tracking_folder and not (
            found_mat_files or found_positions_txt or found_hiob_execution):
        return "hiob_tracking_folder"
    elif found_mat_files and not (
            found_positions_txt or found_tracking_folder or found_tracker_config or found_hiob_execution):
        return "matlab_tracking_folder"
    elif not_a_folder and "results.mat" in folder:
        return "matlab_sequnce_file"
    elif found_hiob_execution and not (
            found_positions_txt or found_tracker_config or found_tracking_folder or found_mat_files):
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
                and scale_estimator_conf["approach"] == "custom_dsst" \
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

        if determine_folder_type(tracking_dir) == "multiple_hiob_executions":
            flat_list = [item for sublist in sequences for item in sublist]
        else:
            flat_list = sequences
        for sequence in flat_list:
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
def get_tracking_folders(experiment_dir):
    if os.path.isdir(experiment_dir):
        files_in_dir = os.listdir(experiment_dir)
    else:
        # matlab result_file...
        return experiment_dir

    trackings = []

    for item in files_in_dir:
        if os.path.isdir(os.path.join(experiment_dir, item)):
            tracking_items = os.listdir(os.path.join(experiment_dir, item))

            if "tracker.yaml" in tracking_items and "trackings.txt" in tracking_items \
                    and "evaluation.txt" in tracking_items:
                trackings.append(os.path.join(experiment_dir, item))

    return trackings


def get_same_parameter_values(trackings):
    changing_parameter_values = {}
    tracker_configs = []

    # load all tracking_configurations
    print("loading all tracker configurations...")
    for tracking in trackings:
        with open(tracking + "/tracker.yaml", "r") as stream:
            try:
                configuration = yaml.safe_load(stream)
                scale_estimator_conf = configuration["scale_estimator"]
                tracker_configs.append({"tracking": tracking, "conf": scale_estimator_conf})
            except yaml.YAMLError as exc:
                print(exc)

    # find parameter that is changed in the current optimization folder
    all_parameter_values = {}
    for parameter in tracker_configs[0]["conf"].keys():
        # all_parameter_values[str(parameter)] = {"tracking": tracking, "parameter": parameter}
        all_parameter_values[str(parameter)] = []
        for curr_dict in tracker_configs:
            try:
                all_parameter_values[str(parameter)].append(
                    {"tracking": curr_dict["tracking"], "parameter": parameter, "value": curr_dict["conf"][parameter]})
            # all_parameter_values[str(parameter)]["value"] = curr_dict["conf"][parameter]
            except KeyError:
                if parameter == "hog_cell_size" or parameter == "hog_block_norm_size":
                    pass

    print("finding tracker configuration which have the same value for the optimization parameter...")
    # if the parameter has different values, it is a parameter changed in opt (hog opt changes multiple parameters)
    for parameter in all_parameter_values:
        # for special case adjust_max_scale_diff_after, which has different values but no impact
        if parameter == 'adjust_max_scale_diff':
            adjust_scale_diff_values = [curr_dict["value"] for curr_dict in all_parameter_values[parameter]]
        parameter_values = [curr_dict["value"] for curr_dict in all_parameter_values[parameter]]
        if not only_item_in_list(all_parameter_values[parameter][0]["value"], parameter_values):
            changing_parameter_values[str(parameter)] = all_parameter_values[parameter]

    if 'adjust_max_scale_diff_after' in changing_parameter_values and only_item_in_list(False,
                                                                                        adjust_scale_diff_values):
        del changing_parameter_values["adjust_max_scale_diff_after"]

    if "hog_cell_size" in changing_parameter_values:
        hog_cell_vals = []
        for tracking in changing_parameter_values["hog_cell_size"]:
            if tracking["value"] not in hog_cell_vals:
                hog_cell_vals.append(tracking["value"])
        if len(hog_cell_vals) == 2:
            del changing_parameter_values["hog_cell_size"]

    if 'hog_block_norm_size' in changing_parameter_values:
        del changing_parameter_values["hog_block_norm_size"]

    # get trackings where value is the same to average over those
    parameter_value_trackings = {}
    for item in changing_parameter_values:
        for entry in changing_parameter_values[item]:
            if str(entry["value"]) not in parameter_value_trackings.keys():
                parameter_value_trackings[str(entry["value"])] = []
            parameter_value_trackings[str(entry["value"])].append(entry)

    # for each value of the changing parameter return the collection of trackings
    return parameter_value_trackings


def only_item_in_list(item, list_of_items):
    for item_of_list in list_of_items:
        if item_of_list != item:
            return False
    return True


# check whether tracking run completed
def is_valid_tracking(tracking_dir):
    return "evaluation.txt" in os.listdir(tracking_dir)


if __name__ == "__main__":
    folder_type = determine_folder_type(results_path)
    # just one sequence folder from one tracking folder
    if folder_type == "hiob_sequence_folder":
        print("detected single sequence folder")
        create_graphs_metrics_for_set(results_path, "avg_full_set")

    # one hiob execution containing multiple sequence folders
    elif folder_type == "hiob_tracking_folder":
        print("detected hiob execution folder")
        create_graphs_metrics_for_set(results_path, "avg_full_set")
        create_sequence_score_csv(results_path, "sequence_results")
        create_attribute_score_csv(results_path, "attribute_results")

    # matlab tracking folder, containing the saved workplaces from each sequence
    elif folder_type == "matlab_tracking_folder":
        print("detected matlab workplace folder")
        create_graphs_metrics_for_set(results_path, "avg_full_set")
        create_sequence_score_csv(results_path, "sequence_results")
        create_attribute_score_csv(results_path, "attribute_results")

    # experiment folder containing multiple hiob executions, h_opt for example
    elif folder_type == "multiple_hiob_executions":
        if mode == "" or None:
            print("EITHER --mode opr or --mode exp")
        elif mode == "opt":
            print("detected multiple hiob executions, mode = opt")
            print("creating opt csv")
            create_opt_csv(results_path, "opt")
            print("creating graphs for parameter results")
            create_graphs_from_opt_csv(results_path)
        elif mode == "exp":
            print("detected multiple hiob executions, mode = exp")
            get_avg_results_from_experiment(results_path)
            print("creating graphs for average experiment metrics")
            create_graphs_metrics_for_set(results_path, "avg_full_set")

