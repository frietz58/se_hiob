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

parser.add_argument("-ptr", "--pathresults", nargs='+', help="Absolute path to the folder that contains the tracking "
                                                             "results, the script determines what can of folder it "
                                                             "is handling and gets the results accordingly")

parser.add_argument("-m", "--mode", help="To differentiate between an opt folder with multiple hiob executions and an "
                                         "experiment folder with multiple hiob executions")

parser.add_argument("-p_mfig", "--path_multi_figure", help="Path where the figure with multiple graphs shall be saved")

parser.add_argument("-t", "--task", help="Which task the shall be executed? framerate_vs_approach gen_final_comp_tab")

args = parser.parse_args()


results_path = args.pathresults
mode = args.mode
tb100_gt_path = "data/tb100"
tb100_attributes_path = "config/data_sets/tb100.yaml"
nicovis_attributes_path = "config/data_sets/nicovision.yaml"

if os.path.isdir("/media/finn/linux-ssd/nicovision"):
    nicovis_gt_path = "/media/finn/linux-ssd/nicovision"
else:
    nicovis_gt_path = "data/nicovision"

if args.path_multi_figure is not None:
    path_multi_figure = args.path_multi_figure
else:
    path_multi_figure = results_path[0]

# ======= csv names =======
csv_avg_success = "Avg. Success"
csv_avg_precision = "Avg. Precision"
csv_avg_fps = "Avg. Frame rate"
csv_avg_se_fps = "Avg. SE Frame rate"
csv_avg_ss = "Avg. Size Score"
csv_total_precision = "Total Precision"
csv_total_success = "Total Success"
sd_csv_name = "standard_deviations.csv"


# ======= plot font =======
# font = {'family': 'normal',
#         'size': 15}
# font = {'size': 15}
# matplotlib.rc('font', **font)

# ======= parameter name formatter =======

parameter_names = {
    "scale_factor": "Scale factor",
    "scale_sigma_factor": "Scale sigma",
    "learning_rate": "Learning rate",
    "scale_model_max": "Max scale model size",
    "scale_model_size": "Static scale model size",
    "dsst_number_scales": "Number samples",
    "d_change_aspect_ratio": "Change aspect ratio",
    "hog_cell_size": "Hog cell size",
    "hog_block_norm_size": "Hog block norm size",

    "c_scale_factor": "Scale factor",
    "inner_punish_threshold": "Inner punish threshold",
    "outer_punish_threshold": "Outer punish threshold",
    "c_number_scales": "Number candidates",
    "max_scale_difference": "Limit scale difference",
    "scale_window_step_size": "Scale window steps",
    "adjust_max_scale_diff": "Adjust scale limit",
    "adjust_max_scale_diff_after": "Adjust scale limit after",
    "c_change_aspect_ratio": "Change aspect ratio"
}

# ================================= GET FUNCTIONS =================================
# get all workplace files
def get_saved_workplaces(result_dir):
    files_in_dir = os.listdir(result_dir)

    for i, file in enumerate(files_in_dir):
        if not ".mat" in file:
            del files_in_dir[i]

    for i, file in enumerate(files_in_dir):
        if not ".mat" in file:
            del files_in_dir[i]

    return files_in_dir


# get the rects of predictions and gt from one workplace
def get_rects_from_workplace(workplace):
    results = scipy.io.loadmat(workplace)

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


# get the gt rects from a tb100 zip file
def get_gt_rects_from_zip(sequence, dataset_path=None):

    if dataset_path is None:
        dataset_path = tb100_gt_path

    zip = zipfile36.ZipFile(os.path.join(dataset_path, sequence + ".zip"))
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

            if "," in coord_string:
                coord_list = [pos_int_form_string(number) for number in coord_string.split(",")]
            elif " " in coord_string:
                coord_list = [pos_int_form_string(number) for number in coord_string.split(" ")]
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
            preds, gts = get_rects_from_workplace(os.path.join(result_dir, workplace))
            pred_gt_rects["preds"].append(preds)
            pred_gt_rects["gts"].append(gts)

    elif current_folder_type == "hiob_sequence_folder":
        sequence_name = result_dir.split("/")[-1].split("-")[-1]
        preds = get_pred_rects_from_sequence(result_dir)
        if "tb100" in result_dir:
            gts = get_gt_rects_from_zip(sequence_name, tb100_gt_path)
        elif "nicovision" in result_dir:
            gts = get_gt_rects_from_zip(sequence_name, nicovis_gt_path)
        pred_gt_rects["preds"].append(preds)
        pred_gt_rects["gts"].append(gts)

    elif current_folder_type == "hiob_tracking_folder":
        sequence_folders = get_sequences(result_dir)

        for sequence_folder in sequence_folders:
            sequence_name = sequence_folder.split("/")[-1].split("-")[-1]
            preds = get_pred_rects_from_sequence(os.path.join(result_dir, sequence_folder))
            if "tb100" in sequence_folder:
                gts = get_gt_rects_from_zip(sequence_name, tb100_gt_path)
            elif "nicovision" in sequence_folder:
                gts = get_gt_rects_from_zip(sequence_name, nicovis_gt_path)
            pred_gt_rects["preds"].append(preds)
            pred_gt_rects["gts"].append(gts)

    elif current_folder_type == "multiple_hiob_executions":
        hiob_executions = get_tracking_folders(result_dir)

        for hiob_execution in hiob_executions:
            sequence_folders = get_sequences(hiob_execution)
            for sequence in sequence_folders:
                sequence_folder = os.path.join(hiob_execution, sequence)
                sequence_name = os.path.basename(sequence_folder).split("-")[-1]
                preds = get_pred_rects_from_sequence(sequence_folder)
                if "tb100" in sequence:
                    gts = get_gt_rects_from_zip(sequence_name, tb100_gt_path)
                elif "nicovision" in sequence:
                    gts = get_gt_rects_from_zip(sequence_name, nicovis_gt_path)

                # nico dataset seems to have some problems with the ground truth, as they are often one of
                if len(preds) != len(gts):
                    print("diff in pred vs gts: " + str(abs(len(preds) - len(gts))) + ", sequence: " + str(sequence))
                    gts = gts[0:len(preds)]

                pred_gt_rects["preds"].append(preds)
                pred_gt_rects["gts"].append(gts)

    elif current_folder_type == "matlab_sequence_file":
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


# get the evaluation values from each tracking in an experiment
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

    # calculate average metrics on each sequence
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
        csv_fields = ["Sample", "Frames", "Precision", "Success", "Size Score"]
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

            row_dict = {
                "Sample": sample,
                "Frames": frames,
                "Precision": avg_prec,
                "Success": avg_succ,
                "Size Score": avg_ss_rating}

            writer.writerow(row_dict)
            rows.append(row_dict)

    create_attribute_tex_table_include(save_path=eval_path,
                                       csv_file=out_csv,
                                       tex_name="sequence_averages_tab_include.tex")


    # create csv with the average values for each attribute
    print("creating average over attributes csv")
    out_csv = os.path.join(eval_path, "attribute_averages.csv")
    csv_fields = ["Attribute", "Samples", "Frames", csv_avg_precision, csv_avg_success, csv_avg_ss]
    with open(out_csv, 'w', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=csv_fields)
        writer.writeheader()
        attribute_sequences = get_attribute_collections(experiment_folder)

        for attribute in attribute_sequences:
            print("attribute: {0}".format(attribute))
            print("loading all rects for sequences with same attribute")
            specific_attribute_sequences = attribute_sequences[attribute]
            if specific_attribute_sequences != []:
                all_preds = []
                all_gts = []
            elif specific_attribute_sequences == []:
                print("no sequences for : " + str(attribute) + ", skipping")
                continue

            prec_sum = 0
            succ_sum = 0
            size_score_sum = 0
            unique_samples = []
            frames = 0

            for sequence in specific_attribute_sequences:
                if sequence.split("/")[-1].split("-")[-1] not in unique_samples:
                    unique_samples.append(sequence.split("/")[-1].split("-")[-1])

                sequence_preds, sequence_gts = get_all_rects(sequence)
                sequence_results = get_metrics_from_rects("attribute", all_preds=sequence_preds, all_gts=sequence_gts)
                prec_sum += sequence_results["Total Precision"]
                succ_sum += sequence_results["Total Success"]
                size_score_sum += sequence_results["Size Score"]
                frames += len(sequence_gts)

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

            row = {
                "Attribute": attribute,
                "Samples": len(unique_samples),
                "Frames": int(frames / (len(specific_attribute_sequences) / len(unique_samples))),
                csv_avg_precision: np.around(score_dict["Total Precision"], decimals=3),
                csv_avg_success: np.around(score_dict["Total Success"], decimals=3),
                # dont take value from score dict, that normalizes across every sequences adn thus is always to small
                csv_avg_ss: np.around(size_score_sum / len(specific_attribute_sequences), decimals=3)
            }

            writer.writerow(row)

            print(row)

            # writer.writerow({
            #     "Attribute": attribute,
            #     "Samples": len(unique_samples),
            #     "Frames": int(frames) / (len(specific_attribute_sequences) / len(unique_samples)),
            #     csv_avg_precision: np.around(prec_sum / len(specific_attribute_sequences), decimals=3),
            #     csv_avg_success: np.around(succ_sum / len(specific_attribute_sequences), decimals=3),
            #     csv_avg_ss: np.around(size_score_sum / len(specific_attribute_sequences), decimals=3)
            # })

        # get the average values of the rows:
        final_precs = []
        final_succs = []
        final_ss = []
        final_fails = []
        final_updates = []
        samples = 0
        frames = 0
        for row in rows:
            final_precs.append(row["Precision"])
            final_succs.append(row["Success"])
            final_ss.append(row["Size Score"])
            # final_fails.append(row["Fail %"])
            # final_updates.append(row["Updates"])
            samples += 1
            frames += int(row["Frames"])

        hiob_executions = get_tracking_folders(experiment_folder)

        if len(hiob_executions) != 1:
            # get average of totals
            total_succs = []
            total_precs = []
            tracking_results = {}
            for hiob_execution in hiob_executions:
                with open(os.path.join(hiob_executions[0], "evaluation.txt"), "r") as eval_txt:
                    lines = eval_txt.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        key_val = line.split("=")
                        if key_val[0] == "total_precision_rating":
                            total_precs.append(float(key_val[1]))
                        elif key_val[0] == "total_success_rating":
                            total_succs.append(float(key_val[1]))

            tracking_results["total_precision_rating"] = sum(total_precs) / len(total_precs)
            tracking_results["total_success_rating"] = sum(total_succs) / len(total_succs)

        else:
            # read values from tracking eval file
            tracking_results = {}
            with open(os.path.join(hiob_executions[0], "evaluation.txt"), "r") as eval_txt:
                lines = eval_txt.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    key_val = line.split("=")
                    tracking_results[str(key_val[0])] = key_val[1]

        final_avg_precs = np.around(np.sum(final_precs) / samples, decimals=3)
        final_avg_succs = np.around(np.sum(final_succs) / samples, decimals=3)
        final_avg_ss = np.around(np.sum(final_ss) / samples, decimals=3)
        # final_avg_fails = np.around(np.sum(final_fails) / samples, decimals=3)
        # final_avg_updates = np.around(np.sum(final_updates) / samples, decimals=3)


        dataset = get_dataset_from_name(experiment_folder)

        summarizing_row = {
            "Attribute": dataset,
            "Samples": samples,
            "Frames": np.around(frames),
            csv_avg_precision: np.around(float(tracking_results["total_precision_rating"]), decimals=3),
            csv_avg_success: np.around(float(tracking_results["total_success_rating"]), decimals=3),
            csv_avg_ss: final_avg_ss}

        summarizing_row2 = {
            "Attribute": dataset,
            "Samples": samples,
            "Frames": np.around(frames),
            csv_avg_precision: final_avg_precs,
            csv_avg_success: final_avg_succs,
            csv_avg_ss: final_avg_ss}

        writer.writerow(summarizing_row)
        #writer.writerow(summarizing_row2)

    create_attribute_tex_table_include(save_path=eval_path,
                                       csv_file=out_csv,
                                       tex_name="attribute_averages_tab_include.tex")

    return None


# create a tex table for based on csv file for tex include:
def create_attribute_tex_table_include(save_path, csv_file, tex_name):
    tex_path = os.path.join(save_path, tex_name)
    print("saving table include tex to: " + str(tex_path))
    df = pd.read_csv(csv_file)
    cols = list(df)
    # sorted = df.sort_values(cols[0])
    lines = [
        #"\\begin{table}[]\label{tab:asdasd}\n",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{\n",
        "\\begin{tabular}{@{}cccccc@{}}\n",
        "\\toprule\n",
    ]

    header_line = ""
    for key in df.keys():
        header_line += "\\textbf{" + str(key) + "} & "
    # remove & at end of lline
    header_line = header_line[0:-2]
    header_line += "\\\\ \\midrule\n"

    lines.append(header_line)

    for index, row in df.iterrows():
        line = ""
        for key in df.keys():
            if type(row[key]) == float:
                line += str(np.around(row[key], decimals=3)) + " & "
            else:
                line += str(row[key]) + " & "
        # remove & at end of lline
        line = line[0:-2]
        line += "\\\\ \n"
        lines.append(line)
        print(line)

    lines[-1] = lines[-1][0:-2] + " \\bottomrule\n"
    lines.append("\\end{tabular}\n")
    lines.append("}\n")
    lines.append("\\caption{Generated from:" + str(csv_file).replace("_", "\_") + "}\n")
    #lines.append("\\end{table}")

    with open(tex_path, "w") as tex_file:
        tex_file.writelines(lines)


def create_framerate_tex_include(eval_folder, dict=None):
    hard_values = {
        "Candidats static cont.": [8.03, 5.52, 518.81, 587.09],
        "Candidates static limited": [8.01, 5.68, 3092.23, 3237.78],
        "Candidates dynamic cont.": [7.71, 5.7, 331.68, 330.09],
        "Candidates dynamic limited": [7.8, 5.75, 2399.54, 2574.43],
        "DSST static cont.": [7.82, 5.23, 80.19, 54.58],
        "DSST staic limited": [7.82, 5.78, 949.44, 734.70],
        "DSST dynamic cont.": [6.64, 4.37, 46.74, 17.49],
        "DSST dynamic limited": [7.73, 5.64, 573.09, 280.75],
        "No SE": [7.86, 5.84, "-", "-"]
    }

    lines = [
        #"\\begin{table}[]\label{tab:asdasd}\n",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{\n",
        "\\begin{tabular}{@{}cccccc@{}}\n",
        "\\toprule\n",
    ]

    header_line = "\\textbf{Approach} & \\textbf{Overall FPS TB100} & \\textbf{Overall FPS NICO} & \\textbf{SE FPS TB100} & \\textbf{SE FPS NICO} \\\\ \midrule \n"
    lines.append(header_line)

    for key in hard_values.keys():
        line = str(key.replace("_", "\_")) + " & "
        for val in hard_values[key]:
            line += str(val) + " & "
        # remove & at end of lline
        line = line[0:-2]
        line += "\\\\ \n"
        lines.append(line)
        print(line)

    lines[-1] = lines[-1][0:-2] + " \\bottomrule\n"
    lines.append("\\end{tabular}\n")
    lines.append("}\n")
    lines.append("\\caption{Framerates of overall tracker and isolated SE module on both datasets}\n")
    #lines.append("\\end{table}")

    if not os.path.isdir(eval_folder):
        os.mkdir(eval_folder)

    with open(os.path.join(eval_folder, "fps_vs_approach_tab_include.tex"), "w") as tex_file:
        tex_file.writelines(lines)


# get metric from rects
def get_metrics_from_rects(result_folder, all_preds=None, all_gts=None, sequences=None):
    print("getting metrics for {0}".format(result_folder))

    if all_preds is None and all_gts is None:
        all_preds, all_gts = get_all_rects(result_folder)

    if result_folder != "attribute" and result_folder is not None:
        folder_type = determine_folder_type(result_folder)
    center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)

    # calculate the metrics based on the results for each frame of the sequences in the collection
    scores_for_rects = {}

    if result_folder != "attribute" and result_folder is not None:
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

    folder_type = determine_folder_type(results_folder)
    if folder_type == "matlab_sequence_file":
        sequence_name = os.path.basename(results_folder).split("_")[0]
        eval_path = os.path.join(os.path.dirname(results_folder), sequence_name + "_evaluation")
    else:
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
    folder_type = determine_folder_type(result_folder)

    if folder_type == "matlab_sequence_file":
        sequence_name = os.path.basename(result_folder).split("_")[0]
        eval_path = os.path.join(os.path.dirname(result_folder), sequence_name + "_evaluation")
    else:
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

        if type(sequences) == list:
            for sequence in sequences:
                if type(results_path) == list and len(results_path) == 1:
                    str_results_path = results_path[0]
                elif type(results_path) == str:
                    str_results_path = results_path

                if type(sequence) == list:
                    str_sequence = sequence[0]
                elif type(sequence) == str:
                    str_sequence = sequence

                score_dict = get_metrics_from_rects(os.path.join(str_results_path, str_sequence))

                sequence_name = str_sequence.split("-")[-1]
                writer.writerow({
                    "Sample": sequence_name,
                    "Frames": score_dict["Frames"],
                    "Precision": np.around(score_dict["Total Precision"], decimals=3),
                    "Success": np.around(score_dict["Total Success"], decimals=3),
                    "Size Score": score_dict["Size Score"]
                })

        elif type(sequences) == str and folder_type == "matlab_sequence_file":

            score_dict = get_metrics_from_rects(results_path[0])

            writer.writerow({
                "Sample": sequence_name,
                "Frames": score_dict["Frames"],
                "Precision": np.around(score_dict["Total Precision"], decimals=3),
                "Success": np.around(score_dict["Total Success"], decimals=3),
                "Size Score": score_dict["Size Score"]
            })

    create_attribute_tex_table_include(eval_path, out_csv, "sequences_tab_tex_include.tex")


# create a csv for the attribute collections from one tracking
def create_attribute_score_csv(result_folder, eval_folder):
    sequences = get_sequences(result_folder)
    folder_type = determine_folder_type(result_folder)

    if folder_type == "matlab_sequence_file":
        sequence_name = os.path.basename(result_folder).split("_")[0]
        eval_path = os.path.join(os.path.dirname(result_folder), sequence_name + "_evaluation")
    else:
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
        rows = []

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

                row = {
                    "Attribute": attribute,
                    "Samples": len(attribute_sequences),
                    "Frames": score_dict["Frames"],
                    csv_total_precision: np.around(score_dict["Total Precision"], decimals=3),
                    csv_total_success: np.around(score_dict["Total Success"], decimals=3),
                    csv_avg_ss: score_dict["Size Score"]
                }

                print(row)

                writer.writerow(row)
                rows.append(row)

        # get the average values of the rows:
        final_precs = []
        final_succs = []
        final_ss = []
        final_fails = []
        final_updates = []
        samples = 0
        frames = 0
        for row in rows:
            final_precs.append(row[csv_total_precision])
            final_succs.append(row[csv_total_success])
            final_ss.append(row[csv_avg_ss])
            samples += 1
            frames += int(row["Frames"])
        final_avg_precs = np.around(np.sum(final_precs) / samples, decimals=3)
        final_avg_succs = np.around(np.sum(final_succs) / samples, decimals=3)
        final_avg_ss = np.around(np.sum(final_ss) / samples, decimals=3)
        final_avg_fails = np.around(np.sum(final_fails) / samples, decimals=3)
        final_avg_updates = np.around(np.sum(final_updates) / samples, decimals=3)

        if samples == 0:
            samples = 1

        summarizing_row = {
            "Attribute": "TB100",
            "Samples": samples,
            "Frames": int(np.around(frames / samples)),
            csv_total_precision: final_avg_precs,
            csv_total_success: final_avg_succs,
            csv_avg_ss: final_avg_ss}

        writer.writerow(summarizing_row)

    create_attribute_tex_table_include(save_path=eval_path,
                                       csv_file=out_csv,
                                       tex_name="attribute_averages_tab_include.tex")


# create a csv comparing the trackings in one csv folder (opt)
def create_opt_csv(experiment_folder, eval_folder):
    if experiment_folder.split("/")[-1] == "":
        filename = experiment_folder.split("/")[-2]
    else:
        filename = experiment_folder.split("/")[-1]
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

            sd_df = pd.DataFrame(columns=[
                changing_parameter,
                "avg_succ",
                "avg_prec",
                "avg_fps",
                "avg_se_fps",
                "succ_sd",
                "prec_sd",
                "avg_fps_sd",
                "avg_se_fps_sd"
            ])

            # get the average values for the same parameter value for a row in the opt csv

            for i, value in enumerate(same_parameter_value_collection.values()):
                avg_succs = []
                avg_precs = []
                se_framerates = []
                framerates = []
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

                print("succs: " + str(avg_succs))
                print("precs: " + str(avg_precs))
                print("framerates: " + str(framerates))
                print("se_framerates: " + str(se_framerates))

                # calc sds
                succ_sd_helper = [x - final_avg_succ for x in avg_succs]
                succ_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in succ_sd_helper]), len(avg_succs)))

                prec_sd_helper = [x - final_avg_precc for x in avg_precs]
                prec_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in prec_sd_helper]), len(avg_precs)))

                avg_fps_helper = [x - final_framerate for x in framerates]
                avg_fps_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in avg_fps_helper]), len(framerates)))

                avg_se_fps_helper = [x - final_se_framerate for x in se_framerates]
                avg_se_fps_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in avg_se_fps_helper]), len(se_framerates)))

                sd_df.loc[i] = [
                    value[0]["value"],
                    final_avg_succ,
                    final_avg_precc,
                    final_framerate,
                    final_se_framerate,
                    succ_sd,
                    prec_sd,
                    avg_fps_sd,
                    avg_se_fps_sd
                ]

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

            sd_df = pd.DataFrame(columns=[
                changing_parameter,
                "avg_succ",
                "avg_prec",
                "avg_fps",
                "avg_se_fps",
                "succ_sd",
                "prec_sd",
                "avg_fps_sd",
                "avg_se_fps_sd"
            ])

            # get the average values for the same parameter value for a row in the opt csv
            for i, value in enumerate(same_parameter_value_collection.values()):
                avg_succs = []
                avg_precs = []
                se_framerates = []
                framerates = []

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
                print("succs: " + str(avg_succs))
                print("precs: " + str(avg_precs))
                print("framerates: " + str(framerates))
                print("se_framerates: " + str(se_framerates))

                writer.writerow({csv_avg_success: final_avg_succ,
                                 csv_avg_precision: final_avg_precc,
                                 csv_avg_fps: final_framerate,
                                 csv_avg_se_fps: final_se_framerate,
                                 changing_parameter: value[0]["value"]})

                # calc sds
                succ_sd_helper = [x - final_avg_succ for x in avg_succs]
                succ_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in succ_sd_helper]), len(avg_succs) - 1))

                prec_sd_helper = [x - final_avg_precc for x in avg_precs]
                prec_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in prec_sd_helper]), len(avg_precs) - 1))

                avg_fps_helper = [x - final_framerate for x in framerates]
                avg_fps_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in avg_fps_helper]), len(framerates) - 1))

                avg_se_fps_helper = [x - final_se_framerate for x in se_framerates]
                avg_se_fps_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in avg_se_fps_helper]), len(se_framerates) - 1))

                sd_df.loc[i] = [
                    value[0]["value"],
                    final_avg_succ,
                    final_avg_precc,
                    final_framerate,
                    final_se_framerate,
                    succ_sd,
                    prec_sd,
                    avg_fps_sd,
                    avg_se_fps_sd
                ]

    # sort created csv by parameter and override old
    df = pd.read_csv(csv_name)
    cols = list(df)
    sorted = df.sort_values(cols[-1])
    sorted.to_csv(csv_name, )

    # save the helper csv with the standard deviation
    sd_df.to_csv(os.path.join(experiment_folder, sd_csv_name), index=False)

    return None


def sequence_boxplots(tracking_folder):
    sequences_scores = {}
    # get the sequence metrics
    hiob_executions = get_tracking_folders(tracking_folder)
    for hiob_execution in hiob_executions:
        sequences = get_sequences(hiob_execution)
        for sequence in sequences:
            sequence_name = sequence.split("-")[-1]
            if sequence_name == "Basketball":
                continue
            if sequence_name not in sequences_scores.keys():
                sequences_scores[str(sequence_name)] = {
                    "total_precs": [],
                    "total_succs": [],
                    "size_scores": []
                }

            with open(os.path.join(hiob_execution, sequence, "evaluation.txt"), "r") as eval_txt:
                lines = eval_txt.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    key_val = line.split("=")
                    if key_val[0] == "success_rating":
                        sequences_scores[str(sequence_name)]["total_succs"].append(float(key_val[1]))
                    elif key_val[0] == "precision_rating":
                        sequences_scores[str(sequence_name)]["total_precs"].append(float(key_val[1]))
                    elif key_val[0] == "area_between_size_curves":
                        sequences_scores[str(sequence_name)]["size_scores"].append(float(key_val[1]))

    # size score boxplots:
    fig, ax = plt.subplots()
    # y_ticks = np.arange(len(list(sequences_scores.keys()))), (list(sequences_scores.keys()))
    y_ticks = np.arange(len(list(sequences_scores.keys())))
    data = []
    for ind, sequence in enumerate(list(sequences_scores.keys())):
        data.append(sequences_scores[str(sequence)]["size_scores"])
        print("{0} median ss: {1}". format(sequence, np.percentile(data, 50)))

    ax.boxplot(data, 0, 'r+', 0, positions=y_ticks, labels=list(sequences_scores.keys()))

    ax.set_title('Size score boxplots')
    plt.yticks(y_ticks)
    # plt.xlim(0, 150)
    plt.savefig(os.path.join(tracking_folder, "sequences_size_boxplots.png"))

    # precision boxplots:
    fig, ax = plt.subplots()
    # y_ticks = np.arange(len(list(sequences_scores.keys()))), (list(sequences_scores.keys()))
    y_ticks = np.arange(len(list(sequences_scores.keys())))
    data = []
    for ind, sequence in enumerate(list(sequences_scores.keys())):
        data.append(sequences_scores[str(sequence)]["total_precs"])

    ax.boxplot(data, 0, 'r+', 0, positions=y_ticks, labels=list(sequences_scores.keys()))

    ax.set_title('Precision rating boxplots')
    plt.yticks(y_ticks)
    plt.savefig(os.path.join(tracking_folder, "precision_boxplots.png"))


# create figure with precision and success of different trackings
def multiple_trackings_graphs(tracking_folders, eval_folder, what_is_plotted, font, tex_name, legend_by):

    plt.rcParams.update(font)

    if not os.path.isdir(eval_folder):
        os.mkdir(eval_folder)
    print("saving multiple graphs figs at: " + str(eval_folder))

    # precision plot
    f = plt.figure()
    x = np.arange(0., 50.1, .1)
    figure_file2 = os.path.join(eval_folder, 'multiple_precision_plot.svg')
    figure_file3 = os.path.join(eval_folder, 'multiple_precision_plot.pdf')

    if len(tracking_folders) > 2:
        wide_legend = True
    else:
        wide_legend = False

    labels = []
    lines = []
    for tracking_folder in tracking_folders:
        all_preds, all_gts = get_all_rects(tracking_folder)
        center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)
        algorithm = get_approach_from_yaml(tracking_folder)
        dataset = get_dataset_from_name(tracking_folder)

        dfun = build_dist_fun(center_distances)
        y = [dfun(a) for a in x]
        at20 = dfun(20)
        tx = "prec(20) = %0.4f" % at20#
        # plt.text(5.05, 0.05, tx)
        plt.xlabel("center distance [pixels]")
        plt.ylabel("occurrence")
        plt.xlim(xmin=0, xmax=50)
        plt.ylim(ymin=0.0, ymax=1.0)
        if legend_by == "algorithm":
            label = str(np.around(at20, decimals=3)) + " " + algorithm
        elif legend_by == "dataset":
            label = str(np.around(at20, decimals=3)) + " on " + dataset
        line = plt.plot(x, y, label=label)
        lines.append(line)
        labels.append(label)

    plt.axvline(x=20, linestyle=':', color='k')
    if wide_legend:
        plt.legend(ncol=2, mode="expand", loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.subplots_adjust(bottom=0.35)
        plt.title(str(what_is_plotted))
    else:
        plt.legend(loc="lower right")
        plt.subplots_adjust(bottom=0.15)
        if legend_by == "dataset":
            plt.title(str(algorithm + " precision"))
        elif legend_by == "algorithm":
            plt.title(str(what_is_plotted))

    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    # success plot
    f = plt.figure()
    x = np.arange(0., 1.001, 0.001)
    figure_file2 = os.path.join(eval_folder, 'multiple_success_plot.svg')
    figure_file3 = os.path.join(eval_folder, 'multiple_success_plot.pdf')

    labels = []
    lines = []
    for tracking_folder in tracking_folders:
        all_preds, all_gts = get_all_rects(tracking_folder)
        center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)
        algorithm = get_approach_from_yaml(tracking_folder)
        dataset = get_dataset_from_name(tracking_folder)

        ofun = build_over_fun(overlap_scores)
        y = [ofun(a) for a in x]
        auc = np.trapz(y, x)
        tx = "AUC = %0.4f" % auc
        # plt.text(0.05, 0.05, tx)
        plt.xlabel("overlap score")
        plt.ylabel("occurrence")
        plt.xlim(xmin=0.0, xmax=1.0)
        plt.ylim(ymin=0.0, ymax=1.0)
        if legend_by == "algorithm":
            label = str(np.around(auc, decimals=3)) + " " + algorithm
        elif legend_by == "dataset":
            label = str(np.around(auc, decimals=3)) + " on " + dataset
        line = plt.plot(x, y, label=label)
        lines.append(line)
        labels.append(label)

    if wide_legend:
        plt.legend(ncol=2, mode="expand", loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.subplots_adjust(bottom=0.35)
        plt.title(str(what_is_plotted))
    else:
        plt.legend(loc="lower left")
        plt.subplots_adjust(bottom=0.15)
        if legend_by == "dataset":
            plt.title(str(algorithm + " success"))
        elif legend_by == "algorithm":
            plt.title(str(what_is_plotted))

    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    create_multiple_graphs_tex_include(save_folder=eval_folder,
                                       path_in_src="dsst_validation",
                                       tex_name=tex_name,
                                       subfigures=["multiple_precision_plot", "multiple_success_plot"],
                                       )


# create a tex figure inclue with the the precision and success graph of multiple trackings:
def create_multiple_graphs_tex_include(save_folder, path_in_src, tex_name, subfigures):
    tex_path = os.path.join(save_folder, tex_name)
    print("saving tex figure to " + str(tex_path))
    with open(tex_path, "w") as tex_file:
        tex_file.writelines(
            [
                "\\begin{figure}[H]\label{TODO}\n"
                "\\begin{subfigure}[t]{0.5\\textwidth}\n",
                "\\centering\\captionsetup{width=.9\\linewidth}\n",
                "\\includegraphics[width=\\textwidth]{" + os.path.join(path_in_src, str(subfigures[0])) + "}\n",
                "\\subcaption{The precision plot.}",
                "\\end{subfigure}\n",
                "\\begin{subfigure}[t]{0.5\\textwidth}\n",
                "\\centering\\captionsetup{width=.9\\linewidth}\n",
                "\\includegraphics[width=\\textwidth]{" + os.path.join(path_in_src, str(subfigures[1])) + "}\n",
                "\\subcaption{The success plot.}",
                "\\end{subfigure}\n",
                "\\caption{The plotted results of TODO. Generate from " + str(save_folder) + "}"
                "\\end{figure}"
            ]
        )


# create the graphs from rects
def create_graphs_from_rects(result_folder, eval_folder):
    all_preds, all_gts = get_all_rects(result_folder)
    center_distances, overlap_scores, gt_size_scores, size_scores, frames = get_scores_from_rects(all_preds, all_gts)
    folder_type = determine_folder_type(result_folder)

    if folder_type == "matlab_sequence_file":
        sequence_name = os.path.basename(result_folder).split("_")[0]
        eval_path = os.path.join(os.path.dirname(result_folder), sequence_name + "_evaluation")
    else:
        eval_path = os.path.join(result_folder, "evaluation")

    eval_path = os.path.join(eval_path, eval_folder)

    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)

    # precision plot
    plt.rcParams.update({'font.size': 15})
    dfun = build_dist_fun(center_distances)
    figure_file2 = os.path.join(eval_path, 'precision_plot.svg')
    figure_file3 = os.path.join(eval_path, 'precision_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 50.1, .1)
    y = [dfun(a) for a in x]
    at20 = dfun(20)
    tx = "Precision(20) = %0.4f" % at20
    # plt.text(5.05, 0.05, tx)
    plt.xlabel("center distance [pixels]")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0, xmax=50)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y, color="#ffb000", label=tx)
    plt.axvline(20, color='k', linestyle=':')
    plt.legend(loc="lower right")
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    # success plot
    plt.rcParams.update({'font.size': 15})
    ofun = build_over_fun(overlap_scores)
    figure_file2 = os.path.join(eval_path, 'success_plot.svg')
    figure_file3 = os.path.join(eval_path, 'success_plot.pdf')
    f = plt.figure()
    x = np.arange(0., 1.001, 0.001)
    y = [ofun(a) for a in x]
    auc = np.trapz(y, x)
    tx = "AUC = %0.4f" % auc
    # plt.text(0.05, 0.05, tx)
    plt.xlabel("overlap score")
    plt.ylabel("occurrence")
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=1.0)
    plt.plot(x, y, color='#648fff', label=tx)
    plt.legend(loc="lower left")
    plt.subplots_adjust(bottom=0.15)
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
    plt.xlabel("Frame")
    plt.ylabel("Size")
    # plt.text(x=10, y=10, s="abc={0}".format(abc))
    size_label = "Size score = {0}".format(abc)
    plt.fill_between(dim, normalized_size_score, normalized_gt_size_scores, color="#ffb000", alpha=0.7, label=size_label)
    plt.plot(dim, normalized_gt_size_scores, color='#785ef0', label='Groundtruth size', alpha=1)
    plt.plot(dim, normalized_size_score, color='#fe6100', label='Predicted size', alpha=1)
    plt.axhline(y=normalized_size_score[0], color='k', linestyle=':', label='Initial size')
    plt.legend(ncol=2, mode="expand", loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.subplots_adjust(bottom=0.3)
    plt.xlim(1, len(normalized_size_score))
    plt.savefig(figure_file2)
    plt.savefig(figure_file3)
    plt.close()

    create_multiple_graphs_tex_include(save_folder=eval_path,
                                       path_in_src=eval_path,
                                       tex_name="precision_vs_success_fig_include.tex",
                                       subfigures=["precision_plot", "success_plot"])


# create graphs and metrics for a specific set of results
def create_graphs_metrics_for_set(set_of_results, set_name):
    create_avg_score_csv(set_of_results, set_name)
    create_graphs_from_rects(set_of_results, set_name)


# create the graphs for the opt plotting parameter value vs x
def create_graphs_from_opt_csv(obt_folder):
    # set font size for graphs that are generated
    plt.rcParams.update({'font.size': 15})

    # obt folder are named like the parameter that is obtimized, this:
    if obt_folder.split("/")[-1] == "":
        parameter_name = obt_folder.split("/")[-2]
    else:
        parameter_name = obt_folder.split("/")[-1]

    if os.path.isdir(obt_folder):
        files_in_dir = os.listdir(obt_folder)

    csv_name = parameter_name + "_opt.csv"

    if csv_name in files_in_dir:
        df = pd.read_csv(os.path.join(obt_folder, csv_name))
        sorted_df = df.sort_values(parameter_name)

        sd_df = pd.read_csv(os.path.join(obt_folder, sd_csv_name))
        sorted_sd = sd_df.sort_values(parameter_name)

        # ================ PARAMETER vs METRIC ================
        success = sorted_df[csv_avg_success]
        success_sd = sorted_sd["succ_sd"]
        precision = sorted_df[csv_avg_precision]
        precision_sd = sorted_sd["prec_sd"]

        ind = np.arange(len(success))  # the x locations datapoints

        fig, ax = plt.subplots()

        success_graph = ax.errorbar(ind, success, yerr=success_sd, color='#648fff', capsize=3)
        precision_graph = ax.errorbar(ind, precision, yerr=precision_sd, color='#ffb000', capsize=3)

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Scores')
        fig.text(0.5, 0.04, parameter_names[str(parameter_name)], ha='center')
        ax.set_xticks(ind)
        ax.set_xticklabels(sorted_df[parameter_name])
        if len(ind) > 5:
            plt.xticks(rotation=45)
            plt.subplots_adjust(bottom=0.2)
        else:
            plt.subplots_adjust(bottom=0.15)

        ax.legend((success_graph[0], precision_graph[0]), (csv_avg_success, csv_avg_precision))
        ax.set_ylim(0, 1)

        # plt.show()
        figure_file1 = os.path.join(obt_folder, 'parameter_vs_metrics.pdf')
        plt.savefig(figure_file1)

        # ================ PARAMETER vs FRAMERATE ================
        framerate = sorted_df[csv_avg_fps]
        framerate_sd = sorted_sd["avg_fps_sd"]

        se_framerate = sorted_df[csv_avg_se_fps]
        se_framerate_sd = sorted_sd["avg_se_fps_sd"]

        ind = np.arange(len(framerate))  # the x locations for the groups

        if max(se_framerate) < 70:
            fig, ax = plt.subplots()

            se_framerate_graph = ax.errorbar(ind, se_framerate, yerr=se_framerate_sd, color='#785ef0', capsize=3)
            framerate_graph = ax.errorbar(ind, framerate, yerr=framerate_sd, color='#fe6100', capsize=3)

            ax.set_ylim(0, 60)
            ax.set_ylabel('Frame-rate')
            fig.text(0.5, 0.04, parameter_names[str(parameter_name)], ha='center')
            ax.set_xticks(ind)
            ax.set_xticklabels(sorted_df[parameter_name])
            ax.legend((se_framerate_graph[0], framerate_graph[0]), ('Avg. SE Frame-rate', 'Avg. Frame-rate'))
            if len(ind) > 5:
                plt.xticks(rotation=45)
                plt.subplots_adjust(bottom=0.2)
            else:
                plt.subplots_adjust(bottom=0.15)

        else:
            fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)

            se_framerate_graph = ax.errorbar(ind, se_framerate, yerr=se_framerate_sd, color='#785ef0', capsize=3)
            ax.set_ylim((min(se_framerate) - max(se_framerate_sd) * 5), (max(se_framerate) + max(se_framerate_sd) * 5))
            ax.set_ylabel('SE Frame-rate')
            fig.text(0.5, 0.04, parameter_names[str(parameter_name)], ha='center')
            ax.set_xticks(ind)
            ax.set_xticklabels(sorted_df[parameter_name])
            if len(ind) > 5:
                plt.xticks(rotation=45)
                plt.subplots_adjust(bottom=0.2)
            else:
                plt.subplots_adjust(bottom=0.15)

            framerate_graph = ax2.errorbar(ind, framerate, yerr=framerate_sd, color='#fe6100', capsize=3)
            ax2.set_ylim((min(framerate) - max(framerate_sd) * 5), (max(framerate) + max(framerate_sd) * 5))
            ax2.set_ylabel('Frame-rate')
            ax2.set_xticks(ind)
            ax2.set_xticklabels(sorted_df[parameter_name])

            ax.legend((se_framerate_graph[0], framerate_graph[0]), ('Avg. SE Frame-rate', 'Avg. Frame-rate'))

            # hide the spines between ax and ax2
            ax.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax.xaxis.tick_top()
            ax.tick_params(labeltop='off')  # don't put tick labels at the top
            ax2.xaxis.tick_bottom()

            d = .015  # how big to make the diagonal lines in axes coordinates
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
            ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        # plt.show()
        figure_file2 = os.path.join(obt_folder, 'parameter_vs_framerate.pdf')
        plt.savefig(figure_file2)

    create_obt_fig_tex_include(dataframe=sorted_df,
                               tex_name="include_figure.tex",
                               obt_folder=obt_folder,
                               path_in_src="opt_results",  # for the include command in the tex file
                               parameter_name=str(parameter_name))


# create a tex file with a subfigure containing the two graphs for the parameter
def create_obt_fig_tex_include(dataframe, obt_folder, tex_name, path_in_src, parameter_name):
    tex_path = os.path.join(obt_folder, tex_name)
    approach = get_approach_to_parameter(parameter_name)
    with open(tex_path, "w") as tex_file:
        tex_file.writelines(
                [
                    "\\begin{subfigure}[t]{0.5\\textwidth}\n",
                    "\\centering\\captionsetup{width=.9\\linewidth}\n",
                    "\includegraphics[width=\\textwidth]{" + os.path.join(path_in_src, approach, parameter_name, "parameter_vs_metrics") + "}\n",
                    "\\end{subfigure}\n",
                    "\\begin{subfigure}[t]{0.5\\textwidth}\n",
                    "\\centering\\captionsetup{width=.9\\linewidth}\n",
                    "\\includegraphics[width=\\textwidth]{" + os.path.join(path_in_src, approach, parameter_name, "parameter_vs_framerate") + "}\n",
                    "\\end{subfigure}\n",
                    "\\subcaption[]{The average metric scores and the average impact on the computational load of the "
                        "values of the \\textit{" + str(parameter_name.replace("_", "\\_")) + "} parameter. The best performing value is " +
                        str(dataframe.loc[dataframe[csv_avg_success].idxmax()][parameter_name]) + ", with an average success"
                        " rating of " + str(dataframe.loc[dataframe[csv_avg_success].idxmax()][csv_avg_success]) + " and "
                        "an overall tracking frame-rate of " + str(dataframe.loc[dataframe[csv_avg_success].idxmax()][csv_avg_fps]) + ". }"
                 ]
            )


# create a figure with the framerates of the different approaches
def create_framerate_vs_approach_fig(trackings, eval_folder):

    # get the approaches from all trackings
    approaches = {}
    for tracking in trackings:
        approach = get_approach_from_yaml(tracking)
        if approach not in approaches.keys():
            approaches[str(approach)] = {}
            approaches[str(approach)]["trackings"] = []
            approaches[str(approach)]["overall_fps"] = 0
            approaches[str(approach)]["se_fps"] = 0
            approaches[str(approach)]["trackings"].append(tracking)
        else:
            approaches[str(approach)]["trackings"].append(tracking)

    # get the fps of all trackings of one approach
    print("gettings fps and se_fps for approaches")

    # split into candidates and dsst
    dsst_dicts = {}
    candidate_dicts = {}

    for approach in approaches:
        overall_fps = 0
        overall_fps_singles = []
        se_fps = 0
        se_fps_singles = []
        for tracking in approaches[approach]["trackings"]:
            hiob_executions = get_tracking_folders(tracking)
            if len(hiob_executions) > 1:
                print("warning, found multiple hiob executions, might cause buggy behavious")
            # read fps form eval
            with open(os.path.join(hiob_executions[0], "evaluation.txt"), "r") as eval_txt:
                lines = eval_txt.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    key_val = line.split("=")
                    if key_val[0] == "frame_rate":
                        overall_fps += float(key_val[1])
                        overall_fps_singles.append(key_val[1])
                    elif key_val[0] == "se_frame_rate":
                        se_fps += float(key_val[1])
                        se_fps_singles.append(key_val[1])

        approaches[approach]["overall_fps"] = overall_fps / len(approaches[approach]["trackings"])  # mean
        approaches[approach]["overall_fps_singles"] = overall_fps_singles
        approaches[approach]["se_fps"] = se_fps / len(approaches[approach]["trackings"])  # mean
        approaches[approach]["se_fps_singles"] = se_fps_singles

        # calc std
        overall_fps_sd_helper = [float(x) - approaches[approach]["overall_fps"] for x in approaches[approach]["overall_fps_singles"]]
        overall_fps_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in overall_fps_sd_helper]), len(overall_fps_singles)))

        se_fps_sd_helper = [float(x) - approaches[approach]["se_fps"] for x in approaches[approach]["se_fps_singles"]]
        se_fps_sd = np.sqrt(np.divide(np.sum([x ** 2 for x in se_fps_sd_helper]), len(se_fps_singles)))

        approaches[approach]["overall_fps_sd"] = overall_fps_sd
        approaches[approach]["se_fps_sd"] = se_fps_sd

        if ("candidates" or "cand") in approach.lower():
            candidate_dicts[str(approach)] = approaches[approach]
        elif "dsst" in approach.lower():
            dsst_dicts[str(approach)] = approaches[approach]
        elif "no_se" in approach:
            pass

    # plot figures
    plot_candidates_framrate_fig(candidate_dicts, eval_folder)

    plot_dsst_framerate_fig(dsst_dicts, eval_folder)



def create_final_tab_comp(tracking_folders):
    df = pd.DataFrame(
        columns=["Approach", "Precision", "Success", "Average size score"])

    for tracking in tracking_folders:
        folder_type = determine_folder_type(tracking)
        if folder_type == "multiple_hiob_executions":
            hiob_exeutions = get_tracking_folders(tracking)
            for hiob_execution in hiob_exeutions:
                approach = get_approach_from_yaml(hiob_execution)

                # get prec succ
                with open(os.path.join(hiob_execution, "evaluation.txt"), "r") as eval_txt:
                    lines = eval_txt.readlines()
                    for line in lines:
                        line = line.replace("\n", "")
                        key_val = line.split("=")
                        if key_val[0] == "total_precision_rating":
                            total_prec = float(key_val[1])
                        elif key_val[0] == "total_success_rating":
                            total_succ = float(key_val[1])

                # get size values for each sequence
                files_in_execution = os.listdir(hiob_execution)
                sequences_in_execution = []
                for file in files_in_execution:
                    if "tracking" and os.path.isdir(os.path.join(hiob_execution, file)):
                        sequences_in_execution.append(os.path.join(hiob_execution, file))

                curr_execution_size_scores = []
                for sequence in sequences_in_execution:
                    with open(os.path.join(sequence, "evaluation.txt"), "r") as seq_eval_txt:
                        lines = seq_eval_txt.readlines()
                        for line in lines:
                            line = line.replace("\n", "")
                            key_val = line.split("=")
                            if key_val[0] == "area_between_size_curves":
                                curr_execution_size_scores.append(float(key_val[1]))

                average_size_score = np.around(sum(curr_execution_size_scores) / len(curr_execution_size_scores),
                                               decimals=3)

                row = {"Approach": approach,
                       "Precision": total_prec,
                       "Success": total_succ,
                       "Average size score": average_size_score}

                df = df.append(row, ignore_index=True)

    if not os.path.isdir(path_multi_figure):
        os.mkdir(path_multi_figure)

    df.to_csv(os.path.join(path_multi_figure, "final_comp_table.csv"))
    print("saved comparison tabel to " + str(os.path.join(path_multi_figure, "final_comp_table.csv")))

    # write tex include file
    lines = [
        # "\\begin{table}[]\label{tab:asdasd}\n",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{\n",
        "\\begin{tabular}{@{}cccccc@{}}\n",
        "\\toprule\n",
    ]

    header_line = "\\textbf{Approach} & \\textbf{Precision} & \\textbf{Success} & \\textbf{Average size score} \\\\ \\midrule \n"
    lines.append(header_line)

    for i in range(df.shape[0]):
        line = ""
        for key in df.keys():
            curval = df.loc[i, key]
            if type(curval) == float:
                curval = np.around(curval, decimals=3)
            line += str(curval) + " & "

        # remove & at end of lline
        line = line[0:-2]
        line += "\\\\ \n"
        lines.append(line)
        print(line)

    lines[-1] = lines[-1][0:-2] + " \\bottomrule\n"
    lines.append("\\end{tabular}\n")
    lines.append("}\n")
    lines.append("\\caption{final comp table}\n")
    # lines.append("\\end{table}")

    if not os.path.isdir(path_multi_figure):
        os.mkdir(path_multi_figure)

    with open(os.path.join(path_multi_figure, "final_comp_table.tex"), "w") as tex_file:
        tex_file.writelines(lines)
    print("saved final comp table to " + str(os.path.join(path_multi_figure, "final_comp_table.tex")))


# ================================= HELPER FUNCTIONS =================================

def plot_dsst_framerate_fig(dsst_dicts, eval_folder):
    dsst_ind = np.arange(len(dsst_dicts))  # the x locations for the groups

    dsst_overall_fps = [dsst_dicts[approach]["overall_fps"] for approach in dsst_dicts]
    dsst_overall_fps_sd = [dsst_dicts[approach]["overall_fps_sd"] for approach in dsst_dicts]

    dsst_se_fps = [dsst_dicts[approach]["se_fps"] for approach in dsst_dicts]
    dsst_se_sd = [dsst_dicts[approach]["se_fps_sd"] for approach in dsst_dicts]

    fig, ax = plt.subplots()

    se_framerate_graph = ax.errorbar(dsst_ind, dsst_se_fps, yerr=dsst_se_sd, color='#785ef0', capsize=3)
    framerate_graph = ax.errorbar(dsst_ind, dsst_overall_fps, yerr=dsst_overall_fps_sd, color='#fe6100', capsize=3)

    ax.set_ylim(0, 60)
    ax.set_ylabel('Frame-rate')
    # fig.text(0.5, 0.04, parameter_names[str(parameter_name)], ha='center')
    ax.set_xticks(dsst_ind)
    ax.set_xticklabels(list(dsst_dicts.keys()))
    ax.legend((se_framerate_graph[0], framerate_graph[0]), ('Avg. SE Frame-rate', 'Avg. Frame-rate'))

    if not os.path.isdir(eval_folder):
        os.mkdir(eval_folder)

    figure_file2 = os.path.join(eval_folder, 'dsst_parameter_vs_framerate.pdf')
    plt.savefig(figure_file2)
    print("saved candidates framerates figure to " + figure_file2)


# plot the candidates framerate figure with split y axis
def plot_candidates_framrate_fig(candidate_dicts, eval_folder):
    cand_ind = np.arange(len(candidate_dicts))  # the x locations for the groups

    cand_overall_fps = [candidate_dicts[approach]["overall_fps"] for approach in candidate_dicts]
    cand_overall_fps_sd = [candidate_dicts[approach]["overall_fps_sd"] for approach in candidate_dicts]

    cand_se_fps = [candidate_dicts[approach]["se_fps"] for approach in candidate_dicts]
    cand_se_sd = [candidate_dicts[approach]["se_fps_sd"] for approach in candidate_dicts]

    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)

    cand_se_framerate_graph = ax.errorbar(cand_ind, cand_se_fps, yerr=cand_se_sd, color='#785ef0', capsize=3)
    ax.set_ylim((min(cand_se_fps) - max(cand_se_sd) * 5),
                (max(cand_se_fps) + max(cand_se_sd) * 5))
    ax.set_ylabel('SE Frame-rate')
    # fig.text(0.5, 0.04, parameter_names[str(parameter_name)], ha='center')
    ax.set_xticks(cand_ind)
    ax.set_xticklabels(list(candidate_dicts.keys()))

    cand_overall_framerate_graph = ax2.errorbar(cand_ind, cand_overall_fps, yerr=cand_overall_fps_sd, color='#fe6100', capsize=3)
    ax2.set_ylim((min(cand_overall_fps) - max(cand_overall_fps_sd) * 5), (max(cand_overall_fps) + max(cand_overall_fps_sd) * 5))
    ax2.set_ylabel('Frame-rate')
    ax2.set_xticks(cand_ind)
    ax2.set_xticklabels(list(candidate_dicts.keys()))

    # ax.legend((se_framerate_graph[0], framerate_graph[0]), ('Avg. SE Frame-rate', 'Avg. Frame-rate'))

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    if not os.path.isdir(eval_folder):
        os.mkdir(eval_folder)

    figure_file2 = os.path.join(eval_folder, 'cand_parameter_vs_framerate.pdf')
    plt.savefig(figure_file2)
    print("saved candidates framerates figure to " + figure_file2)


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
        return "matlab_sequence_file"
    elif found_hiob_execution and not (
            found_positions_txt or found_tracker_config or found_tracking_folder or found_mat_files):
        return "multiple_hiob_executions"


# return a positive int from a negative one (HIOB handles negative gts like positive ones...)
def pos_int_form_string(string_number):
    if int(string_number) <= 0:
        return int(string_number) * -1
    else:
        return int(string_number)


# return the approach to which a parameter belongs
def get_approach_to_parameter(parameter):
    if parameter in ["dsst_number_scales", "hog_cell_size", "learning_rate", "scale_factor", "scale_model_max",
                     "scale_sigma_factor"]:
        return "dsst"
    elif parameter in ["adjust_max_scale_diff", "adjust_max_scale_diff_after", "c_number_scales", "c_scale_factor",
                       "inner_punish_threshold", "max_scale_difference", "outer_punish_threshold",
                       "scale_window_step_size"]:
        return "candidates"
    else:
        raise ValueError("Parameter not found")


# read the information from the tracker.yaml file to reconstruct which tracking algorithm has been used
def get_approach_from_yaml(tracking_dir):
    folder_type = determine_folder_type(tracking_dir)

    # is for now only for when matlab results and the dsst validation folder are given as params for ptr
    if folder_type == 'matlab_tracking_folder':
        return "DSST reference"
    elif folder_type == 'multiple_hiob_executions':
        items_in_folder = os.listdir(tracking_dir)
        for item in items_in_folder:
            if 'hiob-execution' in item:
                approach = get_approach_from_yaml(os.path.join(tracking_dir, item))
                return approach

    with open(tracking_dir + "/tracker.yaml", "r") as stream:
        try:
            configuration = yaml.safe_load(stream)
            scale_estimator_conf = configuration["scale_estimator"]
        except yaml.YAMLError as exc:
            print(exc)

        algorithm = None

        if not scale_estimator_conf["use_scale_estimation"]:
            algorithm = "No se"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "custom_dsst" \
                and scale_estimator_conf["update_strategy"] == "cont" \
                and not scale_estimator_conf["d_change_aspect_ratio"]:
            algorithm = "DSST static continuous"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "custom_dsst" \
                and scale_estimator_conf["update_strategy"] == "limited" \
                and not scale_estimator_conf["d_change_aspect_ratio"]:
            algorithm = "DSST static limited"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "custom_dsst" \
                and scale_estimator_conf["update_strategy"] == "cont" \
                and scale_estimator_conf["d_change_aspect_ratio"]:
            algorithm = "DSST dynamic continuous"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "custom_dsst" \
                and scale_estimator_conf["update_strategy"] == "limited" \
                and scale_estimator_conf["d_change_aspect_ratio"]:
            algorithm = "DSST dynamic limited"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "candidates" \
                and scale_estimator_conf["update_strategy"] == "cont" \
                and scale_estimator_conf["c_change_aspect_ratio"]:
            algorithm = "Candidates dynamic continuous"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "candidates" \
                and scale_estimator_conf["update_strategy"] == "limited" \
                and scale_estimator_conf["c_change_aspect_ratio"]:
            algorithm = "Candidates dynamic limited"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "candidates" \
                and scale_estimator_conf["update_strategy"] == "cont" \
                and not scale_estimator_conf["c_change_aspect_ratio"]:
            algorithm = "Candidates static continuous"

        elif scale_estimator_conf["use_scale_estimation"] \
                and scale_estimator_conf["approach"] == "candidates" \
                and scale_estimator_conf["update_strategy"] == "limited" \
                and not scale_estimator_conf["c_change_aspect_ratio"]:
            algorithm = "Candidates static limited"

        return algorithm


# get the tracking dataset based on the tracking name
def get_dataset_from_name(tracking_name):

    folder_type = determine_folder_type(tracking_name)

    if folder_type == "multiple_hiob_executions":
        hiob_executions = get_tracking_folders(tracking_name)
        datasets = []
        for hiob_execution in hiob_executions:
            sequences = get_sequences(os.path.join(tracking_name, hiob_execution))
            for sequence in sequences:
                if "tb100" in sequence:
                    datasets.append("tb100")
                elif "nicovision" in sequence:
                    datasets.append("nicovision")
                else:
                    raise ValueError("No datase for sequence")
    else:
        print("here")

    if all("tb100" in tracking for tracking in datasets):
        return "TB100"
    elif all("nicovision" in tracking for tracking in datasets):
        return "NICO"
    else:
        raise ValueError("could not determine dataset")


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
                            "LR": [],
                            "bright": [],
                            "square": [],
                            "size-change": [],
                            "occlusion": [],
                            "dark": [],
                            "motion-blur": [],
                            "part-occlusion": [],
                            "non-square": [],
                            "contrast": []}

    if get_dataset_from_name(tracking_dir) == "TB100":
        dataset_path = tb100_attributes_path
    elif get_dataset_from_name(tracking_dir) == "NICO":
        dataset_path = nicovis_attributes_path

    with open(dataset_path, "r") as stream:
        try:
            data_set = yaml.safe_load(stream)
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
            for sample in data_set["samples"]:
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
            try:# to star a complete new run (aka do every experiment a second time, empty the log file or rename the )

                configuration = yaml.safe_load(stream)

                # if hog cell size not in configuration insert baseline values,
                # which have been used but where in hog ocode instead of config file
                if "hog_cell_size" not in configuration["scale_estimator"]:
                    configuration["scale_estimator"]["hog_cell_size"] = 1
                    configuration["scale_estimator"]["hog_block_norm_size"] = 4

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

    if len(changing_parameter_values) != 1:
        if "c_number_scales" in changing_parameter_values.keys():
            print("Multiple changing paramters: " + str(changing_parameter_values.keys()))
            print("Removing c_number_scales")
            del changing_parameter_values["c_number_scales"]

            if len(changing_parameter_values) != 1:
                raise ValueError(
                    "Still multiple changing parameters, aborting: " + str(changing_parameter_values.keys()))

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


# cut the first word from string and return for sort:
def ignore_leading_value(label):
    return " ".join(label.split(" ")[1:])


def main(results_path):
    print("git test")
    if len(results_path) == 1:
        results_path = results_path[0]
        folder_type = determine_folder_type(results_path)
        # just one sequence folder from one tracking folder
        if folder_type == "hiob_sequence_folder":
            print("detected single sequence folder")
            create_sequence_score_csv(results_path, "sequence_results_test")
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

        elif folder_type == "matlab_sequence_file":
            print("detected matlab_sequence file")
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
                # print("creating sequence boxplots")
                # sequence_boxplots(tracking_folder=results_path)
                print("detected multiple hiob executions, mode = exp")
                get_avg_results_from_experiment(results_path)
                print("creating graphs for average experiment metrics")
                create_graphs_metrics_for_set(results_path, "avg_full_set")

    elif len(results_path) >= 2:
        print("len >= 2")
        # -ptr /path/to/tracking1 /path/to/tracking2
        folder_types = [determine_folder_type(folder) for folder in results_path]
        print(str(folder_types))

        if args.task == "framerate_vs_approach":
            create_framerate_vs_approach_fig(trackings=results_path, eval_folder=path_multi_figure)
            create_framerate_tex_include(eval_folder=path_multi_figure)

        elif args.task == "gen_final_comp_tab":
            create_final_tab_comp(tracking_folders=results_path)

        elif 'matlab_tracking_folder' in folder_types:
            print("matlab tracking dir")
            multiple_trackings_graphs(tracking_folders=results_path,
                                      eval_folder=path_multi_figure,
                                      what_is_plotted="DSST reference vs implementation",
                                      font={'font.size': 15},
                                      tex_name="reference_vs_hiob_fig_include.tex",
                                      legend_by="algorithm")

        elif only_item_in_list('hiob_tracking_folder', folder_types) or only_item_in_list('multiple_hiob_executions', folder_types):
            if all("tb100full" in tracking for tracking in results_path):
                print("only tb100 trackings")
                multiple_trackings_graphs(tracking_folders=results_path,
                                          eval_folder=path_multi_figure,
                                          what_is_plotted="Approaches on TB100",
                                          font={'font.size': 10},
                                          tex_name="tb100full_all_approaches_fig_include.tex",
                                          legend_by="algorithm")



            elif all("nico" in tracking for tracking in results_path):
                print("only nico trackings")
                multiple_trackings_graphs(tracking_folders=results_path,
                                          eval_folder=path_multi_figure,
                                          what_is_plotted="Approaches on NICO",
                                          font={'font.size': 10},
                                          tex_name="nico_all_approaches_fig_include.tex",
                                          legend_by="algorithm")

            elif all("nico" or "tb100full" in tracking for tracking in results_path):
                print("nico and tb100 trackings")
                multiple_trackings_graphs(tracking_folders=results_path,
                                          eval_folder=path_multi_figure,
                                          what_is_plotted="Approaches on NICO and TB100",
                                          font={'font.size': 15},
                                          tex_name="nico_tb100_compare.tex",
                                          legend_by="dataset")

            else:
                print("no matching implementation for collection ofn tracking folders")

    else:
        print("nothing matches path")


if __name__ == "__main__":
    main(results_path)
