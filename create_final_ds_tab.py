import os
import argparse
import matplotlib

matplotlib.use('Agg')  # for plot when display is undefined
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

parser = argparse.ArgumentParser(description="")
parser.add_argument("-ptr", "--pathresults", nargs="+", required=True)
parser.add_argument("-s", "--savepath", required=True)
args = parser.parse_args()

# load both datasets
with open("config/data_sets/tb100.yaml", "r") as stream:
    try:
        tb100 = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open("config/data_sets/nicovision.yaml", "r") as stream:
    try:
        nicovision = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# true when folder is hiob execution
def is_hiob_exe(folder):
    return os.path.isdir(folder) and "hiob-execution" in folder


# get all sequence from one tracking folder
def get_sequences(tracking_dir):
    if os.path.isdir(tracking_dir):
        files_in_dir = os.listdir(tracking_dir)
    else:
        raise IOError("Tracking dir is not a directory")

    sequences = []
    for file in files_in_dir:
        if os.path.isdir(os.path.join(tracking_dir, file)) \
                and "tracking" in file \
                and "tb100" in file or "nicovision" in file:
            sequences.append(os.path.join(tracking_dir, file))

    return sequences


# read the information from the tracker.yaml file to reconstruct which tracking algorithm has been used
def get_approach_from_yaml(tracking_dir):
    # read configuration from yaml
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


# get attributes of a sequence
def get_sequence_attributes(sequence):
    sequence_name = os.path.basename(sequence).split("-")[-1]

    if any(values["name"] == sequence_name for values in nicovision["samples"]):
        for sample in nicovision["samples"]:
            if sample["name"] == sequence_name:
                sequence_attributes = sample["attributes"]
    elif any(values["name"] == sequence_name for values in tb100["samples"]):
        for sample in tb100["samples"]:
            if sample["name"] == sequence_name:
                sequence_attributes = sample["attributes"]

    return sequence_attributes


def create_multicolumn_csv_for_tab():
    # check that we have 3 hiob executions to fill the table with values
    if len(args.pathresults) != 3:
        raise IOError("Expected 3 hiob executions, got " + str(len(args.pathresults)))

    # get hiob executions from experiment folders like io_candidates_stat_cont_nico
    hiob_executions = []
    for experiment in args.pathresults:
        files_in_experiment = os.listdir(experiment)
        for file in files_in_experiment:
            if os.path.isdir(os.path.join(experiment, file)) and "hiob-execution" in file:
                hiob_executions.append(os.path.join(experiment, file))

    if len(hiob_executions) != len(args.pathresults):
        raise IOError("Expected to find " + str(
            len(args.pathresults) + " hiob executions but found " + str(len(hiob_executions))))

    # make sure all are hiob executions and determine the dataset of the three hiob executions
    hiob_executions_info_dict = {}
    unique_datasets = []

    for hiob_execution in hiob_executions:
        if not is_hiob_exe(hiob_execution):
            raise IOError(str(hiob_execution) + " is not a hiob execution")

        # get sequences
        sequences = get_sequences(hiob_execution)
        datasets = []
        sequence_metrics = {}
        for sequence in sequences:
            # dataset determination
            if "tb100" in sequence:
                datasets.append("tb100")
                sequence_dataset = "TB100"
            elif "nicovision" in sequence:
                datasets.append("nicovision")
                sequence_dataset = "NICO"
            else:
                raise ValueError("No datase for sequence")

            # get metric scores
            with open(os.path.join(sequence, "evaluation.txt"), "r") as eval_txt:
                lines = eval_txt.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    key_val = line.split("=")
                    if key_val[0] == "precision_rating":
                        sequence_prec = float(key_val[1])
                    elif key_val[0] == "success_rating":
                        sequence_succ = float(key_val[1])
                    elif key_val[0] == "area_between_size_curves":
                        sequence_size = float(key_val[1])

            # get attributes for sequence
            sequence_attributes = get_sequence_attributes(sequence)

            sequence_metrics[str(sequence)] = {
                "precision": sequence_prec,
                "success": sequence_succ,
                "size": sequence_size,
                "attributes": sequence_attributes
            }

        # determine overall tracking dataset
        if all("tb100" in tracking for tracking in datasets):
            dataset = "TB100"
            unique_datasets.append(dataset)
        elif all("nicovision" in tracking for tracking in datasets):
            dataset = "NICO"
            unique_datasets.append(dataset)
        else:
            raise ValueError("could not determine dataset")

        # get algorithm from yaml for csv setup
        algorithm = get_approach_from_yaml(hiob_execution)
        if hiob_execution not in hiob_executions_info_dict.keys():
            hiob_executions_info_dict[str(hiob_execution)] = {"algorithm": algorithm,
                                                              "dataset": dataset,
                                                              "sequences_scores": sequence_metrics}

    if all(unique_dataset == "TB100" for unique_dataset in unique_datasets):
        pass
        overall_datase = "TB100"
    elif all(unique_dataset == "NICO" for unique_dataset in unique_datasets):
        pass
        overall_datase = "NICO"
    else:
        raise ValueError("Hiob executions are on different datasets")

    # csv setup
    df = pd.DataFrame(columns=[
        "Algorithm",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[0]]["algorithm"]) + " Precision",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[0]]["algorithm"]) + " Success",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[0]]["algorithm"]) + " Size",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[1]]["algorithm"]) + " Precision",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[1]]["algorithm"]) + " Success",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[1]]["algorithm"]) + " Size",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[2]]["algorithm"]) + " Precision",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[2]]["algorithm"]) + " Success",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[2]]["algorithm"]) + " Size"])

    second_header_row = {
        "Attribute": "Metric",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[0]][
                "algorithm"]) + " Precision": "Precision",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[0]]["algorithm"]) + " Success": "Success",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[0]]["algorithm"]) + " Size": "Size",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[1]][
                "algorithm"]) + " Precision": "Precision",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[1]]["algorithm"]) + " Success": "Success",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[1]]["algorithm"]) + " Size": "Size",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[2]][
                "algorithm"]) + " Precision": "Precision",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[2]]["algorithm"]) + " Success": "Success",
        str(hiob_executions_info_dict[list(hiob_executions_info_dict.keys())[2]]["algorithm"]) + " Size": "Size"
    }

    df = df.append(second_header_row, ignore_index=True)

    tb100_attributes = {"IV": [],
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

    nico_attributes = {"bright": [],
                       "size-change": [],
                       "occlusion": [],
                       "dark": [],
                       "motion-blur": [],
                       "part-occlusion": [],
                       "non-square": [],
                       "contrast": []}

    if overall_datase == "NICO":
        attribute_collection = nico_attributes
    elif overall_datase == "TB100":
        attribute_collection = tb100_attributes

    for attribute in attribute_collection:  # row
        row_dict = {}
        row_dict["Attribute"] = attribute
        for key in hiob_executions_info_dict.keys():  # col
            # get prec succ size avg for attribute at algorithm
            hiob_execution = hiob_executions_info_dict[key]
            cell_precs = []
            cell_succs = []
            cell_sizes = []
            for seq_key in list(hiob_execution["sequences_scores"].keys()):
                sequence = hiob_execution["sequences_scores"][seq_key]
                if attribute in sequence["attributes"]:
                    cell_precs.append(float(sequence["precision"]))
                    cell_succs.append(float(sequence["success"]))
                    cell_sizes.append(float(sequence["size"]))
                    # cell_precs = [metrics["precision"] for metrics in list(hiob_execution["sequences_scores"].values())]
                    # cell_succs = [metrics["success"] for metrics in list(hiob_execution["sequences_scores"].values())]
                    # cell_sizes = [metrics["size"] for metrics in list(hiob_execution["sequences_scores"].values())]

            cell_avg_prec = np.around(np.sum(cell_precs) / len(cell_precs), decimals=3)
            cell_avg_succ = np.around(np.sum(cell_succs) / len(cell_succs), decimals=3)
            cell_avg_size = np.around(np.sum(cell_sizes) / len(cell_sizes), decimals=3)

            row_dict[hiob_execution["algorithm"] + " Precision"] = cell_avg_prec
            row_dict[hiob_execution["algorithm"] + " Success"] = cell_avg_succ
            row_dict[hiob_execution["algorithm"] + " Size"] = cell_avg_size

        df = df.append(row_dict, ignore_index=True)

    if not os.path.isdir(args.savepath):
        os.mkdir(args.savepath)
    csv_path = os.path.join(args.savepath, "nico_komplex_tab.csv")
    df.to_csv(csv_path, index=False)

    return csv_path


# create a tex file with the value from the tab csv
def create_tex_for_tab(csv_file, tex_name):
    df = pd.read_csv(csv_file)

    # normal tab header
    lines = [
         "\\begin{table}\n",
        "\\centering\n",
        "\\resizebox{\\textwidth}{!}{\n",
        "\\begin{tabular}{c c c c|c c c|c c c}\n",
        "\\toprule\n",
    ]

    # first row with the three algorithms
    algorithm_row_str = "\\multicolumn"
    algos_in_header = []
    for entry in list(df.columns):
        if entry == "Algorithm":
            algorithm_row_str += "{1}{c}{Algorithm} & "
        elif "No" in entry or "Candidates" in entry or "DSST" in entry:
            words = entry.split(" ")
            algorithm = " ".join(words[0:-1])
            if algorithm not in algos_in_header:
                if len(algos_in_header) == 2:
                    algorithm_row_str += "\\multicolumn{3}{c}{" + algorithm + "} & "  # last dont give bar at side
                else:
                    algorithm_row_str += "\\multicolumn{3}{c|}{" + algorithm + "} & "
                algos_in_header.append(algorithm)

    # remove last & at end of row and append//
    algorithm_row_str = algorithm_row_str[0:-2] + "\\\\\n"
    print(algorithm_row_str)
    lines.append(algorithm_row_str)

    lines.append("\\midrule\n")

    # second row with metrics and precision success size repeating three times
    pre_succ_size_row_str = "\\multicolumn{1}{c}{Metric} & Precision & Success & Size & Precision & Success & Size & Precision & Success & Size \\\\\n"
    print(pre_succ_size_row_str)
    lines.append(pre_succ_size_row_str)

    lines.append("\\midrule\n")

    # rows for an attribute for each metric of each algorithm
    for index, row in df.iterrows():
        attribute_row_str = ""
        # skip the pre succ size... row
        if row["Attribute"] == "Metric":
            continue

        attribute_row_str += row["Attribute"] + " & "

        # get the cell values by iterating over the row
        for column in row.keys():
            if column == "Algorithm":
                continue
            elif column == "Attribute":
                continue # attribute is already on row because it needs to be the first entry
            else:
                attribute_row_str += row[column] + " & "

        # remove last & at end of row and append//
        attribute_row_str = attribute_row_str[0:-2] + "\\\\\n"
        print(attribute_row_str)
        lines.append(attribute_row_str)

    # finish table
    lines.append("\\bottomrule\n")
    lines.append("\\end{tabular}\n")
    lines.append("}\n")
    lines.append("\\end{table}\n")

    tex_path = os.path.join(args.savepath, tex_name)
    print("saving table include tex to: " + str(tex_path))

    if not os.path.isdir(os.path.dirname(args.savepath)):
        os.mkdir(os.path.dirname(args.savepath))

    with open(tex_path, "w") as tex_file:
        tex_file.writelines(lines)



def main():
    tab_csv = create_multicolumn_csv_for_tab()
    create_tex_for_tab(tab_csv, "dataset_complex_tab.tex")


if __name__ == "__main__":
    main()
