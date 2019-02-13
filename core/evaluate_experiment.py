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

parser.add_argument("-p", "--path", help="Absolute path to the folder which contains the tracking logs of the "
                                         "experiment.")
args = parser.parse_args()

# get the path from the commandline argument
path = args.path

# get all subfolders
subfolders = [x[0] for x in os.walk(path)]
del subfolders[0]


def evaluate_single_sequence(sequence_dir):

    sequence_name = sequence_dir.split('-')[-1]

    # # read the gt_coords from the ground_truth file
    # global gt_coords
    # with open(os.path.join(sequence_dir, str(sequence_name + ".txt")), mode="r") as f:
    #     gt_coords = f.readlines()
    #     for i in range(0, len(gt_coords)):
    #         gt_coords[i] = gt_coords[i].replace("\n", "")
    #         gt_coords[i] = gt_coords[i].split(',')
    #         gt_coords[i] = list(map(int, gt_coords[i]))
    #
    # # parse lists into of coords into rects.
    # gt_rects = [None] * len(gt_coords)
    # for i in range(0, len(gt_coords)):
    #     gt_rects[i] = Rect(gt_coords[i][0], gt_coords[i][1], gt_coords[i][2], gt_coords[i][3])

    print(sequence_name)

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
                                                       working_line[i+1],
                                                       working_line[i+2],
                                                       working_line[i+3])
                i += 3
            elif i == 5:
                line_dict["prediction_quality"] = working_line[i]
            elif i == 6:
                line_dict["roi_position"] = Rect(working_line[i],
                                                 working_line[i+1],
                                                 working_line[i+2],
                                                 working_line[i+3])
                i += 3
            elif i == 10:
                line_dict["center_distamce"] = working_line[i]
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
                line_dict["size_score"] = working_line[i]

        print(line_dict)
        results.append(line_dict)


# traverse all sub folders and do evaluation for each sequence
for folder in subfolders:
    evaluate_single_sequence(folder)
