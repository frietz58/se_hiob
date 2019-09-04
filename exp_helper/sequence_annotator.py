import argparse
import PIL
from PIL import Image, ImageDraw
import os
import numpy as np

parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-sd', '--sequence_dir')
parser.add_argument('-si', '--sequence_images')
parser.add_argument('-pd', '--predictions')
parser.add_argument('-gt', '--groundtruths')
parser.add_argument('-sp', '--save_path')
args = parser.parse_args()

# its enough to employ this, we can create the other args dynamically
args.sequence_dir = ("/informatik2/students/home/5rietz/WTM/ROOT_WTM_HIOB/hiob_logs/hiob-execution-"
                     "wtmpc30-2019-08-27-18.41.25.710050/tracking-0002-tb100-BlurBody")

sequence_name = os.path.basename(args.sequence_dir).split("-")[-1]
main_dir = os.path.dirname(args.sequence_dir)

args.sequence_images = os.path.join(args.sequence_dir, sequence_name, "img")
args.groundtruths = os.path.join(args.sequence_dir, sequence_name, "groundtruth_rect.txt")
args.predictions = os.path.join(args.sequence_dir, "tracking_log.txt")
args.save_path = os.path.join(args.sequence_dir, "annotated_imgs")

# GLOBALS:
DRAW_GT = True
DRAW_PRED = True
DRAW_ROI = False
DRAW_SE_EXECUTION_TEXT = False


def main():
    # get images
    images_files = os.listdir(args.sequence_images)
    images_files.sort()

    # get predictions
    with open(args.predictions, "r") as predicitions_txt:
        predictions = predicitions_txt.readlines()

    qualities = [None] * len(predictions)
    rois = [None] * len(predictions)
    for i, prediction in enumerate(predictions):
        # read the data we want from the log: prediction, roi, quality/confidence
        str_pred = prediction.replace("\n", "").split(",")[1:5]
        qualities[i] = float(prediction.replace("\n", "").split(",")[5])
        str_roi = prediction.replace("\n", "").split(",")[6:10]

        int_pred = [int(number) for number in str_pred]
        int_roi = [int(number) for number in str_roi]

        rois[i] = int_roi

        predictions[i] = int_pred

    if len(predictions) != len(images_files):
        images_files = images_files[0:len(predictions)]

    # get groundtruths
    with open(args.groundtruths, "r") as groundtruths_txt:
        ground_truths = groundtruths_txt.readlines()

    for i, ground_truth in enumerate(ground_truths):
        if "NaN" in ground_truth:
            # go back to find the last valid prediction
            j = 1
            while "NaN" in ground_truths[i - j]:
                j += 1
            # past valid positions have already been converted and saved in list, just use them
            int_gt = ground_truths[i - j]
        else:
            str_gt = ground_truth.replace("\n", "")
            if "," in str_gt:
                str_gt = str_gt.split(",")
            elif "\t" in str_gt:
                str_gt = str_gt.split("\t")
            int_gt = [int(number) for number in str_gt]

        ground_truths[i] = int_gt

    # if len(predictions) != len(ground_truths):
    #     ground_truths = ground_truths[0:len(predictions)]

    # make save dir
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    # annotate frame images
    for i in range(len(images_files)):
        pil_im = PIL.Image.open(os.path.join(args.sequence_images, str(images_files[i])))

        # convert s/w to colour:
        if pil_im.mode != "RGB":
            pil_im = pil_im.convert("RGB")

        # draw frame
        draw = ImageDraw.Draw(pil_im)

        # annotate whether SE has been executed on a frame or not
        if DRAW_SE_EXECUTION_TEXT:
            if "cont" in args.sequence_dir:
                draw.text((10, pil_im.height - 20),
                          str(np.around(qualities[i], decimals=3)) + " SE executed", fill="lime")

            elif "no_se" in args.sequence_dir:
                draw.text((10, pil_im.height - 20),
                          str(np.around(qualities[i], decimals=3)) + " SE skipped",
                          fill="red")
            else:
                # if qualities[i] >= 0.4 and qualities[i] <= 0.6:
                if 0.4 <= qualities[i] <= 0.6:
                    draw.text((10, pil_im.height - 20),
                              str(np.around(qualities[i], decimals=3)) + " SE executed", fill="lime")

                elif qualities[i] < 0.4 or qualities[i] > 0.6:
                    draw.text((10, pil_im.height - 20),
                              str(np.around(qualities[i], decimals=3)) + " SE skipped", fill="red")

        # get line representation of gt bounding box
        gt_rep = [ground_truths[i][0], ground_truths[i][1], ground_truths[i][0] + ground_truths[i][2],
                  ground_truths[i][1] + ground_truths[i][3]]
        gt_line_points = ((gt_rep[0], gt_rep[1]),
                          (gt_rep[2], gt_rep[1]),
                          (gt_rep[2], gt_rep[3]),
                          (gt_rep[0], gt_rep[3]),
                          (gt_rep[0], gt_rep[1]))

        # get line representation of prediction bounding box
        pred_rep = [predictions[i][0], predictions[i][1], predictions[i][0] + predictions[i][2],
                    predictions[i][1] + predictions[i][3]]
        pred_line_points = ((pred_rep[0], pred_rep[1]),
                            (pred_rep[2], pred_rep[1]),
                            (pred_rep[2], pred_rep[3]),
                            (pred_rep[0], pred_rep[3]),
                            (pred_rep[0], pred_rep[1]))

        # get line representation of roi bounding box
        roi_rep = [rois[i][0], rois[i][1], rois[i][0] + rois[i][2],
                   rois[i][1] + rois[i][3]]
        roi_line_points = ((roi_rep[0], roi_rep[1]),
                            (roi_rep[2], roi_rep[1]),
                            (roi_rep[2], roi_rep[3]),
                            (roi_rep[0], roi_rep[3]),
                            (roi_rep[0], roi_rep[1]))

        if DRAW_GT:
            draw.line(gt_line_points, fill=(0, 129, 1), width=4)

        if DRAW_PRED:
            draw.line(pred_line_points, fill=(253, 252, 16), width=4)

        if DRAW_ROI:
            draw.line(roi_line_points, fill=(25, 219, 213), width=4)

        pil_im.save(os.path.join(args.save_path, str(images_files[i])))


if __name__ == "__main__":
    main()
