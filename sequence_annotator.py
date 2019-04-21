import argparse
import PIL
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-sd', '--sequence_dir')
parser.add_argument('-si', '--sequence_images')
parser.add_argument('-pd', '--predictions')
parser.add_argument('-gt', '--groundtruths')
parser.add_argument('-sp', '--save_path')

args = parser.parse_args()
args.sequence_dir = "/media/finn/linux-ssd/paper_dsst_dyn_cont_nico/hiob-execution-wtmgws11-2019-03-30-14.41.21.600499/tracking-0009-nicovision-lift_red_car_02"

sequence_name = os.path.basename(args.sequence_dir).split("-")[-1]
main_dir = os.path.dirname(args.sequence_dir)

args.sequence_images = os.path.join(args.sequence_dir, sequence_name, "img")
args.groundtruths = os.path.join(args.sequence_dir, sequence_name, "groundtruth_rect.txt")
args.predictions = os.path.join(args.sequence_dir, "tracking_log.txt")
args.save_path = os.path.join(args.sequence_dir, "annotated_imgs")


def main():
    # get images
    images_files = os.listdir(args.sequence_images)
    images_files.sort()

    # get predictions
    with open(args.predictions, "r") as predicitions_txt:
        predictions = predicitions_txt.readlines()

    qualities = [None] * len(predictions)
    for i, prediction in enumerate(predictions):
        str_pred = prediction.replace("\n", "").split(",")[1:5]
        qualities[i] = float(prediction.replace("\n", "").split(",")[5])
        int_pred = [int(number) for number in str_pred]

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
            int_gt = ground_truths[
                i - j]  # past valid positions have already been converted and saved in list, just use them
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

    # annotate frames with gt
    for i in range(len(images_files)):
        # print(predictions[i], ground_truths[i])
        pil_im = PIL.Image.open(os.path.join(args.sequence_images, str(images_files[i])))
        if pil_im.mode != "RGB":
            # convert s/w to colour:
            pil_im = pil_im.convert("RGB")

        draw = ImageDraw.Draw(pil_im)
        gt_rep = [ground_truths[i][0], ground_truths[i][1], ground_truths[i][0] + ground_truths[i][2],
                  ground_truths[i][1] + ground_truths[i][3]]
        gt_line_points = (
        (gt_rep[0], gt_rep[1]) , (gt_rep[2], gt_rep[1]), (gt_rep[2], gt_rep[3]), (gt_rep[0], gt_rep[3]), (gt_rep[0], gt_rep[1]))
        # draw.rectangle(gt_rep, fill=None, outline="green")
        draw.line(gt_line_points, fill="gold", width=4)

        # if "cont" in args.sequence_dir:
        #     draw.text((10, pil_im.height - 20), str(np.around(qualities[i], decimals=3)) + " SE executed", fill="lime")
        # elif "no_se" in args.sequence_dir:
        #     draw.text((10, pil_im.height - 20), str(np.around(qualities[i], decimals=3)) + " SE skipped",
        #               fill="red")
        # else:
        #     if qualities[i] >= 0.4 and qualities[i] <= 0.6:
        #         draw.text((10, pil_im.height - 20), str(np.around(qualities[i], decimals=3)) + " SE executed", fill="lime")
        #     elif qualities[i] < 0.4 or qualities[i] > 0.6:
        #         draw.text((10, pil_im.height - 20), str(np.around(qualities[i], decimals=3)) + " SE skipped", fill="red")

        pred_rep = [predictions[i][0], predictions[i][1], predictions[i][0] + predictions[i][2],
                    predictions[i][1] + predictions[i][3]]
        pred_line_points = (
            (pred_rep[0], pred_rep[1]), (pred_rep[2], pred_rep[1]), (pred_rep[2], pred_rep[3]), (pred_rep[0], pred_rep[3]), (pred_rep[0], pred_rep[1]))
        draw.rectangle(pred_rep, fill=None, outline="red")
        draw.line(pred_line_points, fill="magenta", width=4)

        pil_im.save(os.path.join(args.save_path, str(images_files[i])))


if __name__ == "__main__":
    main()
