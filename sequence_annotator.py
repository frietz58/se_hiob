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
args.sequence_dir = "/media/finn/linux-ssd/candidates_stat_cont_tb100full/hiob-execution-wtmgws11-2019-03-25-18.00.00.141037/tracking-0048-tb100-Freeman3"

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

    for i, prediction in enumerate(predictions):
        str_pred = prediction.replace("\n", "").split(",")[1:5]
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
            while "NaN" in ground_truths[i-j]:
                j += 1
            int_gt = ground_truths[i-j]  # past valid positions have already been converted and saved in list, just use them
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
        gt_rep = [ground_truths[i][0], ground_truths[i][1], ground_truths[i][0] + ground_truths[i][2], ground_truths[i][1] + ground_truths[i][3]]
        draw.rectangle(gt_rep, fill=None, outline="green")
        # draw.text((10, pil_im.height - 20), "Sample Text", color="red")
        pred_rep = [predictions[i][0], predictions[i][1], predictions[i][0] + predictions[i][2], predictions[i][1] + predictions[i][3]]
        draw.rectangle(pred_rep, fill=None, outline="red")
        pil_im.save(os.path.join(args.save_path, str(images_files[i])))


if __name__ == "__main__":
    main()

