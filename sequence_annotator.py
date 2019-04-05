import argparse
import PIL
from PIL import Image, ImageDraw
import os

parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-si', '--sequence_images')
parser.add_argument('-pd', '--predictions')
parser.add_argument('-gt', '--groundtruths')
parser.add_argument('-sp', '--save_path')

args = parser.parse_args()

args.sequence_images = "/media/finn/linux-ssd/paper_dsst_dyn_cont_tb100full/hiob-execution-wtmgws11-2019-03-30-05.32.11.810807/tracking-0099-tb100-Walking2/Walking2/img"
args.predictions = "/media/finn/linux-ssd/paper_dsst_dyn_cont_tb100full/hiob-execution-wtmgws11-2019-03-30-05.32.11.810807/tracking-0099-tb100-Walking2/tracking_log.txt"
args.save_path = "/media/finn/linux-ssd/paper_dsst_dyn_cont_tb100full/hiob-execution-wtmgws11-2019-03-30-05.32.11.810807/tracking-0099-tb100-Walking2/annotated_sequence"
args.groundtruths = "/media/finn/linux-ssd/paper_dsst_dyn_cont_tb100full/hiob-execution-wtmgws11-2019-03-30-05.32.11.810807/tracking-0099-tb100-Walking2/Walking2/groundtruth_rect.txt"


def main():
    # get images
    images_files = os.listdir(args.sequence_images)
    images_files.sort()

    # # get predictions
    # with open(args.predictions, "r") as predicitions_txt:
    #     predictions = predicitions_txt.readlines()
    #
    # for i, prediction in enumerate(predictions):
    #     str_pred = prediction.replace("\n", "").split(",")[1:5]
    #     int_pred = [int(number) for number in str_pred]
    #
    #     predictions[i] = int_pred
    #
    # if len(predictions) != len(images_files):
    #     images_files = images_files[0:len(predictions)]

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

        draw = ImageDraw.Draw(pil_im)
        gt_rep = [ground_truths[i][0], ground_truths[i][1], ground_truths[i][0] + ground_truths[i][2], ground_truths[i][1] + ground_truths[i][3]]
        draw.rectangle(gt_rep, fill=None, outline="green")
        # draw.text((10, pil_im.height - 20), "Sample Text", color="red")
        # draw.rectangle(predictions[i], fill=None, outline="red")
        pil_im.save(os.path.join(args.save_path, str(images_files[i])))


if __name__ == "__main__":
    main()

