import argparse
from subprocess import Popen, PIPE
import os

# Use like:
# evaluate all parameters in an "opt folder" containing the folders with the trackings with the different param values
# python eval_executor -t /informatik2/students/home/5rietz/BA/gws5_opt/candidates_opt
# -c all // -c c_number_scales inner_punish_threshold

parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-c', '--childs', nargs='+', help="specify parameter_opt folder or all")
parser.add_argument('-t', '--target', required=True, help="folder containing parameter_opt folder")

args = parser.parse_args()

if args.childs is None:
    args.childs = "all"


def main():
    all_dsst_params = [
        "dsst_number_scales", "hog_cell_size", "learning_rate", "scale_factor", "scale_model_max", "scale_sigma_factor"]

    all_candidates_params = [
        "adjust_max_scale_diff", "adjust_max_scale_diff_after", "c_number_scales",
        "c_scales_factor", "inner_punish_threshold", "max_scale_difference", "outer_punish_Factor",
        "scale_window_step_size"]

    gt_path = "data/tb100/"
    attribute_path = "config/data_sets/tb100.yaml"
    mode = "opt"

    if "dsst" in args.target:
        for param in all_dsst_params:
            if "all" in args.childs or param in args.childs:
                print("evaluating " + os.path.join(args.target, str(param)))
                call = "python evaluate_results.py -ptgt " + gt_path + " -pta " + attribute_path + " -m " + mode + \
                       " -ptr " + os.path.join(args.target, str(param))
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                if err != b"":
                    print(err)
                else:
                    print("successfully finished evaluating " + param)
                    print("")

    elif "candidates" in args.target:
        for param in all_candidates_params:
            if "all" in args.childs or param in args.childs:
                print("evaluating " + os.path.join(args.target, str(param)))
                call = "python evaluate_results.py -ptgt " + gt_path + " -pta " + attribute_path + " -m " + mode + \
                       " -ptr " + os.path.join(args.target, str(param))
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                if err != b"":
                    print(err)
                else:
                    print("successfully finished evaluating " + param)
                    print("")


if __name__ == "__main__":
    main()
