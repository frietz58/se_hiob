import argparse
from subprocess import Popen, PIPE
import os

# Use like:
# evaluate all parameters in an "opt folder" containing the folders with the trackings with the different param values
# python run_opt_eval -t /informatik2/students/home/5rietz/BA/gws5_opt/candidates_opt
# -c all // -c c_number_scales inner_punish_threshold

parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-t', '--targets', nargs='+', required=True, help="folders containing hiob execution dir (with sequence results and evalautio.txt )")
args = parser.parse_args()


def main():

    gt_path = "data/tb100/"
    attribute_path = "config/data_sets/tb100.yaml"
    mode = "exp"

    for experiment in args.targets:

        print("evaluating " + str(experiment))
        call = "python evaluate_results.py -ptgt " + gt_path + " -pta " + attribute_path + " -m " + mode + \
               " -ptr " + str(experiment)
        p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        if err != b"":
            print("failed evaluation for: " + str(experiment))
            print(err)
            print("")
        else:
            print("successfully finished evaluating " + str(experiment))
            print("")


if __name__ == "__main__":
    main()
