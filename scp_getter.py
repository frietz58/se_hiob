import argparse
from subprocess import Popen, PIPE
import os

# Use like:
# python scp_getter -s /informatik2/students/home/5rietz/BA/gws5_opt/candidates_opt
# -t /home/finn/parameter_opt_results/gws5_candidates -c all

parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-s', '--source', required=True, help="remote source folder")
parser.add_argument('-c', '--childs', nargs='+')
parser.add_argument('-t', '--target', required=True, help="local target folder")

args = parser.parse_args()

if args.childs is None:
    args.childs = "all"


def main():
    all_dsst_params = [
        "dsst_number_scales", "hog_cell_size", "learning_rate", "scale_factor", "scale_model_max", "scale_sigma_factor"]

    all_candidates_params = [
        "adjust_max_scale_diff", "adjust_max_scale_diff_after", "c_number_scales",
        "c_scale_factor", "inner_punish_threshold", "max_scale_difference", "outer_punish_threshold",
        "scale_window_step_size"]

    if "dsst" in args.source:
        for param in all_dsst_params:

            local_save_path = os.path.join(str(args.target), str(param))

            if not os.path.isdir(local_save_path):
                os.mkdir(local_save_path)

            if "all" in args.childs or param in args.childs:
                print("getting files for " + os.path.join(args.source, str(param)))
                call = "scp 5rietz@rzssh1.informatik.uni-hamburg.de:\'" + os.path.join(args.source, str(param),
                       "*.csv") + "\' " + str(local_save_path)
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                if err != b"":
                    print(err)

                call = "scp 5rietz@rzssh1.informatik.uni-hamburg.de:\'" + os.path.join(args.source, str(param),
                        "*.pdf") + "\' " + str(local_save_path)
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                if err != b"":
                    print(err)

                call = "scp 5rietz@rzssh1.informatik.uni-hamburg.de:\'" + os.path.join(args.source, str(param),
                        "*.tex") + "\' " + str(local_save_path)
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                if err != b"":
                    print(err)
                print("saved to " + str(local_save_path))
                print("")

    elif "candidates" in args.source:
        for param in all_candidates_params:

            local_save_path = os.path.join(str(args.target), str(param))

            if not os.path.isdir(local_save_path):
                os.mkdir(local_save_path)

            if "all" in args.childs or param in args.childs:
                print("getting files for " + os.path.join(args.source, str(param)))
                call = "scp 5rietz@rzssh1.informatik.uni-hamburg.de:\'" + os.path.join(args.source, str(param),
                       "*.csv") + "\' " + str(local_save_path)
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                if err != b"":
                    print(err)

                call = "scp 5rietz@rzssh1.informatik.uni-hamburg.de:\'" + os.path.join(args.source, str(param),
                        "*.pdf") + "\' " + str(local_save_path)
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                if err != b"":
                    print(err)

                call = "scp 5rietz@rzssh1.informatik.uni-hamburg.de:\'" + os.path.join(args.source, str(param),
                        "*.tex") + "\' " + str(local_save_path)
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                if err != b"":
                    print(err)
                print("saved to " + str(local_save_path))
                print("")




if __name__ == "__main__":
    main()
