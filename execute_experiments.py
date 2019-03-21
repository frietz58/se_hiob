import argparse
from subprocess import Popen, PIPE

# Use like:
# python execute_experiments.py -g 0 -exp all // -exp exp1 exp2 exp3 exp4
parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-g', '--gpu')
parser.add_argument('-exp', '--experiments', nargs='+', help='<Required> Set flag', required=True)


args = parser.parse_args()

if __name__ == '__main__':

    print("specified GPU: " + args.gpu + ", experiments: " + str(args.experiments))
    print("")

    experiment_names = ["candidates_stat_cont_tb100full",
                        "candidates_stat_limited_tb100full",

                        "candidates_dyn_cont_tb100full",
                        "candidates_dyn_limited_tb100full",

                        "dsst_stat_cont_tb100full",
                        "dsst_stat_limited_tb100full",

                        "dsst_dyn_cont_tb100full",
                        "dsst_dyn_limited_tb100full"]

    for experiment in experiment_names:
        if "all" in args.experiments or experiment in args.experiments:
            call = "python hiob_cli.py -e config/env_" + experiment + ".yaml -t config/tracker_" + experiment \
                   + ".yaml -g " + args.gpu
            print("starting experiment " + experiment)

            p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, err = p.communicate(b"input data that is passed to subprocess' stdin")
            rc = p.returncode

            if "frame_rate" not in err.decode():
                print("EXPERIMENT " + experiment + " FAILED!")
                print("Log:")
                print(err.decode())
            else:
                print("finished experiment " + experiment)
                print("")


