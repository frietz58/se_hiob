import argparse
from subprocess import Popen, PIPE
import logging
logger = logging.getLogger('experiment_execution')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('experiment_execution.log')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

# Use like:
# python execute_experiments.py -g 0 -exp all // -exp exp1 exp2 exp3 exp4
parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-g', '--gpu')
parser.add_argument('-exp', '--experiments', nargs='+', help='<Required> Set flag', required=True)

args = parser.parse_args()

if __name__ == '__main__':

    logger.info("starting continuous experiment execution")
    logger.info("specified GPU: " + args.gpu + ", experiments: " + str(args.experiments))
    logger.info("")

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
            logger.info("starting experiment " + experiment)

            p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, err = p.communicate(b"input data that is passed to subprocess' stdin")
            rc = p.returncode

            if "frame_rate" not in err.decode():
                logger.error("EXPERIMENT " + experiment + " FAILED!")
                logger.error("Log:")
                logger.error(err.decode())
            else:
                logger.info("finished experiment " + experiment)
                logger.info("")


