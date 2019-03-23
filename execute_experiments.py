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
# to star a complete new run (aka do every experiment a second time, empty the log file or rename the )
parser = argparse.ArgumentParser(description='Execute hiob experiments')
parser.add_argument('-g', '--gpu')
parser.add_argument('-exp', '--experiments', nargs='+', help='<Required> Set flag', required=True)

args = parser.parse_args()


def get_progress_from_log():
    log_file_path = fh.baseFilename
    progress = {"started": [],
                "finished": []}

    with open(log_file_path, "r") as logfile:
        for line in logfile:
            line = line.replace("\n", "")
            words = line.split(" ")

            if "INFO" in words:
                if "starting" in words:
                    progress["started"].append(words[-1])
                elif "finished" in words:
                    progress["finished"].append(words[-1])

    return progress


def main():
    logger.info("=========== NEW EXECUTION ===========")
    logger.info("specified GPU: " + args.gpu + ", experiments: " + str(args.experiments))

    experiment_names = [
        "dsst_validation0",
        "dsst_validation1",
        "dsst_validation2",
        "dsst_validation3",
        "dsst_validation4",
        "dsst_validation5",
        "dsst_validation6",
        "dsst_validation7",
        "dsst_validation8",
        "dsst_validation9",

        "candidates_stat_cont_tb100full",
        "candidates_stat_limited_tb100full",

        # "candidates_dyn_cont_tb100full",
        # "candidates_dyn_limited_tb100full",

        "dsst_stat_cont_tb100full",
        "dsst_stat_limited_tb100full",

        # "dsst_dyn_cont_tb100full",
        # "dsst_dyn_limited_tb100full"
    ]

    log_progress = True

    for experiment in experiment_names:

        if log_progress:
            progress = get_progress_from_log()
            logger.info("completed experiments so far: " + str(progress["finished"]))
            log_progress = False

        # if experiment has not been completed and its either an explicit parameter or all experiments are executed
        if experiment not in progress["finished"] and ("all" in args.experiments or experiment in args.experiments):

            logger.info("starting experiment " + experiment)

            # for dsst validation call extra script
            if "dsst_validation" in experiment:
                call = "python hiob_cli.py -e config/env_dsst_validation.yaml -t config/tracker_dsst_validation.yaml " \
                       + "-g " + args.gpu
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                rc = p.returncode

            # other experiments, call hiob_cli directly
            else:
                call = "python hiob_cli.py -e config/env_" + experiment + ".yaml -t config/tracker_" + experiment \
                       + ".yaml -g " + args.gpu
                p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate()
                rc = p.returncode

            if "frame_rate" not in err.decode():
                logger.error("EXPERIMENT " + experiment + " FAILED!")
                logger.error("Log:")
                logger.error(err.decode())
            else:
                logger.info("finished experiment " + experiment)

            # after a new experiment, log status
            log_progress = True

    logger.warning("EXECUTED EACH EXPERIMENT ONCE")


if __name__ == '__main__':
    main()
