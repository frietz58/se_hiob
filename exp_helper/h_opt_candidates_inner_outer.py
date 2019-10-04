import argparse
from subprocess import Popen, PIPE
import logging
import subprocess
import ruamel.yaml
import gc
import argparse
import numpy as np

logger = logging.getLogger('inner_outer_opt')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('inner_outer_opt.log')
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

parser = argparse.ArgumentParser(description='Execute Parameter Optimization')
parser.add_argument('-g', '--gpu', required=True)
parser.add_argument('-e', '--environment', required=True)
parser.add_argument('-t', '--tracker', required=True)
parser.add_argument('-tr', '--test_run')

args = parser.parse_args()


def set_keyval(key_val_list, load_from, save_to):
    # load_from = "config/backup_tracker.yaml"

    yaml = ruamel.yaml.YAML()

    with open(load_from) as f:
        yaml_file = yaml.load(f)

        for key_val in key_val_list:
            key = key_val[0]
            val = key_val[1]

            try:
                # if its the tracker.yaml
                yaml_file["scale_estimator"][key] = val
            except KeyError:
                # if its environment.yaml
                yaml_file[key] = val

    # save_to = "config/tracker_candidates.yaml"
    # save_to = args.tracker
    with open(save_to, 'w') as f:
        yaml.dump(yaml_file, f)


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
    if args.gpu is None:
        args.gpu = 0

    if args.test_run is None:
        args.test_run = True

    for i in np.arange(0.1, 1.1, 0.1):
        for j in np.arange(0.1, 1.1, 0.1):
            change_list = [
                ["inner_punish_threshold", float(np.around(i, decimals=1))],
                ["outer_punish_threshold", float(np.around(j, decimals=1))]
            ]

            set_keyval(key_val_list=change_list, load_from="config/" + args.tracker, save_to="config/" + args.tracker)

            exp_key = "inner" + str(np.around(i, decimals=1)) + "_outer" + str(np.around(j, decimals=1))

            progress = get_progress_from_log()

            if exp_key not in progress["finished"]:
                logger.info("completed experiments so far: " + str("\n".join(progress["finished"])))
                logger.info("curr change: " + str(change_list))
                logger.info("starting experiment " + exp_key)

                call = "python hiob_cli.py -e config/" + args.environment + " -t config/" + args.tracker + " -g " + args.gpu
                logger.info(call)
                krenew = "krenew"
                if args.test_run == "False":

                    p = Popen(krenew, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                    krenew_out, krenew_err = p.communicate()
                    krenew_rc = p.returncode

                    p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                    output, err = p.communicate()
                    rc = p.returncode

                    if "frame_rate" not in err.decode():
                        logger.error("EXPERIMENT " + exp_key + " FAILED!")
                        logger.error("Log:")
                        logger.error(err.decode())
                    else:
                        logger.info("finished experiment " + exp_key)

                else:
                    logger.info("test, not calling hiob_cli")


if __name__ == '__main__':
    main()
