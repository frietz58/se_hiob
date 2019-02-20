import subprocess
import ruamel.yaml
import gc
import argparse


parser = argparse.ArgumentParser(description='Execute Parameter Optimization')
parser.add_argument('-g', '--gpu')
parser.add_argument('-e', '--environment')
parser.add_argument('-t', '--tracker')

args = parser.parse_args()

def set_keyval(key_val_list):
    load_from = "config/backup.yaml"

    yaml = ruamel.yaml.YAML()

    with open(load_from) as f:
        tb100 = yaml.load(f)

        for key_val in key_val_list:
            key = key_val[0]
            val = key_val[1]

            tb100["scale_estimator"][key] = val

    save_to = "config/tracker.yaml"
    with open(save_to, 'w') as f:
        yaml.dump(tb100, f)


tracker_changes = [

    [["adjust_max_scale_diff", False]],
    [["adjust_max_scale_diff", True], ["adjust_max_scale_diff_after", 1]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 2]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 3]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 4]],
    [["adjust_max_scale_diff", True], ["adjust_max_scale_diff_after", 5]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 6]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 7]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 8]],

    [["inner_punish_threshold", 0.2], ["outer_punish_threshold", 0.2]],
    [["inner_punish_threshold", 0.2], ["outer_punish_threshold", 0.3]],
    [["inner_punish_threshold", 0.2], ["outer_punish_threshold", 0.4]],
    [["inner_punish_threshold", 0.2], ["outer_punish_threshold", 0.5]],
    [["inner_punish_threshold", 0.2], ["outer_punish_threshold", 0.6]],

    [["inner_punish_threshold", 0.3], ["outer_punish_threshold", 0.2]],
    [["inner_punish_threshold", 0.3], ["outer_punish_threshold", 0.3]],
    [["inner_punish_threshold", 0.3], ["outer_punish_threshold", 0.4]],
    [["inner_punish_threshold", 0.3], ["outer_punish_threshold", 0.5]],
    [["inner_punish_threshold", 0.3], ["outer_punish_threshold", 0.6]],

    [["inner_punish_threshold", 0.4], ["outer_punish_threshold", 0.2]],
    [["inner_punish_threshold", 0.4], ["outer_punish_threshold", 0.3]],
    [["inner_punish_threshold", 0.4], ["outer_punish_threshold", 0.4]],
    [["inner_punish_threshold", 0.4], ["outer_punish_threshold", 0.5]],
    [["inner_punish_threshold", 0.4], ["outer_punish_threshold", 0.6]],

    [["inner_punish_threshold", 0.5], ["outer_punish_threshold", 0.2]],
    [["inner_punish_threshold", 0.5], ["outer_punish_threshold", 0.3]],
    [["inner_punish_threshold", 0.5], ["outer_punish_threshold", 0.4]],
    [["inner_punish_threshold", 0.5], ["outer_punish_threshold", 0.5]],
    [["inner_punish_threshold", 0.5], ["outer_punish_threshold", 0.6]],

    [["inner_punish_threshold", 0.6], ["outer_punish_threshold", 0.2]],
    [["inner_punish_threshold", 0.6], ["outer_punish_threshold", 0.3]],
    [["inner_punish_threshold", 0.6], ["outer_punish_threshold", 0.4]],
    [["inner_punish_threshold", 0.6], ["outer_punish_threshold", 0.5]],
    [["inner_punish_threshold", 0.6], ["outer_punish_threshold", 0.6]]

]

for change in tracker_changes:
    set_keyval(change)
    print(change)
    #subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])
    if args.gpu is None:
        args.gpu = 0
    subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker,  '-g', str(args.gpu)])
    gc.collect()
