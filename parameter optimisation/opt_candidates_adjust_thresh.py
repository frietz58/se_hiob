import subprocess
import ruamel.yaml
import numpy as np
import gc


def set_keyval(key_val_list):
    file_name = "/home/finn/PycharmProjects/code-git/HIOB/config/tracker.yaml"

    yaml = ruamel.yaml.YAML()

    with open(file_name) as f:
        tb100 = yaml.load(f)

        for key_val in key_val_list:
            key = key_val[0]
            val = key_val[1]

            tb100["scale_estimator"][key] = val

    with open(file_name, 'w') as f:
        yaml.dump(tb100, f)


# first run without changing, than run candidates approach, than dsst
# subprocess.call(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])

tracker_changes = [[["use_scale_estimation", False]],
                   [["use_scale_estimation", True], ["approach", "candidates"]],
                   [["use_scale_estimation", True], ["approach", 'custom_dsst']]]

candidates_hyper_opt = [

    [["adjust_max_scale_diff", False]],
    [["adjust_max_scale_diff", True], ["adjust_max_scale_diff_after", 1]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 2]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 3]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 4]],
    [["adjust_max_scale_diff", True], ["adjust_max_scale_diff_after", 5]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 6]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 7]],
    [["adjust_max_scale_diff", True], ["outer_punish_threshold", 8]],

]

for change in candidates_hyper_opt:
    set_keyval(change)
    print(change)
    subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])
    gc.collect()
