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

for change in candidates_hyper_opt:
    set_keyval(change)
    print(change)
    subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])
    gc.collect()
