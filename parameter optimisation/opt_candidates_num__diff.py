import subprocess
import ruamel.yaml
import numpy as np
import gc


def set_keyval(key_val_list):
    file_name = "/home/finn/PycharmProjects/code-git/HIOB/config/backup.yaml"

    yaml = ruamel.yaml.YAML()

    with open(file_name) as f:
        tb100 = yaml.load(f)

        for key_val in key_val_list:
            key = key_val[0]
            val = key_val[1]

            tb100["scale_estimator"][key] = val

    new_file = "/home/finn/PycharmProjects/code-git/HIOB/config/tracker.yaml"
    with open(new_file, 'w') as f:
        yaml.dump(tb100, f)


# first run without changing, than run candidates approach, than dsst
# subprocess.call(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])

tracker_changes = [[["use_scale_estimation", False]],
                   [["use_scale_estimation", True], ["approach", "candidates"]],
                   [["use_scale_estimation", True], ["approach", 'custom_dsst']]]

candidates_hyper_opt = [

    [["c_number_scales", 11], ["max_scale_difference", 0.01]],
    [["c_number_scales", 11], ["max_scale_difference", 0.02]],
    [["c_number_scales", 11], ["max_scale_difference", 0.03]],
    [["c_number_scales", 11], ["max_scale_difference", 0.04]],
    [["c_number_scales", 11], ["max_scale_difference", 0.05]],

    [["c_number_scales", 21], ["max_scale_difference", 0.01]],
    [["c_number_scales", 21], ["max_scale_difference", 0.02]],
    [["c_number_scales", 21], ["max_scale_difference", 0.03]],
    [["c_number_scales", 21], ["max_scale_difference", 0.04]],
    [["c_number_scales", 21], ["max_scale_difference", 0.05]],

    [["c_number_scales", 27], ["max_scale_difference", 0.01]],
    [["c_number_scales", 27], ["max_scale_difference", 0.02]],
    [["c_number_scales", 27], ["max_scale_difference", 0.03]],
    [["c_number_scales", 27], ["max_scale_difference", 0.04]],
    [["c_number_scales", 27], ["max_scale_difference", 0.05]],

    [["c_number_scales", 33], ["max_scale_difference", 0.01]],
    [["c_number_scales", 33], ["max_scale_difference", 0.02]],
    [["c_number_scales", 33], ["max_scale_difference", 0.03]],
    [["c_number_scales", 33], ["max_scale_difference", 0.04]],
    [["c_number_scales", 33], ["max_scale_difference", 0.05]],

    [["c_number_scales", 37], ["max_scale_difference", 0.01]],
    [["c_number_scales", 37], ["max_scale_difference", 0.02]],
    [["c_number_scales", 37], ["max_scale_difference", 0.03]],
    [["c_number_scales", 37], ["max_scale_difference", 0.04]],
    [["c_number_scales", 37], ["max_scale_difference", 0.05]],

    [["c_number_scales", 41], ["max_scale_difference", 0.01]],
    [["c_number_scales", 41], ["max_scale_difference", 0.02]],
    [["c_number_scales", 41], ["max_scale_difference", 0.03]],
    [["c_number_scales", 41], ["max_scale_difference", 0.04]],
    [["c_number_scales", 41], ["max_scale_difference", 0.05]],

]

for change in candidates_hyper_opt:
    set_keyval(change)
    print(change)
    subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])
    gc.collect()
