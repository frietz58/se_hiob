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

    ["scale_window_step_size", 0.005],
    ["scale_window_step_size", 0.0005],
    ["scale_window_step_size", 0.001],
    ["scale_window_step_size", 0.0015],
    ["scale_window_step_size", 0.002],
    ["scale_window_step_size", 0.0025]
    ["scale_window_step_size", 0.003],
    ["scale_window_step_size", 0.0035],
    ["scale_window_step_size", 0.004],
    ["scale_window_step_size", 0.0045],
    ["scale_window_step_size", 0.005],
    ["scale_window_step_size", 0.0055],
    ["scale_window_step_size", 0.006],
    ["scale_window_step_size", 0.0065],
    ["scale_window_step_size", 0.007],
    ["scale_window_step_size", 0.0075],
    ["scale_window_step_size", 0.008],
    ["scale_window_step_size", 0.0085],
    ["scale_window_step_size", 0.009],

]

for change in candidates_hyper_opt:
    set_keyval(change)
    print(change)
    subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])
    gc.collect()
