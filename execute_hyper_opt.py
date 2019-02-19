import subprocess
import ruamel.yaml
import numpy as np
import gc

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
    [["inner_punish_threshold", 0.6], ["outer_punish_threshold", 0.6]],

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

    [["scale_window_step_size", 0.005]],
    [["scale_window_step_size", 0.0005]],
    [["scale_window_step_size", 0.001]],
    [["scale_window_step_size", 0.0015]],
    [["scale_window_step_size", 0.002]],
    [["scale_window_step_size", 0.0025]],
    [["scale_window_step_size", 0.003]],
    [["scale_window_step_size", 0.0035]],
    [["scale_window_step_size", 0.004]],
    [["scale_window_step_size", 0.0045]],
    [["scale_window_step_size", 0.005]],
    [["scale_window_step_size", 0.0055]],
    [["scale_window_step_size", 0.006]],
    [["scale_window_step_size", 0.0065]],
    [["scale_window_step_size", 0.007]],
    [["scale_window_step_size", 0.0075]],
    [["scale_window_step_size", 0.008]],
    [["scale_window_step_size", 0.0085]],
    [["scale_window_step_size", 0.009]]

]

for change in tracker_changes:
    set_keyval(change)
    print(change)
    subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])
    gc.collect()
