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

    save_to = "config/tracker_3.yaml"
    with open(save_to, 'w') as f:
        yaml.dump(tb100, f)


tracker_changes = [

    #[["c_number_scales", 21], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 21], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 21], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 21], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 21], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 21], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 27], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 27], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 27], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 27], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 27], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 27], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 33], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 33], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 33], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 33], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 33], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 33], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 37], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 37], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.007]],

   ##### [["c_number_scales", 37], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.003]],
   # [["c_number_scales", 37], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 37], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 37], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.003]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.004]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.005]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.006]],
    #[["c_number_scales", 37], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.007]],

    #[["c_number_scales", 41], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.003]],
    [["c_number_scales", 41], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.004]],
    [["c_number_scales", 41], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.005]],
    [["c_number_scales", 41], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.006]],
    [["c_number_scales", 41], ["max_scale_difference", 0.01], ["scale_window_step_size", 0.007]],

    [["c_number_scales", 41], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.003]],
    [["c_number_scales", 41], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.004]],
    [["c_number_scales", 41], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.005]],
    [["c_number_scales", 41], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.006]],
    [["c_number_scales", 41], ["max_scale_difference", 0.02], ["scale_window_step_size", 0.007]],

    [["c_number_scales", 41], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.003]],
    [["c_number_scales", 41], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.004]],
    [["c_number_scales", 41], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.005]],
    [["c_number_scales", 41], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.006]],
    [["c_number_scales", 41], ["max_scale_difference", 0.03], ["scale_window_step_size", 0.007]],

    [["c_number_scales", 41], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.003]],
    [["c_number_scales", 41], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.004]],
    [["c_number_scales", 41], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.005]],
    [["c_number_scales", 41], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.006]],
    [["c_number_scales", 41], ["max_scale_difference", 0.04], ["scale_window_step_size", 0.007]],

    [["c_number_scales", 41], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.003]],
    [["c_number_scales", 41], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.004]],
    [["c_number_scales", 41], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.005]],
    [["c_number_scales", 41], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.006]],
    [["c_number_scales", 41], ["max_scale_difference", 0.05], ["scale_window_step_size", 0.007]],

]


for change in tracker_changes:
    set_keyval(change)
    print(change)
    # subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])
    if args.gpu is None:
        args.gpu = 0
    subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker, '-g', str(args.gpu)])
    gc.collect()
