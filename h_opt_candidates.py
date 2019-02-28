import subprocess
import ruamel.yaml
import gc
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Execute Parameter Optimization')
parser.add_argument('-g', '--gpu')
parser.add_argument('-e', '--environment')
parser.add_argument('-t', '--tracker')

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


def change_c_number_scales_old():
    print("Parameter: c_number_scales")
    c_number_scales_start = 33
    c_number_scales_step = 2
    c_number_scales_times = 5

    # make val bigger,
    for i in range(1, c_number_scales_times + 1):

        c_number_scales_val = c_number_scales_start + c_number_scales_step ** i
        print(c_number_scales_val)
        c_number_scales_change = [
            ["c_number_scales", c_number_scales_val],
            ["use_scale_estimation", True],
            ["use_update_strategies", False],
            ["approach", "custom_dsst"],
            ["c_change_aspect_ratio", False]
            ]

        set_keyval(key_val_list=c_number_scales_change, load_from="config/backup_tracker.yaml", save_to=args.tracker)

        environment_change = [["environment_name", "candidates"], ["log_dir", "../candidates_opt/c_number_scales"]]
        set_keyval(key_val_list=environment_change, load_from="config/environment.yaml", save_to=args.environment)

        if args.gpu is None:
            args.gpu = 0

        subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker, '-g', str(args.gpu)])
        gc.collect()
        print()

    # original val
    print(c_number_scales_start)
    c_number_scales_change = [
            ["c_number_scales", c_number_scales_start],
            ["use_scale_estimation", True],
            ["use_update_strategies", False],
            ["approach", "custom_dsst"],
            ["c_change_aspect_ratio", False]
            ]
    print()

    set_keyval(key_val_list=c_number_scales_change, load_from="config/backup_tracker.yaml",  save_to=args.tracker)

    if args.gpu is None:
        args.gpu = 0

    subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker, '-g', str(args.gpu)])
    gc.collect()

    # make val smaller,
    for i in range(1, c_number_scales_times + 1):

        c_number_scales_val = c_number_scales_start - c_number_scales_step ** i
        print(c_number_scales_val)
        c_number_scales_change = [
            ["c_number_scales", c_number_scales_val],
            ["use_scale_estimation", True],
            ["use_update_strategies", False],
            ["approach", "custom_dsst"],
            ["c_change_aspect_ratio", False]
            ]

        set_keyval(key_val_list=c_number_scales_change, load_from="config/backup_tracker.yaml",  save_to=args.tracker)

        if args.gpu is None:
            args.gpu = 0

        subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker, '-g', str(args.gpu)])
        gc.collect()
        print()


def change_parameter(parameter_name, start, step, times, change_function, test):
    parameter_name = str(parameter_name)
    print("Parameter: " + parameter_name)

    # make val bigger
    print("Increasing Parameter Value: ")
    for i in range(1, times + 1):

        # val_change = start + (step ** i) - 1
        val_change = change_function(start, step, i, "bigger")
        print(val_change)
        val_changes_bigger = [
            [str(parameter_name), val_change],
            ["use_scale_estimation", True],
            ["use_update_strategies", False],
            ["approach", "candidates"],
            ["c_change_aspect_ratio", False]
        ]

        set_keyval(key_val_list=val_changes_bigger, load_from="config/backup_tracker.yaml",
                   save_to=args.tracker)

        environment_change = [["environment_name", "candidates"],
                              ["log_dir", "../candidates_opt/" + str(parameter_name)]]
        set_keyval(key_val_list=environment_change, load_from="config/environment.yaml", save_to=args.environment)

        if not test:
            subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker, '-g', str(args.gpu)])
        gc.collect()
    print()

    # original val
    print("Base Value")
    print(start)
    start_val = [
        [str(parameter_name), start],
        ["use_scale_estimation", True],
        ["use_update_strategies", False],
        ["approach", "candidates"],
        ["c_change_aspect_ratio", False],
    ]
    print()

    set_keyval(key_val_list=start_val, load_from="config/backup_tracker.yaml", save_to=args.tracker)

    environment_change = [["environment_name", "candidates"],
                          ["log_dir", "../candidates_opt/" + str(parameter_name)]]
    set_keyval(key_val_list=environment_change, load_from="config/environment.yaml", save_to=args.environment)

    if not test:
        subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker, '-g', str(args.gpu)])
    gc.collect()

    # make val smaller
    print("Decreasing Parameter Value: ")
    for i in range(1, times + 1):

        # val_change = start  (step ** i) - 1
        val_change = change_function(start, step, i, "smaller")
        print(val_change)
        val_changes_smaller = [
            [str(parameter_name), val_change],
            ["use_scale_estimation", True],
            ["use_update_strategies", False],
            ["approach", "candidates"],
            ["c_change_aspect_ratio", False]
        ]

        set_keyval(key_val_list=val_changes_smaller, load_from="config/backup_tracker.yaml",
                   save_to=args.tracker)

        environment_change = [["environment_name", "candidates"],
                              ["log_dir", "../candidates_opt/" + str(parameter_name)]]
        set_keyval(key_val_list=environment_change, load_from="config/environment.yaml", save_to=args.environment)

        if not test:
            subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker, '-g', str(args.gpu)])
        gc.collect()
    print()


def change_c_number_scales(start, step, i, direction):
    if direction == "bigger":
        return float(np.around(start + step ** i, decimals=2))
    elif direction == "smaller":
        return float(np.around(start - step ** i, decimals=2))


def change_inner_punish_threshold(start, step, i, direction):
    if direction == "bigger":
        return float(np.around(start + (step ** i) - 1, decimals=2))
    elif direction == "smaller":
        return float(np.around(start - (step ** i) + 1, decimals=2))


def change_c_scale_factor(start, step, i, direction):
    # special case, we want it to only grow, so split growth over both direction cases...
    if direction == "bigger":
        return float(np.around(start + (step * i) - (1 * i), decimals=2))
    elif direction == "smaller":
        return float(np.around(start + 0.05 + (step * i) - (1 * i), decimals=2))


def change_max_scale_difference(start, step, i, direction):
    # special case, we want it to only grow, so split growth over both direction cases...
    if direction == "bigger":
        return float(np.around(start + (step * i) - (1 * i), decimals=2))
    elif direction == "smaller":
        return float(np.around(start + 0.05 + (step * i) - (1 * i), decimals=2))


if __name__ == '__main__':

    if args.gpu is None:
        args.gpu = 0

    test = True

    change_parameter(parameter_name='c_number_scales', start=33, step=2, times=5,
                     change_function=change_c_number_scales, test=test)
    print("==== new parameter ==== \n")
    change_parameter(parameter_name='inner_punish_threshold', start=0.5, step=1.05, times=5,
                     change_function=change_inner_punish_threshold, test=test)

    print("==== new parameter ==== \n")
    change_parameter(parameter_name='outer_punish_threshold', start=0.5, step=1.05, times=5,
                     change_function=change_inner_punish_threshold, test=test)

    print("==== new parameter ==== \n")
    change_parameter(parameter_name='c_scale_factor', start=1.01, step=1.01, times=5,
                     change_function=change_c_scale_factor, test=test)

    print("==== new parameter ==== \n")
    change_parameter(parameter_name='max_scale_difference', start=0.01, step=1.01, times=5,
                     change_function=change_max_scale_difference, test=test)

