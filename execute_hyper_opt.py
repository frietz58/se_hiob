import subprocess
import ruamel.yaml


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

# subprocess.call(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])

for change in tracker_changes:

    set_keyval(change)
    print(change)

    subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])


