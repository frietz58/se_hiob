import subprocess
import ruamel.yaml
import numpy as np
import gc


def set_keyval(key_val_list):
    load = "/home/finn/PycharmProjects/code-git/HIOB/config/backup.yaml"

    yaml = ruamel.yaml.YAML()

    with open(load) as f:
        tb100 = yaml.load(f)

        for key_val in key_val_list:
            key = key_val[0]
            val = key_val[1]

            tb100["scale_estimator"][key] = val

    safe = "/home/finn/PycharmProjects/code-git/HIOB/config/tracker.yaml"
    with open(safe, 'w') as f:
        yaml.dump(tb100, f)


candidates_hyper_opt = [
    [["test1", True]],
    [["test2", True]]
]

for change in candidates_hyper_opt:
    set_keyval(change)
    print(change)
    subprocess.run(['./execute_experiments.sh', 'config', 'config/environment_experiments.yaml'])
    gc.collect()
