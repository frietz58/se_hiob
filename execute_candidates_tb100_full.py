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

if __name__ == '__main__':

    if args.gpu is None:
        args.gpu = 0

    test = False

    for i in range(0, 3):
        print("execution  " + str(i))

        if not test:
            subprocess.call(['python', 'hiob_cli.py', '-e', 'config/environment_candidates.yaml', '-t', 'config/tracker_candidates.yaml', '-g', str(args.gpu)])
