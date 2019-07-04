import subprocess

import argparse

parser = argparse.ArgumentParser(description='Execute Parameter Optimization')
parser.add_argument('-g', '--gpu')
parser.add_argument('-e', '--environment')
parser.add_argument('-t', '--tracker')

args = parser.parse_args()

if __name__ == '__main__':

    if args.gpu is None:
        args.gpu = 0

    test = False

    for i in range(0, 10):
        print("execution  " + str(i))

        if not test:
            subprocess.call(['python', 'hiob_cli.py', '-e', args.environment, '-t', args.tracker, '-g', str(args.gpu)])
