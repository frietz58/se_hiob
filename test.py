import argparse
import sys

parser = argparse.ArgumentParser(description='Execute Parameter Optimization')
parser.add_argument('-t1', '--test1')
parser.add_argument('-t2', '--test2')
parser.add_argument('-t3', '--test3')

args = parser.parse_args()

#adasdasdasdasd this should chrash

print(sys.argv)