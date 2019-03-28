import argparse
from subprocess import Popen, PIPE
import os

# Use like:
# python -t /local/folder -s /data/5rietz/test1 /data/5rietz/test1 -c 5rietz@rzssh1.informatik.uni-hamburg.de

parser = argparse.ArgumentParser(description='Execute hiob experiments')

parser.add_argument('-s', '--sources', nargs='+', required=True, help="remote source folders")
parser.add_argument('-t', '--target', required=True, help="local target folder")
parser.add_argument('-c', '--connect', required=True, help="remote host")

args = parser.parse_args()


def main():
    for source_folder in args.sources:

        print("getting files for " + str(source_folder) + " into " + str(args.target))
        local_source_name = name_helper(source_folder.split("/"))

        call = "scp -r " + str(args.connect) + ":" + str(source_folder) + " " \
               + os.path.join(str(args.target), local_source_name)
        print(call)

        p = Popen(call, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        if err != b"":
            print("failed for " + str(source_folder))
            print(err)
            print("")
        else:
            print("success for " + str(source_folder))
            print("")


def name_helper(list_of_strings):
    for word in list_of_strings:
        if "candidates" in word or "dsst" in word or "no_se" in word:
            return word


if __name__ == "__main__":
    main()
