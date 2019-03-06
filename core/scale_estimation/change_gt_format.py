import argparse
import os
import errno

parser = argparse.ArgumentParser(description="Changes the Structure of a TB100 sequence so that is can be run with the "
                                             "old matlab dsst implementation. The imgs will be renamed, the frames.txt "
                                             "file generated and the gt.txt file will be formatted.")

parser.add_argument("-pts", "--path", help="Absolute path to sequences folder in a folder exists for each sequence."
                                           "is located.")
args = parser.parse_args()
path = args.path

# change working dir into sequence folder
os.chdir(path)
print("changed working dir to : " + os.getcwd())

# get the contents of the sequence folder
root_content = os.listdir(os.getcwd())
print("content of the sequence dir: " + str(root_content))


def change_single_sequence(sequence_dir):
    print("")
    # get sample name for name of new file
    sample_name = sequence_dir.split("/")[-1]
    number_of_imgs = 0

    # if structure has a second folder (inconsistent in tb100 in dataset...)
    if sample_name in os.listdir(sequence_dir):
        sequence_dir = os.path.join(sequence_dir, sample_name)

    for item in os.listdir(sequence_dir):

        if item == "groundtruth_rect.txt":
            with open(os.path.join(sequence_dir, 'groundtruth_rect.txt'), "r") as file_object:
                # read the data
                text = file_object.read()
                formatted_text = text.replace("\t", ",")

                # write formatted data into new file
                new_gt = open(os.path.join(sequence_dir, sample_name + "_gt.txt"), "w")
                new_gt.write(formatted_text)
                new_gt.close()

            print("made a new file " + sample_name + "_gt.txt file in " + path)

        elif item == "img":
            # rename the folder
            os.rename(os.path.join(sequence_dir, "img"), os.path.join(sequence_dir, "imgs"))
            print("changed foldername 'img' to 'imgs'")

            # get a list of everything in the folder and rename all jpg's where the first char is a zero
            imgs_content = os.listdir(sequence_dir + '/imgs')
            for img in imgs_content:
                if ".jpg" in img and img[0] == "0":
                    number_of_imgs += 1
                    full_img_path = os.path.join(sequence_dir, 'imgs/' + img)
                    target = sequence_dir + "/imgs/img0" + img
                    os.rename(full_img_path, target)

            # make new frames file
            frame_file_name = sample_name + "_frames.txt"
            frames_txt = open(os.path.join(sequence_dir, frame_file_name), "w")
            frames_txt.write("1," + str(number_of_imgs))
            frames_txt.close()
            print("created " + sample_name + "_frames.txt in " + sequence_dir)


def change_every_sequence(sequences_dir):
    # get all sequences
    sequences = []
    for item in os.listdir(sequences_dir):
        if os.path.isdir(os.path.join(sequences_dir, item)):
            sequences.append(os.path.join(sequences_dir, item))

    for sequence_dir in sequences:
        print(sequence_dir)
        change_single_sequence(sequence_dir)


if __name__ == "__main__":
    change_every_sequence(path)
