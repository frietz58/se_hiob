import argparse
import os

parser = argparse.ArgumentParser(description="Changes the Structure of a TB100 sequence so that is can be run with the "
                                             "old matlab dsst implementation. The imgs will be renamed, the frames.txt "
                                             "file generated and the gt.txt file will be formatted.")

parser.add_argument("-pts", "--path", help="Absolute path to sequence folder in which the img folder and gt.txt file "
                                           "is located.")
args = parser.parse_args()

# get the path from the commandline argument
path = args.path
print("Sequence folder path: " + path)

# change working dir into sequence folder
os.chdir(path)
print("changed working dir to : " + os.getcwd())

# get the contents of the sequence folder
root_content = os.listdir(os.getcwd())
print("content of the sequence dir: " + str(root_content))

# get sample name for name of new file
split_path = path.split("/")
sample_name = split_path[-1]
number_of_imgs = 0

for item in root_content:
    if item == "groundtruth_rect.txt":
        with open(item, "r") as file_object:
            # read the data
            text = file_object.read()
            formatted_text = text.replace("\t", ",")

            # write formatted data into new file
            new_gt = open(sample_name + "_gt.txt", "w")
            new_gt.write(formatted_text)
            new_gt.close()

        print("made a new file " + sample_name + "_gt.txt file in " + path)

    elif item == "img":
        # rename the folder
        os.rename("img", "imgs")
        print("changed foldername 'img' to 'imgs'")

        # go into imgs folder and rename all the jpg files accordingly
        os.chdir("./imgs")

        # get a list of everything in the folder and rename all jpg's where the first char is a zero
        imgs_content = os.listdir(os.getcwd())
        for img in imgs_content:
            if ".jpg" in img and img[0] == "0":
                number_of_imgs += 1
                os.rename(img, "img0" + img)
        print("renamed the img files from 0001.jpg to img00001.jpg")

        # go back into the main root dir and crate the frames.txt file
        os.chdir("../")
        frames_txt = open(sample_name + "_frames.txt", "w")
        frames_txt.write("1," + str(number_of_imgs))
        frames_txt.close()
        print("created " + sample_name + "_frames.txt in " + path)

print("done")
