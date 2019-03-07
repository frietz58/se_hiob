import argparse
import os
import shutil

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
    double_folder_structure = False

    # if structure has a second folder (inconsistent in tb100 in dataset...)
    if sample_name in os.listdir(sequence_dir):
        double_folder_structure = True
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
            image_dir = os.path.join(sequence_dir, "imgs")
            print("changed foldername 'img' to 'imgs'")

            # get a list of everything in the folder and rename all jpg's where the first char is a zero
            imgs_content = os.listdir(image_dir)
            for img in imgs_content:
                if ".jpg" in img:
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

    if double_folder_structure:
        print("double dir structure, moving everything into main dir")
        for item in os.listdir(sequence_dir):
            target = "/".join(sequence_dir.split("/")[1:-1])
            target = "/" + target
            print("source, dst: {0}, {1}".format(os.path.join(sequence_dir, item), os.path.join(target, item)))
            shutil.move(os.path.join(sequence_dir, item), os.path.join(target, item))

    if sample_name == "David":
        images = os.listdir(image_dir)
        for image in images:
            image_number = image[3:-4]
            image_number = image_number.replace('0', '', 2)
            image_number = int(image_number)
            if image_number < 300:
                os.remove(os.path.join(image_dir, image))

        # shutil.copyfile(src, dst)
        create_folder(os.path.join(sequence_dir, "new_imgs"))

        new_images = os.listdir(image_dir)
        for image in new_images:

            image_number = image[3:-4]
            image_number = image_number.replace('0', '', 2)
            image_number = int(image_number)
            new_number = image_number - 299
            new_str = str(new_number)
            if len(new_str) == 1:
                new_str = '0000' + new_str
            elif len(new_str) == 2:
                new_str = '000' + new_str
            elif len(new_str) == 3:
                new_str = '00' + new_str

            new_name = "img" + new_str + ".jpg"
            shutil.copyfile(os.path.join(image_dir, image), os.path.join(sequence_dir, 'new_imgs/' + new_name))
            os.remove(os.path.join(image_dir, image))

        os.rmdir(image_dir)
        os.rename(os.path.join(sequence_dir, 'new_imgs'), os.path.join(sequence_dir, 'imgs'))

        # make new frames file
        os.remove(os.path.join(sequence_dir, "David_frames.txt"))
        frame_file_name = sample_name + "_frames.txt"
        frames_txt = open(os.path.join(sequence_dir, frame_file_name), "w")
        frames_txt.write("1,471")
        frames_txt.close()
        print("created " + sample_name + "_frames.txt in " + sequence_dir)

    elif sample_name == "Freeman3":
        images = os.listdir(image_dir)
        for image in images:
            image_number = image[3:-4]
            image_number = image_number.replace('0', '', 2)
            image_number = int(image_number)
            if image_number > 460:
                os.remove(os.path.join(image_dir, image))

        # make new frames file
        os.remove(os.path.join(sequence_dir, "Freeman3_frames.txt"))
        frame_file_name = sample_name + "_frames.txt"
        frames_txt = open(os.path.join(sequence_dir, frame_file_name), "w")
        frames_txt.write("1,460")
        frames_txt.close()
        print("created " + sample_name + "_frames.txt in " + sequence_dir)

    elif sample_name == "Football1":
        images = os.listdir(image_dir)
        for image in images:
            image_number = image[3:-4]
            image_number = image_number.replace('0', '', 2)
            image_number = int(image_number)
            if image_number > 74:
                os.remove(os.path.join(image_dir, image))

        # make new frames file
        os.remove(os.path.join(sequence_dir, "Football1_frames.txt"))
        frame_file_name = sample_name + "_frames.txt"
        frames_txt = open(os.path.join(sequence_dir, frame_file_name), "w")
        frames_txt.write("1,74")
        frames_txt.close()
        print("created " + sample_name + "_frames.txt in " + sequence_dir)

    elif sample_name == "Freeman4":
        images = os.listdir(image_dir)
        for image in images:
            image_number = image[3:-4]
            image_number = image_number.replace('0', '', 2)
            image_number = int(image_number)
            if image_number > 283:
                os.remove(os.path.join(image_dir, image))

        # make new frames file
        os.remove(os.path.join(sequence_dir, "Freeman4_frames.txt"))
        frame_file_name = sample_name + "_frames.txt"
        frames_txt = open(os.path.join(sequence_dir, frame_file_name), "w")
        frames_txt.write("1,283")
        frames_txt.close()
        print("created " + sample_name + "_frames.txt in " + sequence_dir)


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def change_every_sequence(sequences_dir):
    # get all sequences
    sequences = []
    for item in os.listdir(sequences_dir):
        if os.path.isdir(os.path.join(sequences_dir, item)):
            sequences.append(os.path.join(sequences_dir, item))

    for sequence_dir in sequences:
        change_single_sequence(sequence_dir)
        print(sequence_dir)


if __name__ == "__main__":
    change_every_sequence(path)
