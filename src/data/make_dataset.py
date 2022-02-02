import os
import shutil
import PIL
import pathlib
import argparse
import PIL.Image
from pathlib import Path
from src.utils.basic_functions import get_config

# folder to load config file
root = Path(__file__).parents[2]
CONFIG_FILE = os.path.join(root, 'config.yaml')


# center crop images into a square
# new size is the minimum of width or height
# across all images
def square_crop_image(im: PIL.Image) -> PIL.Image:
    width, height = im.size
    new_size = min(width, height)

    # center crop
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2

    crop_im = im.crop((left, top, right, bottom))
    crop_im = crop_im.convert('RGB')

    return crop_im


# process all images for training
# square crop, downsampling to ensure balanced classes
# and output processed images to new folder
def make_dataset(in_folder: str, out_folder: str):

    # get number of images in each folder (images per class)
    file_count = []
    for fld in os.listdir(in_folder):
        image_count = len(os.listdir(os.path.join(in_folder, fld)))
        file_count.append(image_count)

    # if dealing with imbalanced classes
    # will downsample to the minimum number
    # of images per class
    im_per_class = min(file_count)

    # iterate through all folders (there should be one folder per object class)
    for fld in os.listdir(in_folder):
        # create the output folder for processed images for current class
        # delete folder and contents if there is one already
        out = os.path.join(out_folder, fld)
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)

        fld_path = pathlib.Path(os.path.join(in_folder, fld))
        num_images = 0
        for file in list(fld_path.glob('*')):
            # open image, center crop to a square
            # save to the output folder
            with PIL.Image.open(file) as im:
                crop_im = square_crop_image(im)
                crop_im.save(os.path.join(out, str(num_images) + '.jpg'))
                im.close()
            # break when desired number of images
            # has been processed (to keep classes balance)
            num_images = num_images + 1
            if num_images > im_per_class:
                break


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None,
                        help='What dataset to use for training the model ? Default: Animals-10')

    args = parser.parse_args()

    # get path where trained model resides
    dataset_name = args.dataset

    # read yaml file
    yaml_data = get_config(CONFIG_FILE)

    if not dataset_name:
        dataset_name = yaml_data['dataset']['default_train_data']

    # path to desired image set, relative to current working dir
    in_folder = os.path.join(root, yaml_data['dataset']['raw_data_path'], dataset_name)
    out_folder = os.path.join(root, yaml_data['dataset']['proc_data_path'], dataset_name)

    make_dataset(in_folder, out_folder)


if __name__ == '__main__':
    main()



