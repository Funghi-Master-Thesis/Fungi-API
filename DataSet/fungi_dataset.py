import os
import re
import shutil
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import random

from Config import loc_config


class FungiDataset(torch.utils.data.Dataset):
    def __init__(self, annot_df, transform=None):
        self.annot_df = annot_df
        # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.root_dir = ""
        self.transform = transform

    def __len__(self):
        # return length (numer of rows) of the dataframe
        return len(self.annot_df)

    def __getitem__(self, idx):
        # use image path column (index = 1) in csv file
        image_path = self.annot_df.iloc[idx, 1]
        image = cv2.imread(image_path)  # read image by cv2
        # convert from BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # use class name column (index = 2) in csv file
        class_name = self.annot_df.iloc[idx, 2]
        # use class index column (index = 3) in csv file
        class_index = self.annot_df.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        # when accessing an instance via index, 3 outputs are returned - the image, class name and class index
        return image, class_name, class_index

    # def visualize(self, number_of_img=10, output_width=12, output_height=6):
    #     plt.figure(figsize=(output_width, output_height))
    #     for i in range(number_of_img):
    #         idx = random.randint(0, len(self.annot_df))
    #         image, class_name, class_index = self.__getitem__(idx)
    #         ax = plt.subplot(2, 140, i+1)  # create an axis
    #         # create a name of the axis based on the img name
    #         ax.title.set_text(class_name + '-' + str(class_index))
    #         if self.transform == None:
    #             plt.imshow(image)
    #         else:
    #             plt.imshow(image.permute(1, 2, 0))


def create_validation_dataset(dataset, validation_proportion):
    if (validation_proportion > 1) or (validation_proportion < 0):
        return "The proportion of the validation set must be between 0 and 1"
    else:
        dataset_size = int((1 - validation_proportion) * len(dataset))
        validation_size = len(dataset) - dataset_size
        print(dataset_size, validation_size)
        dataset, validation_set = torch.utils.data.random_split(
            dataset, [dataset_size, validation_size])
        return dataset, validation_set
    
def clean_folder_name(name):
    # Remove all non-letter characters (A-Z, a-z)
    return re.sub(r'[^A-Za-z]', '', name)

def clean_filename(name):
    # Remove all underscores
    return name.replace('_', '')

def rename_folders_and_files(root_directory):
    for dirpath, dirnames, filenames in os.walk(root_directory, topdown=False):
        # Rename files first
        for filename in filenames:
            old_file_path = os.path.join(dirpath, filename)
            new_filename = clean_filename(filename)
            new_file_path = os.path.join(dirpath, new_filename)

            if old_file_path != new_file_path:  # Check if the name has changed
                os.rename(old_file_path, new_file_path)
                print(f"Renamed file '{old_file_path}' to '{new_file_path}'")

        # Then rename folders
        for folder_name in dirnames:
            old_folder_path = os.path.join(dirpath, folder_name)
            new_folder_name = clean_folder_name(folder_name)
            new_folder_path = os.path.join(dirpath, new_folder_name)

            if old_folder_path != new_folder_path:  # Check if the name has changed
                os.rename(old_folder_path, new_folder_path)
                print(f"Renamed folder '{old_folder_path}' to '{new_folder_path}'")
def split_dataset_into_train_val(dataset):
    folder = loc_config.DATASET_LOC + "/" + dataset
    # build_csv(folder, 'dataset.csv')
    print("Splitting dataset into train, test, and validation sets...")
    base_dir = folder  # path to your dataset
    new_base_dir = loc_config.PROC_DATASET_LOC + "/" + dataset
    os.makedirs(new_base_dir, exist_ok=True)

    
    
    
    train_dir = os.path.join(new_base_dir, 'train')
    test_dir = os.path.join(new_base_dir, 'test')
    val_dir = os.path.join(new_base_dir, 'val')


    test_size = 0.1  # 20% for test
    val_size = 0.1   # 20% of remaining for validation
    # # rename_folders_and_files(train_dir)
    # # rename_folders_and_files(test_dir)
    # # rename_folders_and_files(val_dir)
    
    # Create directories for train, test, and validation splits
    for split_dir in [train_dir, test_dir, val_dir]:
        os.makedirs(split_dir, exist_ok=True)

    # DELETE
    # Pesisa-ostracoderma
    #
    # Loop through each class in the dataset
    for class_name in os.listdir(base_dir):
        if class_name == "test" or class_name == "train" or class_name == "validation":
            continue
        print("Doing: " + class_name)
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            # Get a list of files in the class folder
            files = os.listdir(class_path)
            files = [f for f in files if os.path.isfile(os.path.join(class_path, f))]
            if len(files) == 0:
                print(f"Skipping class '{class_name}' because it has no files")
                continue
            # Split files into train and test
            train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
            
            # Further split train into train and validation
            train_files, val_files = train_test_split(train_files, test_size=val_size, random_state=42)
            
            # Create class directories within train, test, and validation
            for split, file_list in zip([train_dir, test_dir, val_dir], [train_files, test_files, val_files]):
                split_class_dir = os.path.join(split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                
                # Move files to the respective directories
                for file_name in file_list:
                    src = os.path.join(class_path, file_name)
                    dest = os.path.join(split_class_dir, file_name)
                    shutil.copyfile(src, dest)
                    
    
