# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:09:28 2024

@author: lkang
"""

import pandas as pd
import shutil
import os

def organize_images_by_label(csv_path, image_directory, output_directory):
    """
    Organize images into folders based on their labels.

    Parameters:
    - csv_path: Path to the CSV file containing 'image' and 'label' columns.
    - image_directory: Directory where the images currently reside.
    - output_directory: Directory where the labeled folders will be created.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Iterate over the DataFrame
    for index, row in df.iterrows():
        # Get the image file name and label
        image_file = f"{row['image']}.jpg"
        label = row['label']
        
        # Create a directory for the label if it doesn't exist
        label_dir = os.path.join(output_directory, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        # Construct the source and destination paths
        src_path = os.path.join(image_directory, image_file)
        dest_path = os.path.join(label_dir, image_file)
        
        # Check if the source file exists before copying
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        else:
            print(f"Warning: {src_path} does not exist.")

if __name__ == "__main__":
    csv_path = 'C:/Users/lkang/Documents/New UAP/ISIC2019_train.csv'  # Path to the CSV file
    image_directory = 'C:/Users/lkang/Documents/ISIC_2019_Training_Input/'  # Directory containing the images
    output_directory = 'C:/Users/lkang/Documents/ISIC_2019/'  # Directory where folders will be created

    organize_images_by_label(csv_path, image_directory, output_directory)
