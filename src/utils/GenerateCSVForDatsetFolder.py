# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:27:25 2024

@author: lkang
"""

import os
import csv
import pandas as pd


def GenerateCSVBasedOnPercentage():
    # Load the CSV file
    input_csv = '/mnt/data/OCT2017-train.csv'
    df = pd.read_csv(input_csv)
    
    # Output CSV file
    output_csv = '/mnt/data/OCT2017-train-adv-0.1.csv'
    
    # Define the percentage
    percentage = 0.1
    
    # Initialize an empty DataFrame for the new CSV
    adv_df = pd.DataFrame(columns=df.columns)
    
    # Group by class and sample 10% of each class
    for class_name, group in df.groupby('label'):
        sample_size = int(len(group) * percentage)
        sampled_group = group.sample(n=sample_size, random_state=1)  # Use a fixed random state for reproducibility
        adv_df = pd.concat([adv_df, sampled_group])
    
    # Save the new DataFrame to a CSV file
    adv_df.to_csv(output_csv, index=False)
    
    print(f"New CSV with 10% of each class saved to: {output_csv}")

# Define the root directory
root_dir = '../OCT2017/test'  # Update this path

# Define the output CSV file
output_csv = 'OCT2017-test.csv'

# Dictionary to hold the count of images in each class
class_counts = {}

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['image', 'label'])
    
    # Iterate through each class folder
    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if os.path.isdir(class_path):
            # Get the label (assuming folder names are the labels)
            label = class_folder
            
            # Initialize the count for this class
            if label not in class_counts:
                class_counts[label] = 0
            
            # Iterate through each image in the class folder
            for image in os.listdir(class_path):
                image_path = os.path.join(class_path, image)
                if os.path.isfile(image_path):
                    # Write the image path and label to the CSV
                    writer.writerow([image, label])
                    # Increment the count for this class
                    class_counts[label] += 1

# Print the summary of image counts for each class
for label, count in class_counts.items():
    print(f"Class: {label}, Number of images: {count}")

print(f"CSV file '{output_csv}' generated successfully.")
