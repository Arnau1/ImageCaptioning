# Required packages
import unicodedata
import numpy as np
import os
from PIL import Image
import pandas as pd
import cv2
from tqdm import tqdm
import pickle
import random

def look_char(dataframe, column):
    '''
    Perform statistical analysis over some set of strings located on a column of a given dataframe.
    The function prints the amount of unique characters, the characters itself and their number of appearances.
    '''
    chars = set()

    for sentence in dataframe[column]:
        for char in sentence:
            chars.add(char) # Add to the set of unique characters

    # Unique characters
    chars = sorted(chars)
    print(f'Unique characters: {len(chars)}')
    print(''.join(chars))
    print()

    # Character counts
    for char in chars:
        char_name = ('<control>' if (unicodedata.category(char) == 'Cc') else
            unicodedata.name(char))
        print(f'Character: {char_name}')
        titles_containing_char = dataframe[column][dataframe[column].str.contains(char, regex=False)]
        print(f'Appearances: {len(titles_containing_char)}')    
        print()



def look_images(dataframe, folder, column='Image_Name', extension='.jpg'):
    # Lists to store widths and heights
    widths = []
    heights = []

    # Iterate over the image names in the DataFrame with a progress bar
    for image_name in tqdm(dataframe[column], desc="Processing Images"):
        image_path = os.path.join(folder, image_name + extension)
        try:
            with Image.open(image_path) as img:
                width, height = img.size  # Get image dimensions
                widths.append(width)
                heights.append(height)
        except FileNotFoundError:
            print(f"Image not found: {image_path}")

    # Calculate statistics for widths and heights
    if widths and heights:  # Ensure there are valid dimensions
        width_stats = {
            "max": max(widths),
            "min": min(widths),
            "mean": np.mean(widths),
            "std": np.std(widths),
        }
        height_stats = {
            "max": max(heights),
            "min": min(heights),
            "mean": np.mean(heights),
            "std": np.std(heights),
        }

        # Print results
        print("Width Stats:")
        print(f"  Max: {width_stats['max']}, Min: {width_stats['min']}, Mean: {width_stats['mean']:.2f}, Standard Deviation: {width_stats['std']:.2f}")
        print("Height Stats:")
        print(f"  Max: {height_stats['max']}, Min: {height_stats['min']}, Mean: {height_stats['mean']:.2f}, Standard Deviation: {height_stats['std']:.2f}")
    else:
        print("No valid image dimensions were found.")



def train_val_test_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                         seed=None, save=False, name='set', show=True):
    """
    Splits a dictionary into train, validation, and test sets.

    Args:
        data (dict): The dictionary to split.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        test_ratio (float): Proportion of data for the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Three dictionaries (train_dict, val_dict, test_dict).
    """
    if seed is not None:
        random.seed(seed)
    
    # Ensure the ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."
    
    # Shuffle the keys
    keys = list(data.keys())
    random.shuffle(keys)
    
    # Calculate split indices
    total_items = len(keys)
    train_end = int(total_items * train_ratio)
    val_end = train_end + int(total_items * val_ratio)
    
    # Split keys
    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]
    
    # Create the dictionaries
    train_dict = {key: data[key] for key in train_keys}
    val_dict = {key: data[key] for key in val_keys}
    test_dict = {key: data[key] for key in test_keys}

    if save: # Store dictionaries as pickle        
        with open(f'train_{name}.pkl', 'wb') as f:
            pickle.dump(train_dict, f)
        with open(f'validation_{name}.pkl', 'wb') as f:
            pickle.dump(val_dict, f)
        with open(f'test_{name}.pkl', 'wb') as f:
            pickle.dump(test_dict, f)
    
    if show:
        print("Original:", len(data.keys()))
        print("Train:", len(train_dict.keys()))
        print("Validation:", len(val_dict.keys()))
        print("Test:", len(test_dict.keys()))
    
    return train_dict, val_dict, test_dict



def extract_data(source_folder = '/fhome/vlia08/ImageCaptioning/DataRaw/', Title = True, Ingredients = False, Instructions = False, IMG_SIZE = 255):
    """
    Extracts recipe data (title, ingredients, instructions, images) from the given source folder.

    Parameters:
    - source_folder (str): Path to the folder containing the dataset files.
    - Title (bool): Whether to include recipe titles in the output data.
    - Ingredients (bool): Whether to include ingredients in the output data.
    - Instructions (bool): Whether to include instructions in the output data.

    Returns:
    - data (list): A list of dictionaries where each dictionary represents a recipe with the specified data fields.
    """
    # Load the recipes metadata CSV file into a DataFrame
    recipes = pd.read_csv(os.path.join(source_folder, "Food\ Ingredients\ and\ Recipe\ Dataset\ with\ Image\ Name\ Mapping.csv"))

    # Initialize an empty list to store processed data
    data = []

    # Iterate over each row (recipe) in the recipes DataFrame
    for _, row in recipes.iterrows():
        # Initialize an empty dictionary to store the data for this recipe
        dictionary = {}

        # If the Title flag is True, include the recipe title
        if Title:
            title = row["Title"]
            dictionary['title'] = title

        # If the Ingredients flag is True, include the cleaned ingredients list
        if Ingredients:
            ingredients = row["Cleaned_Ingredients"]
            dictionary['ingredients'] = ingredients

        # If the Instructions flag is True, include the recipe instructions
        if Instructions:
            instructions = row['Instructions']
            dictionary['instructions'] = instructions

        # Construct the file name for the image corresponding to this recipe
        image_file = row["Image_Name"] + '.jpg'  # Assumes images are named in the CSV

        # Construct the full path to the image file
        image_path = os.path.join(source_folder, image_file)

        # Include the image path in the dictionary
        dictionary['image_path'] = image_path
        
        data.append(dictionary)
    return data
