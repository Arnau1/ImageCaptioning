# Required packages
import unicodedata
import numpy as np
import os
from PIL import Image
import pandas as pd
import cv2
from tqdm import tqdm

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



def extract_data(source_folder, Title = True, Ingredients = False, Instructions = False, IMG_SIZE = 255):
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
    recipes = pd.read_csv(os.path.join(source_folder, "recipes.csv")) # Adjust if it's JSON or named differently

    # Define the path to the images directory
    images = os.path.join(source_folder, "images")  # Directory with images

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

        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Image not found: {image_file}")
                continue
             # Resize the image to the predefined size (IMG_SIZE)
            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0  # Normalize pixel values

            # Append title and image to data list
            dictionary['image'] = img
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
        data.append(dictionary)
    return data
