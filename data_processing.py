# Required packages
import unicodedata
import numpy as np
import os
from PIL import Image

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

    # Iterate over the image names in the DataFrame
    for image_name in dataframe[column]:
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
        }
        height_stats = {
            "max": max(heights),
            "min": min(heights),
            "mean": np.mean(heights),
        }

        # Print results
        print("Width Stats:")
        print(f"  Max: {width_stats['max']}, Min: {width_stats['min']}, Mean: {width_stats['mean']:.2f}")
        print("Height Stats:")
        print(f"  Max: {height_stats['max']}, Min: {height_stats['min']}, Mean: {height_stats['mean']:.2f}")
    else:
        print("No valid image dimensions were found.")
