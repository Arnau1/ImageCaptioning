import pickle
import numpy as np

# Load the pkl file
with open('../Datasets/test_titles.pkl', 'rb') as f:
    titles = pickle.load(f)

# print the first 5 titles
print(titles) # a dictionary with keys 'image_name' and key 'title'