import os
import numpy as np
import utils
import pickle


'''  
Author: Georgie Nahass
Class: CS415
Date: 11/15/23
IMAGE PREPROCESSING CODE
Run this to build initial training and testing dictionaries. Can add more features to extract if desired

Can also just load in dictionaries from pickle file to avoid having to use this
'''

def build_dictionaries(path):

    # store all of the filenames in a list
    filenames = [ ]
    for root, _, files in os.walk(path): 
        for file in files:
            if file.endswith('.png'):
                path_name = os.path.join(root, file) 
                filenames.append(path_name)
                
    # create random permutation of indices
    indices = np.random.permutation(len(filenames))

    # split into 80/20
    split = int(len(filenames) * 0.8)
    train_indices = indices[:split]
    test_indices = indices[split:]

    # split the filenames
    train_filenames = [filenames[i] for i in train_indices]
    test_filenames = [filenames[i] for i in test_indices]

    print("Training set size:", len(train_filenames))
    print("Testing set size:", len(test_filenames))
    
    data_train =  {'label' : [], 'sift': [], 'hog': [], 'wp' : {'127': [], '150': [], '170': [], '200': []}}
    data_test  =  {'label' : [], 'sift': [], 'hog': [], 'wp' : {'127': [], '150': [], '170': [], '200': []}}

    # populate train and test dictionaries separately 
    split_dicts = ['train', 'test']
    for split_type in split_dicts:
        # build training dictionary
        if split_type == 'train':
            for idx, filename in enumerate(train_filenames):
                print(f'Train image: {idx}')
                utils.extraction(filename, data_train) 
                
        # build testing dictionary
        else:
            for idx, filename in enumerate(test_filenames):
                print(f'Test image: {idx}')
                utils.extraction(filename, data_test)      
    return data_train, data_test


''' 
pickle data dictionaries to not have to extract features multiple times
'''
def pickle_out(data_train,data_test):
    # Save to a pickle file
    with open('data_train.pickle', 'wb') as f:
        pickle.dump(data_train, f)

    with open('data_test.pickle', 'wb') as f:
        pickle.dump(data_test, f)
        
        
def pickle_in():
    # Load from a pickle file
    with open('data_train.pickle', 'rb') as f:
        data_train_loaded = pickle.load(f)

    with open('data_test.pickle', 'rb') as f:
        data_test_loaded = pickle.load(f)
    return data_train_loaded, data_test_loaded

'''
TODO: Remove this function at end of project
USED TO VERIFY PICKLING WORKED. ChatGPT BUILT. CAN REMOVE WHEN FINSHED
    # Example usage:
    are_same, message = compare_key_in_dicts(data_train['wp'], data_train_loaded['wp'], '200')
    print("Are same:", are_same)
    if message:
        print("Message:", message)
'''
def compare_key_in_dicts(dict1, dict2, key):
    if key not in dict1 or key not in dict2:
        return False, "Key not found in one or both dictionaries"

    val1, val2 = dict1[key], dict2[key]

    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
        # Compare NumPy arrays
        return np.array_equal(val1, val2), None
    elif isinstance(val1, list) and isinstance(val2, list):
        # Compare lists (including lists of NumPy arrays)
        if len(val1) != len(val2):
            return False, "Lists have different lengths"
        for item1, item2 in zip(val1, val2):
            if isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
                if not np.array_equal(item1, item2):
                    return False, "Arrays in list do not match"
            elif item1 != item2:
                return False, "Elements in list do not match"
        return True, None
    else:
        # Direct comparison for other types
        return val1 == val2, None


