# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
# [Clang 12.0.0 ]
# Embedded file name: /Users/georgienahass/Desktop/fall_classes_PhD/CS415/cs415_final/utils.py
# Compiled at: 2023-11-15 18:21:23
# Size of source mod 2**32: 5215 bytes
import cv2, numpy as np
import matplotlib.pyplot as plt
import pandas as pd, os

def determine_disease(filename):
    """
    Check to see if image is TB or not.
    Input: name of image (last str position is label)
    Output: Boolean value of disease or not
    """
    if filename[-5] == '1':
        return True
    else:
        return False


def increase_contrast(image, box_size=8, lim=2):
    """
    Increase contrast of image
    Input: Image
    Optional: box size and clip limit- can finetune these
    Output: contrast increased image  
    """
    clahe = cv2.createCLAHE(clipLimit=lim, tileGridSize=(box_size, box_size))
    return clahe.apply(image)


def sift_kp_des(image):
    """ 
    find sift keypoints and detection on an image
    Input: Image
    Output: sift kp and descriptors
    """
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)


def hog_descriptors(image, corner_coords):
    """ 
    Use harris corner detections as keypoints for HOG Feature description
    Input: image, corner coordiantes from HCD
    """
    # hog feature data from class
    cell_size = (16, 16)
    block_size = (16, 16)
    multiplier = 1
    winsize = np.multiply(block_size, multiplier)
    nbins = 9
    
    # instantiate Hog Descriptor
    hog = cv2.HOGDescriptor(_winSize=winsize, _blockSize=block_size, _blockStride=cell_size, _cellSize=cell_size, _nbins=nbins)
    hog_features = hog.compute(image, locations=[(int(x[1]), int(x[0])) for x in corner_coords])
    
    # Get length
    feature_len = hog_features.shape[0] // len(corner_coords)
    
    # reshape
    hog_features = hog_features.reshape(-1, feature_len)
    
    return hog_features


def harris_detection(image, blocksize=2, ksize=3, k=0.04):
    """ 
    get HCD corners from image. Idea is to isolate the hilar trunk
    input: image
    Optional: hyperpatameters for HCD which can be finetuned
    return: np array of detected corners
    """
    return cv2.cornerHarris(image, blocksize, ksize, k)


def threshold_image(image, min_val=127, max_val=255):
    """
    input: image
    Optional: hyperpatameter pixel values for thresholding 
    return: count of white pixels normalized by image size
    """
    _, thresh = cv2.threshold(image, min_val, max_val, cv2.THRESH_BINARY)
    white_pixels = np.sum(thresh == max_val)
    total_pixels = image.size
    return white_pixels / total_pixels


def prepare_data(data_train, extractor_type='sift'):
    """ 
    Prepare data for downstream analyses (BoW and Kmeans) by stacking training descriptors and labels to be correct shape
    Input: training dictionary, extractor type (autoset to sift, can also be hog)
    returns appropriate data and labels formatted for later analyses
    """
    
    if extractor_type  not in ('sift', 'hog'):
        raise ValueError('extraction type not in dictionary')
    training_stacked = []
    train_labels = []
    
    # iterate through training data based on extractor type and stack it. also fix labels
    for i, descriptors in enumerate(data_train[extractor_type]):
        if descriptors is not None:
            training_stacked.extend(descriptors)
            label = data_train['label'][i]
            train_labels.extend([label] * len(descriptors))
        
    return training_stacked, train_labels


def extraction(file, data_dict):
    """ 
    Function to extract Harris corners, sift keypoints, and threshold images and store results in data dictionaries
    Inputs: image, path to file to determine label
    outputs: dictionary containing the metrics
    """
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    # increase the contrast
    cont_image = increase_contrast(image)
    
    kp, des = sift_kp_des(cont_image)
    
    # img = cv2.drawKeypoints(cont_image, kp, cont_image, flags=(cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    
    # get harris corners above a certain threshld and use for HOG
    harris = harris_detection(cont_image, blocksize=20, ksize=11)
    corners = np.argwhere((harris > np.percentile(harris, 99.7)) & (harris < np.percentile(harris, 99.9)))
    hog_des = hog_descriptors(cont_image, corners)
    
    
    thresholds = [127, 150, 170, 200]
    # get percent white for every image and store in dict
    for threshold in thresholds:
        per_white = threshold_image(cont_image, min_val=threshold)
        
        if determine_disease(file):
            data_dict['wp'][str(threshold)].append(per_white)
        else:
            data_dict['wp'][str(threshold)].append(per_white)
    
    # store rest of the data in the dictionary
    if determine_disease(file):
        data_dict['label'].append(1)
        data_dict['sift'].append(des)
        data_dict['hog'].append(hog_des)
    else:
        data_dict['label'].append(0)
        data_dict['sift'].append(des)
        data_dict['hog'].append(hog_des)