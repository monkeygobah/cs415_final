# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.13 (main, Aug 25 2022, 18:29:29) 
# [Clang 12.0.0 ]
# Embedded file name: /Users/georgienahass/Desktop/fall_classes_PhD/CS415/cs415_final/assessment.py
# Compiled at: 2023-11-15 20:38:49
# Size of source mod 2**32: 4218 bytes
import numpy as np
import cv2

def bag_of_words(k, kmeans, data, extractor_type='sift'):
    print('made it')
    histograms = np.zeros((len(data[extractor_type]), k))
    for i, descriptors in enumerate(data[extractor_type]):
        if descriptors is not None:
            visual_word_indices = kmeans.predict(np.array(descriptors))
            for idx in visual_word_indices:
                histograms[(i, idx)] += 1

    histograms = histograms / np.linalg.norm(histograms, axis=1, keepdims=True)
    print('end of bow')
    return histograms


def assign_label_to_descriptor(descriptor, centers, majority_labels):
    """ 
    Purpose: Take a descriptor and compute the distance to the closest centroid resulting from k means. Assign the descriptor to be a part of whatever
    the majority label of that cluster is
    Inputs: image descripotors, list of cluster centroids, list of labels of each cluster (index corresponds to closest number, value is label of cluster)
    Returns: majority label for the feature descriptor
    """
    # find the nearest center

    distances = [np.linalg.norm(descriptor - center) for center in centers]
    
    nearest_center_idx = np.argmin(distances)
    
    majority_label = majority_labels[nearest_center_idx]
    # Return the majority label of that cluster
    return majority_label


def cluster_predict_k_means(k_means, k, training_labels, data_test, accuracies):
    """ 
    Purpose: Make predictions on test image features using shortest euclidean distance from clusters
    Assign each cluster a label based on the majority vote of testing descriptors in it. Compare these to gt data and report accuracy
    Input: kmeans model, k value, training labels, testing data dict, accuracies dict
    Returns: accuracy dictionary to report the best k values
    """
    
    cluster_labels = k_means.labels_
    
    # Determine majority label for each cluster- first make a blank dictionary with k subdicts 
    cluster_lists = {i: {'healthy':0,  'tb':0} for i in range(k)}
    
    print(len(cluster_labels))
    # iterate through all of the labels from k means
    for idx, label in enumerate(cluster_labels):
        # find the associated ground truth label
        gt = training_labels[idx]
        
        # count number of correct predictions in each cluster
        if gt == 0:
            cluster_lists[label]['healthy'] += 1
        else:
            cluster_lists[label]['tb'] += 1


    # assign majority labels to all of the clusters
    majority_labels = {}
    for cluster, counts in cluster_lists.items():
        if counts['healthy'] > counts['tb']:
            majority_labels[cluster] = 0
        else:
            majority_labels[cluster] = 1
    
    # Evaluate the testing data using the clusters with a given K

    correct_predictions = 0
    # Loop over the list of descriptors for each image
    for idx, image_descriptors in enumerate(data_test['sift']):
        
        image_labels = [assign_label_to_descriptor(desc, k_means.cluster_centers_, majority_labels) for desc in image_descriptors]
        # Assign the majority label to the image
        predicted_label = max(image_labels, key=(image_labels.count))
        
        # grab the ground truth label from initial labels and compare
        ground_truth = data_test['label'][idx]
        if predicted_label == ground_truth:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(data_test['sift'])
    print(f"accuracy: {accuracy}")
    accuracies[k] = accuracy
    return accuracies