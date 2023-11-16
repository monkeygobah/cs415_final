import numpy as np
import cv2
from sklearn.cluster import KMeans
import assessment
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

''' 
Author: Georgie Nahass
Class: CS415
Date: 11/15/23
Purpose: store the models for this project for clustering and classification
1. create bag of words feature vectors
2. attempt to classify images based on Euclidean distance from closest cluster with label
'''



def k_means(training_stacked, training_labels, data_train, data_test,extractor='sift',  predicting=False, bow=True):
    ''' 
    Purpose: use training data to cluster all the descriptors. CLuster centroid represents a visual word 
    Input: stacked training data ()
    '''
    n = int(training_stacked.shape[0])
    # k_list = [.001*n, .005*n, .01*n, .025*n, .05*n, .075*n, .1*n]
    k_list = [3, 5, 10,20, 100]
    accuracies = {}  
    performance_metrics = {}

    for k in k_list:
        k=int(k)
        print(k)
        # Perform k-means on the dataset
        kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
        kmeans.fit(training_stacked)
       # # Perform k-means on the dataset
        print(kmeans)
        # compactness,labels,centers = cv2.kmeans(training_stacked,k,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),10,cv2.KMEANS_RANDOM_CENTERS)
        
        # # Determine majority label for each cluster- first make a blank dictionary with k subdicts 
        # cluster_lists = {i: {'healthy': 0, 'tb': 0} for i in range(k)}
        
        if predicting:
            accuracies = assessment.cluster_predict_k_means(kmeans,k,training_labels,data_test, accuracies)
        
        if bow:
            print('made it here')
            train_histograms = assessment.bag_of_words(k,kmeans, data_train,extractor_type=extractor)
            test_histograms = assessment.bag_of_words(k, kmeans, data_test,extractor_type=extractor)
            support_vector(train_histograms, test_histograms, data_train, data_test, performance_metrics, k)
        
    return kmeans, accuracies, performance_metrics

def support_vector(train_histograms, test_histograms, data_train, data_test, performance_metrics, k):
    svc = SVC()
    svc.fit(train_histograms, data_train['label'])

    # Evaluate on test set
    predictions = svc.predict(test_histograms)
    accuracy = accuracy_score(data_test['label'], predictions)
    print(accuracy)
    performance_metrics[k] = accuracy