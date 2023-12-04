import numpy as np
import cv2
from sklearn.cluster import KMeans
import assessment
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score



''' 
Author: Georgie Nahass
Class: CS415
Date: 11/15/23
Purpose: store the models for this project for clustering and classification
1. create bag of words feature vectors
2. attempt to classify images based on Euclidean distance from closest cluster with label
'''

''' 
Helper function to return the metrics from the machine learning models
'''
def get_metrics(true_labels, predicted_labels, predicted_probs):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_scored = f1_score(true_labels, predicted_labels)
    tn, fp, _, _ = confusion_matrix(true_labels, predicted_labels).ravel()
    specificity = tn / (tn + fp)
    auroc_score = roc_auc_score(true_labels, predicted_probs)

    return precision, recall, accuracy, f1_scored, specificity, auroc_score


def engine(training_stacked, training_labels, data_train, data_test,extractor='sift',  k_predicting=False, bow=True, tfidf= False):
    ''' 
    Purpose: use training data to cluster all the descriptors. CLuster centroid represents a visual word 
    Input: stacked training data (), training data for each descriptor, extractor type, some flags about whats happening
    streamlines modle evaluation and setup 
    '''
    n = int(training_stacked.shape[0])
    # k_list = [.001*n, .005*n, .01*n, .025*n, .05*n, .075*n, .1*n]
    k_list = [3, 5, 10,20, 100]
    # k_list = [100]
    performance_metrics_svc_bow = {}
    performance_metrics_rf_bow = {}
    performance_metrics_nb_bow = {}
    performance_metrics_mlp_bow = {}    
    

    accuracies = {}
    for k in k_list:
        k=int(k)
        print(f'k is {k}')
        # Perform k-means on the dataset
        kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
        kmeans.fit(training_stacked)
       # # Perform k-means on the dataset
        print(kmeans)

        if k_predicting:
            accuracies = assessment.cluster_predict_k_means(kmeans,k,training_labels,data_test,extractor, accuracies)
        
        if bow:

            print('made it here')
            train_histograms = assessment.bag_of_words(k,kmeans, data_train,extractor_type=extractor)
            test_histograms = assessment.bag_of_words(k, kmeans, data_test,extractor_type=extractor)
            print(len(train_histograms[train_histograms<0]))
            train_and_evaluate_model('svc', train_histograms, test_histograms, data_train['label'], data_test['label'], performance_metrics_svc_bow, k)
            train_and_evaluate_model('random_forest', train_histograms, test_histograms, data_train['label'], data_test['label'], performance_metrics_rf_bow, k)
            train_and_evaluate_model('naive_bayes', train_histograms, test_histograms, data_train['label'], data_test['label'], performance_metrics_nb_bow, k)
            train_and_evaluate_model('mlp', train_histograms, test_histograms, data_train['label'], data_test['label'], performance_metrics_mlp_bow, k)
            

    return kmeans, accuracies, performance_metrics_svc_bow, performance_metrics_rf_bow, performance_metrics_nb_bow, performance_metrics_mlp_bow


''' 
Function to set up and run the thresholding feature vector model training
'''
def threshold_engine(X_train, X_test, y_train, y_test):
        thresh_performance_metrics_svc = {}
        thresh_performance_metrics_rf = {}
        thresh_performance_metrics_nb = {}
        thresh_performance_metrics_mlp = {}
        
        train_and_evaluate_model('svc', X_train, X_test,y_train , y_test, thresh_performance_metrics_svc)
        train_and_evaluate_model('random_forest', X_train, X_test, y_train, y_test, thresh_performance_metrics_rf)
        train_and_evaluate_model('naive_bayes', X_train, X_test, y_train, y_test, thresh_performance_metrics_nb)
        train_and_evaluate_model('mlp', X_train, X_test, y_train, y_test, thresh_performance_metrics_mlp)
        
        return thresh_performance_metrics_svc, thresh_performance_metrics_rf, thresh_performance_metrics_nb, thresh_performance_metrics_mlp

''' 
Function to handle machine learning and performance evaluation of the thresholded feature vectors
'''
def train_and_evaluate_model(model_type, X_train, X_test, y_train, y_test, performance_metrics, k=0):
    if model_type == 'svc':
        model = SVC(kernel='rbf', probability=True)
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    elif model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predicted_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    precision, recall, accuracy, f1_scored, specificity, auroc_score = get_metrics(y_test, predictions, predicted_probs)

    performance_metrics[k] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_scored,
        'specificity': specificity,
        'auroc_score': auroc_score
    }




