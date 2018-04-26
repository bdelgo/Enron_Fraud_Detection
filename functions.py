#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:55:33 2018

@author: behruz
"""
import matplotlib.pyplot

def feature_visualization(data_dict, feature1, feature2):
    """
    It draws a scatter plot with feature1 for X_values, 
    and feature2 for Y_values
    """
    features = [feature1, feature2]
    data = featureFormat(data_dict, features)
    for point in data:
        feature1_data = point[0]
        feature2_data = point[1]
        matplotlib.pyplot.scatter( feature1_data, feature2_data)
    
    matplotlib.pyplot.xlabel(feature1)
    matplotlib.pyplot.ylabel(feature2)
    matplotlib.pyplot.show()

def feature_missing_values(data_dict, feature):
    """
    It calculates the number of missing values for 
    a feature in whole data_dict and returns it
    """
    feature_nan_count = 0
    for key in data_dict:
        if data_dict[key][feature] == 'NaN':
            feature_nan_count +=1
    return feature_nan_count



def person_missing_values(data_dict, feature_list):
    """
    for every person in the dictionary, this function
    will calculate the number of features for which 
    there is no data available. It returns a dictionary
    with names as keys and number of features with missing
    data as values 
    """
    person_nan_count = {}
    for person in data_dict:
        nan_count = 0
        for feature in feature_list:
            if data_dict[person][feature] == 'NaN':
                nan_count +=1
        person_nan_count[person] = nan_count
    return person_nan_count


def computeFraction( poi_messages, all_messages ):
    """
    This function calculates the fraction of messages send/receive to poi by a
    person from all the messages send/receive by that person 
    """
    
    if ((poi_messages == 'NaN') or (all_messages == 'NaN')):
        fraction = 'NaN'
    else:
        fraction = round(float(poi_messages)/all_messages, 3)
    
    return fraction


def add_fractionpoi_to_dataset(data_dict):
    """
    In this function, two new features ('fraction_to_poi', 'fraction_from_poi'
    are calculated and then added to the dataset
    """
    for person in data_dict:
        
        from_poi_to_this_person = data_dict[person]["from_poi_to_this_person"]
        to_messages = data_dict[person]["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        data_dict[person]["fraction_from_poi"] = fraction_from_poi

        from_this_person_to_poi = data_dict[person]["from_this_person_to_poi"]
        from_messages = data_dict[person]["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
        data_dict[person]["fraction_to_poi"] = fraction_to_poi



from sklearn.cross_validation import StratifiedShuffleSplit
from feature_format import featureFormat, targetFeatureSplit

def test_classifier_without_print(clf, dataset, feature_list, folds = 1000):
    """
    this function is a copy test_classifier that returns the result instead of 
    printing them
    """
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    #features = MinMaxScaler().fit_transform(features)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return (accuracy, precision, recall, f1, f2)
    
    except:
        return (0,0,0,0,0)


from itertools import combinations

def best_features(algorithm, data_dict, features, max_cmb = 2, folds = 1000):
    """
    this finds the combination of features which results in the best scores for
    a specific algorithm. it returns the combination of features and the scores
    of accuracy, precision and recall for that combination
    """
    
    max_score = 0
    best_result = (('poi'), 0,0,0)
    if max_cmb > len(features): max_cmb = len(features)
    
    for i in range(2, max_cmb+1):
        feature_comb = combinations(features, i)
        for comb in list(feature_comb):
            features_list = ['poi'] + list(comb)
            clf = algorithm
            accuracy, precision, recall, f1, f2 = test_classifier_without_print(clf, data_dict, features_list, folds = folds)
            score = precision + recall
            if (score > max_score) and (precision >= 0.3) and (recall >= 0.3):
                max_score = score
                best_result = (comb, accuracy, precision, recall)
                #feature_importance = clf.feature_importances_
            
    
    return best_result


    