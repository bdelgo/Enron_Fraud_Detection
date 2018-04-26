#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import matplotlib.pyplot
from functions import feature_visualization, feature_missing_values 
from functions import person_missing_values, best_features, add_fractionpoi_to_dataset

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


all_features = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 
'exercised_stock_options', 'bonus', 'restricted_stock', 
'shared_receipt_with_poi', 'restricted_stock_deferred', 
'total_stock_value', 'expenses', 'loan_advances', 
'from_messages', 'other', 'from_this_person_to_poi',
'director_fees', 'deferred_income', 
'long_term_incentive', 'email_address', 'from_poi_to_this_person']

all_features.remove('email_address') #there is no usful information in this feature
all_features.remove('shared_receipt_with_poi') 


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    


### Task 2: Remove outliers
my_dataset = data_dict.copy()

#Scatter plots for pair of features
feature_visualization(my_dataset, 'salary','total_stock_value')

#Calculating the number of missing values for each person
nan_count_dict = person_missing_values(my_dataset, all_features)

sorted_by_nan_count =  sorted(nan_count_dict, key=nan_count_dict.get, reverse=True)

person_by_missing_value = []
for name in sorted_by_nan_count:
    person_by_missing_value.append((name, nan_count_dict[name]))
                           
print "\nFirts 10 data points with highest Missing Values:\n\n{}".format(person_by_missing_value[0:10]) 
                                                  
del my_dataset['THE TRAVEL AGENCY IN THE PARK']
del my_dataset['TOTAL']
del my_dataset['LOCKHART EUGENE E'] 

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#my_dataset = data_dict

# Here I create and add two new features to the data set
all_features.append("fraction_from_poi")
all_features.append("fraction_to_poi")
add_fractionpoi_to_dataset(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, ['poi'] + all_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Here I use SelectKBest to find the univariate score for each feature

from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k= 'all')
selector.fit(features, labels)

# Here I also calculate the number of missing values for each feature

feature_score = {}
missing_values ={}
for feature, score in zip(all_features, selector.scores_):
        missing_values[feature] = feature_missing_values(my_dataset, feature)
        feature_score[feature] = round(score,1)
        
print "\nNumber of Missing Values for every feature:\n\n", missing_values
print "\nSelectKBest Univariate Scores for every feature : \n\n", feature_score

#There is no data for "loan_advances" in 140 out of 143 data point. 
#So I remove it from the features 

all_features.remove('loan_advances')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.

def feature_select_for_NB(dataset, features):
    """
    Here the Naive-Bayes-Classifier is tested over all possible combinations of 
    all_features and optimum feature_list would be selected!
    (since it takes a lot of time, I just bypassed it here)
    """
    print "\n\nFindig the Optimum Feature List For Naive Bayes......."
    optimum_features, best_accuracy, best_precision, best_recall = best_features(GaussianNB(), dataset, features, max_cmb = len(features), folds = 1000)
    print "\nBest Features are: {}".format(optimum_features)
    print "Accuracy = {} Precision = {}  Recall = {}".format(best_accuracy, best_precision, best_recall)
    
    return(optimum_features)

#feature_select_for_NB(my_dataset, all_features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
#for i in range(2,20):
      
def feature_select_and_tune_for_DT(dataset, features):
    """
    Here the decision-tree-classifier is tuned over all possible combinations of 
    all_features. (since it takes a lot of time, I just bypassed it here 
    and used the results!)
    """
    print "\n\nFindig the Optimum Feature List For Decision Tree......."
    max_score = 0
    for i in range(2, 20): 
        dt = DecisionTreeClassifier(min_samples_split=i)
        comb, accuracy, precision, recall = best_features(dt, dataset, features, max_cmb = 4, folds = 100)
        # max_cmb is set to 4 in order to avoid overfitting over higher number of features
        
        if (precision + recall) > max_score:
            max_score = precision + recall
            optimum_features = list(comb)
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_minsplit = i
            
    print "\nBest Features are: {}".format(optimum_features)
    print "Accuracy = {} Precision = {}  Recall = {}".format(best_accuracy, best_precision, best_recall)
    print "Best min_samples_split = {}".format(best_minsplit)

    return (best_minsplit, optimum_features)
        
#best_minsplit, my_features = feature_select_and_tune_for_DT(my_dataset, all_features)
#features_list = ['poi'] + my_features
#clf = DecisionTreeClassifier(min_samples_split = best_minsplit)

features_list = ['poi','bonus', 'fraction_to_poi']
clf =  DecisionTreeClassifier(min_samples_split = 13)

### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)