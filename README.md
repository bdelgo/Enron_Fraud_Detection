Project Description
===================

In this project I built an algorithm to identify "Person of Interest"
among Enron Employees. POI is someone who may have committed fraud based
on the public Enron financial and email dataset. Dataset is organized as
a dictionary of dictionaries in which each key-value pair corresponds to
one person. The dictionary key is the person\'s name, and the value is
another dictionary, which contains the names of all the features and
their values for that person. The dataset has 146 data points with 21
feature (variable) for each, including 'poi' feature which specifies
whether the related person is a suspect of fraud or not. The 'poi' will
be used as a label to classify our data to two classes: POI and non-POI!
There is only 18 data points which are labeled as POI and the rest is
non-POI.

Features
========

First of all, I removed "email\_address" from the features, because it
contains no useful information and it causes "could not convert string
to float" error in FeatureFormat function. I also removed
"shared\_receipt\_with\_poi" because this feature basically is encoding the 'poi' label for
each person as a feature!
I also added two new features: \"fraction\_from\_poi\" and
\"fraction\_to\_poi\" which are basically scaled features: first one is
\"from\_poi\_to\_this\_person\" divided by \"to\_messages\", and the
second one is \"from\_this\_person\_to\_poi\" divide by
\"from\_messages\". After adding and removing some features, I used
SelectKBest to find out the univariate score of each feature. (Table-1)

I also used feature\_missing\_values function to calculate the number of
missing values for each feature (Table-2) Since there is no data for
\"loan\_advances\" in 140 out of 143 data points, I removed this feature
from the data set.

Before I select any group of features based on their SelectKBest scores
or number of their missing values, I will test different combination of
features to see which combination results in a better score. Since the
number of data points is low, high number of features increases the risk
of overfitting specially for Decision Tree algorithm. In order to avoid
overfitting, I will test the combinations with less than 5 features for
decision tree algorithm.

**Table-1: SelectKBest Scores**

  |Feature                       |Score
  |----------------------------- |-------
  |exercised\_stock\_options     |24.8
  |salary                        |18.3
  |long\_term\_incentive         |9.9
  |loan\_advances                |7.2
  |other                         |4.2
  |director\_fees                |2.1
  |from\_messages                |0.2
  |total\_stock\_value           |24.2
  |fraction\_to\_poi             |16.4
  |restricted\_stock             |9.2
  |expenses                      |6.1
  |fraction\_from\_poi           |3.1
  |to\_messages                  |1.6
  |restricted\_stock\_d          |0.1
  |bonus                         |20.8
  |deferred\_income              |11.5
  |total\_payments               |8.8
  |from\_poi\_to\_this\_person   |5.2
  |from\_this\_person\_to\_poi   |2.4
  |deferral\_payments            |0.2

**Table-2: Number of missing values for each feature**

  |Feature                       |Number of Missing Values
  |----------------------------- |------------------------
  |exercised\_stock\_options     |42
  |salary                        |49
  |long\_term\_incentive         |78
  |loan\_advances                |140
  |other                         |52
  |director\_fees                |127
  |from\_messages                |57
  |total\_stock\_value           |18
  |fraction\_to\_poi             |57
  |restricted\_stock             |34
  |expenses                      |49
  |fraction\_from\_poi           |57
  |to\_messages                  |57
  |restricted\_stock\_d          |126
  |bonus                         |62
  |deferred\_income              |95
  |total\_payments               |20
  |from\_poi\_to\_this\_person   |57
  |from\_this\_person\_to\_poi   |57
  |deferral\_payments            |105


**best\_features** is the function that selects the best combination of
features. It works as follow:

1)  Select a combination of features

2)  Tune the algorithm if possible (I will describe this step in
    Question 4)

3)  Fit the classifier Algorithm and Predict the targets

4)  Find the test scores through the "**test\_classifier"** function

5)  If the sum of Precision and Recall is higher than the current
    maximum sum, update the maximum score

In Table-5 there are some of the feature combinations and their test
results for Decision Tree algorithm. The features importances are also
presented in the same table.

I also used **MinMaxScaler** to scale the features but it did not
improve the results!

Algorithms & Modeling
=====================

I started with two classification algorithms:
Decision-Tree, and Naïve-Bayes. Then for each algorithm, and for every
combination of sub-features, I calculated the evaluation metrics. In
Table-5 and Table-6, you can see some of the results for each algorithm
for different combinations of features.

The recall score for Naïve Bayes is no better than 0.3, which is not
good! Also the sum of precision and recall is higher for Decision Tree;
as a result, I chose the Decision Tree as the best algorithm for Enron
fraud detection. You can see the best result for each algorithm in
Table-3

**Table-3: Best Results for each algorithm**

  |Algorithm    |Features                              |Accu.|Prec.|Recall
  |-------------|--------------------------------------|-----|-----|------
  |Decision Tree|bonus, fraction\_to\_poi              |0.87 |0.65 |0.46
  |Naïve Bayes  |total\_stock\_value, fraction\_to\_poi|0.87 |0.69 |0.30


Parameter Tuning
================

The goal of parameter tuning is to get the best performance out of the
algorithm. It also can help avoiding the overfitting situation. For
example, small amount of Min-Sample-Split will let the Decision Tree to
go very deep and in that case it would be prone to overfitting.

Naïve Bayes does not have parameters that I need to tune. For Decision
Tree, first I tried to use GridSearchCV as follow: param\_grid =
{\'criterion\': \[\'entropy\', \'gini\'\], \'min\_samples\_split\':
\[2,20\]}

Unfortunately, the resulted scores were not satisfactory. So I
incorporated the parameter tuning into the feature/algorithm selection
procedure as follow:

1)  Select a combination of features

2)  Fit and Predict with Decision Tree with \'min\_samples\_split\' in
    the range of 2 to 21

3)  Test the scores with the **test\_classifier** function

4)  If the sum of Precision and Recall is higher than the current
    maximum sum, update the maximum score

Since there is only 143 data points, I chose 20 as the upper bound of
the \'min\_samples\_split'**.** The optimum value with the best results
is 13!

In Table-4, you can see some of the results for Decision-Tree with
features = \[\'bonus\', \'fraction\_to\_poi\'\] and different values of
**Min-Sample-Split. **

**Table-4: Tuning Decision Tree with different Min-Sample-Split values**

  |Min-Sample-Split     |accuracy   |precision  |recall
  |---------------------|-----------|-----------|------------
  |2                    |  0.776    |   0.433   |   0.395
  |3                    |  0.790    |   0.470   |   0.381
  |4                    |  0.784    |   0.450   |   0.355
  |5                    |  0.780    |   0.438   |   0.356
  |6                    |  0.781    |   0.440   |   0.346
  |7                    |  0.782    |   0.438   |   0.316
  |8                    |  0.788    |   0.443   |   0.358
  |9                    |  0.781    |   0.438   |   0.331
  |10                   |  0.787    |   0.457   |   0.347
  |11                   |  0.796    |   0.487   |   0.357
  |12                   |  0.809    |   0.529   |   0.393
  |13                   |  0.821    |   0.566   |   0.459
  |14                   |  0.866    |   0.649   |   0.464
  |15                   |  0.821    |   0.565   |   0.447
  |16                   |  0.810    |   0.532   |   0.405
  |17                   |  0.801    |   0.502   |   0.342
  |18                   |  0.792    |   0.468   |   0.308

Validation
==========

Validation is basically dividing the dataset into different Training and
Testing groups; then fitting the algorithm with training data and
evaluating its performance on the testing data. The classic mistake is
to train and test the algorithm with same data, which highly increases
the chance of overfitting. Since the number of data points in the Enron
dataset is relatively small, by simply dividing the data to two
different training and testing groups, there is still a good chance of
overfitting; therefore, instead of using a simple train\_test\_split, it
is better to use K-Fold cross validation. We can also shuffle the data
points\' order before splitting into folds.

Furthermore, since there is only a small number of POIs in Enron
dataset, there is a huge imbalance between the size of POI and non-POI
classes. In order to handle this imbalance, we can use the Stratified
Cross Validation which assigns data points to folds so that each fold
has approximately the same number of data points of each output class.
By using **StratifiedShuffleSplit** we can address all of these
concerns.

Evaluation
==========

I used Accuracy, Precision and Recall metrics to evaluate the algorithm.
Accuracy is simply the number of correctly predicted targets to the
total number of targets.

Precision shows how many of all the items labeled as positive truly
belong to the positive class. Recall shows how many of all the items
that are truly positive, are correctly classified as positive. In the
context of this project, for example, if our model predicts 10 POIs
while only 6 of those predicted POIs are true POIs, then the precision
of model would be 0.6. If there are 10 POIs in our data and the number
of true POIs predicted by our model is 4, then the recall of the model
would be 0.4

For the Enron dataset, Accuracy is not a significant metric. Since less
than 15% of the data points are POI, if the algorithm predicts all the
data point as non-POI, its accuracy would be still more than 80%. As a
result, Precision and Recall are more important evaluation metrics than
Accuracy.

**Table-5: Decision Tree Results for different combination of features**

|Min.S.S  |Features                                             |Importance         |Accur. |Prec.  |Rec.
|---------|-----------------------------------------------------|-------------------|-------|-------|----
|2        |exercised\_stock\_options,total\_payments, expenses  |0.338, 0.303, 0.359|0.830  |0.415  |0.472
|5        |bonus, expenses                                      |0.338, 0.662       |0.779  |0.401  |0.439
|7        |exercised\_stock\_options, total\_payments, expenses |0.608, 0.089, 0.302|0.837  |0.433  |0.451
|10       |exercised\_stock\_options, total\_payments           |0.682, 0.318       |0.852  |0.484  |0.487
|12       |bonus, fraction\_to\_poi                             |0.633, 0.367       |0.808  |0.527  |0.390
|13       |bonus, fraction\_to\_poi                             |0.619, 0.381       |0.866  |0.649  |0.463

**Table-6: Naïve Bayes Results for different combination of features**

|Feature combination                                                        |accur. |prec.  |rec.
|---------------------------------------------------------------------------|-------|-------|-----
|total\_stock\_value, fraction\_to\_poi                                     |0.869  |0.691  |0.300
|exercised\_stock\_options, total\_stock\_value, bonus                      |0.843  |0.486  |0.351
|'total\_stock\_value, bonus, restricted\_stock, expenses                   |0.859  |0.511  |0.342
|total\_stock\_value, bonus, fraction\_to\_poi, restricted\_stock, expenses |0.859  |0.511  |0.342
|exercised\_stock\_options, total\_stock\_value, bonus, fraction\_to\_poi,  |0.856  |0.495  |0.343
|restricted\_stock, expenses                                                |       |       |               
|exercised\_stock\_options, total\_stock\_value, bonus, salary, , expenses  |0.845  |0.449  |0.352
|fraction\_to\_poi, long\_term\_incentive                                   |       |       |
