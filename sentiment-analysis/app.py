import numpy as np
import pandas as pd
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import csv

def load_TrainingData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    feature_names  = np.array(list(D.columns.values))
    X_train = np.array(list(D['Phrase']))
    Y_train = np.array(list(D['Sentiment']))
    return  X_train, Y_train, feature_names

def load_TestingData(path):     #loads data , caluclate Mean & subtract it data, gets the COV. Matrix.
    D = pd.read_csv(path, sep='\t', header=0)
    X_test=np.array(list(D['Phrase']))
    X_test_PhraseID=np.array(list(D['PhraseId']))
    return  X_test,X_test_PhraseID

print "loading training data"
X_train, Y_train, _ = load_TrainingData("./train.tsv")
print X_train[:10]

print "loading testing data"
X_test, X_phrase_id = load_TestingData("./test.tsv")

print "splitting training data"
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

print x_train[:10]
print x_train.shape

print "applying vectorizer"
vectorizer = TfidfVectorizer(stop_words='english')
x_train_transformed = vectorizer.fit_transform(x_train)
x_test_transformed  = vectorizer.transform(x_test)
#actual Test (Capital X)
actual_test_transformed = vectorizer.transform(X_test)

print "applying svm algorithm"
# clf = svm.SVC(decision_function_shape='ovo')
clf = OneVsOneClassifier(LinearSVC())
# x_train = x_train.reshape(-1, 1)
clf.fit(x_train_transformed, y_train)

print "measuring the score"
score = clf.score(x_test_transformed, y_test)
print("the score is : ",score, "%")

submission = open('scikit_submission.csv', 'w')
csvwriter = csv.writer(submission, delimiter=',')
csvwriter.writerow(['PhraseId', 'Sentiment'])

predictions = clf.predict(actual_test_transformed)
for i in range(len(predictions)):
    csvwriter.writerow([str(X_phrase_id[i]), str(predictions[i])])