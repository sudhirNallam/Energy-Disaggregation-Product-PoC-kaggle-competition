'''
Created on Dec 5, 2015

@author: Joe
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

#high frequency model to predict appliance based on HF field
def model(freqData, target):
    model = LogisticRegression()
    
    
    model.fit(freqData, target) 
    prediction = model.predict(target)
    probabilities = model.predict_proba(freqData)
    return model, prediction, probabilities