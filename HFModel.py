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
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import timeit

#high frequency model to predict appliance based on HF field
def model(freqData, target):
    start = timeit.default_timer()
    neighbors = 5
    model = KNeighborsClassifier(neighbors)
    
    X_train, X_test, Y_train, Y_test = train_test_split(freqData, target, train_size=.80)
    
    model.fit(X_train, Y_train) 
    
    prediction = model.predict(X_test)
    
    print("Accuracy on test = %.3f" % metrics.accuracy_score(preditcion, Y_test))
    print("Accuracy on test = %.3f" % metrics.hamming_loss(preditcion, Y_test))

    stop = timeit.default_timer()
    print("Runtime: ",stop-start," seconds.")
    
    #return model, prediction, probabilities