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
import numpy as np
import pandas as pd

def dataPrep(HF, TagInfo):
    #Todo
    HF['Back Porch Lights'] = pd.Series(np.zeros(HF.shape[0]), index=HF.index)
    for c in range(0, HF.index.size):
        if inRange(HF.loc[c, 'Timestamp'], 'Back Porch Lights', TagInfo["Back Porch Lights" == TagInfo[:,1]]):
            HF.iloc[c, -1] = 1 #[HF.index[c][0]]#['Back Porch Lights']


def inRange(point, appliance, TagInfo):
    for c in range(0, TagInfo.shape[0]):
        if appliance == TagInfo[c][1][0][0][0]:
            # print TagInfo[c][2][0][0], TagInfo[c][3][0][0], point
            if TagInfo[c][2][0][0] <= point <= TagInfo[c][3][0][0]:
                return True
    return False

#high frequency model to predict appliance based on HF field
def HFModel(freqData, target):
    # model = LogisticRegression()
    model = KNeighborsClassifier(5)
    model.fit(freqData, target)
    # prediction = model.predict(target)
    # probabilities = model.predict_proba(freqData)
    return model