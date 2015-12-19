'''
Created on Dec 5, 2015

@author: Joe
'''
import LoadData as ld
import numpy as np
import pandas as pd
import HFModel as hf
from sklearn.cross_validation import train_test_split
from sklearn import metrics

if __name__ == '__main__':
    # H3_Test = ld.loadData('data/H3/Testing_01_21_1358755201.mat')
    H3 = ld.loadData('data/H3/Tagged_Training_07_30_1343631601.mat')
    hf.dataPrep(H3.HF, np.array(H3.tagInfo))

    X = H3.HF.drop(['Timestamp','Back Porch Lights'], axis=1)
    Y = H3.HF['Back Porch Lights']

    # Set randomness so that we all get the same answer
    np.random.seed(841)

    # Split the data into train and test pieces for both X and Y
    X_train, X_test, Y_train, Y_test = train_test_split(X.head(2000), Y.head(2000), train_size=0.80)
    model = hf.HFModel(X_train, Y_train)
    print("Accuracy on test = %.3f" % metrics.accuracy_score(model.predict(X_test), Y_test))


    # print(H3.L1.head(5))
    #print(ld.getApplianceData(H3.HF, H3.tagInfo).head(1))