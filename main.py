'''
Created on Dec 5, 2015

@author: Joe
'''
import LoadData as ld
import HFModel as hf

if __name__ == '__main__':
    H3 = ld.loadData('data/H3/Tagged_Training_07_30_1343631601.mat')
    #print(H3.HF.head(5))
    #print(H3.L1.head())
    #ld.getApplianceData(H3.HF, H3.tagInfo).to_csv('data/H3/Results/appData.csv')
    
    X = H3.HF[0:50]
    Y = ld.getApplianceData(X, H3.tagInfo).head(50)
    #Y = Y.drop('Bedroom Lights 1', axis=1)
    
    hf.model(X,Y)