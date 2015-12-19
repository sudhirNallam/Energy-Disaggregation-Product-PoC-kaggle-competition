'''
Created on Dec 5, 2015

@author: Joe
'''
import LoadData as ld
import HFModel as hf

if __name__ == '__main__':
    H3 = ld.loadData('data/H3/Tagged_Training_07_30_1343631601.mat')
    print(H3.HF.head(5))
    print(H3.L1.head())
    print(ld.getApplianceData(H3.HF, H3.tagInfo).head(5))