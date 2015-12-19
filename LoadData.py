from unittest.mock import inplace
from numpy import dtype
__author__ = 'sudhir'

import scipy.io as sio
import numpy as np
import cmath as cm
import pandas as pd
from datetime import datetime
import collections
import multiprocessing as mp
import sys

# tuple of dataframes that we will return results in
houseData = collections.namedtuple('houseData', ['L1', 'L2', 'HF', 'tagInfo'])

def loadData(filename):
    data = sio.loadmat(filename)
    
    L1_P = data["Buffer"]["LF1V"][0][0] * np.conjugate(data["Buffer"]["LF1I"][0][0])
    L2_P = data["Buffer"]["LF2V"][0][0] * np.conjugate(data["Buffer"]["LF2I"][0][0])
    
    # Compute net Complex power
    L1_ComplexPower = L1_P.sum(axis=1);
    L2_ComplexPower = L2_P.sum(axis=1);
    
    ProcessedData = {};
    
    # Real, Reactive, Apparent powers
    # Phase-1
    ProcessedData["L1_Real"] = L1_ComplexPower.real;
    ProcessedData["L1_Imag"] = L1_ComplexPower.imag;
    ProcessedData["L1_App"] = abs(L1_ComplexPower);
    
    # Real, Reactive, Apparent powers
    # Phase-II
    ProcessedData["L2_Real"] = L2_ComplexPower.real;
    ProcessedData["L2_Imag"] = L2_ComplexPower.imag;
    ProcessedData["L2_App"] = abs(L2_ComplexPower);
    
    # Compute Power Factor, we only consider the first 60Hz component
    ProcessedData["L1_Pf"] = np.cos(np.angle(L1_P[:,0], deg=False));
    ProcessedData["L2_Pf"] = np.cos(np.angle(L2_P[:,0], deg=False));
    
    
    # Copy Time ticks to our processed structure
    ProcessedData["L1_TimeTicks"] = data["Buffer"]["TimeTicks1"][0][0];
    ProcessedData["L2_TimeTicks"] = data["Buffer"]["TimeTicks2"][0][0];
    
    # Move over HF Noise and Device label (tagging) data to our final structure as well
    ProcessedData["HF"] = data["Buffer"]["HF"][0][0];
    ProcessedData["HF_TimeTicks"] = data["Buffer"]["TimeTicksHF"][0][0];
    
    # Copy Labels/TaggingInfo id they exist
    ProcessedData["TaggingInfo"] = data["Buffer"]["TaggingInfo"][0][0];
    
    # multiprocess for epic fun
    q1 = mp.Queue() 
    q2 = mp.Queue() 
    qf = mp.Queue() 
    qa = mp.Queue()
    
    p1 = mp.Process(target=prepLineData, args=(q1, ProcessedData["L1_Real"], ProcessedData["L1_Imag"], ProcessedData["L1_App"], ProcessedData["L1_Pf"], ProcessedData["L1_TimeTicks"]))
    #print("P1 starting")
    p1.start()
    #print("P1 started")
    
    p2 = mp.Process(target=prepLineData, args=(q2, ProcessedData["L2_Real"], ProcessedData["L2_Imag"], ProcessedData["L2_App"], ProcessedData["L2_Pf"], ProcessedData["L2_TimeTicks"]))
    #print("P2 starting")
    p2.start()
    #print("P2 started")
    
    pf = mp.Process(target=prepHFData, args=(qf, ProcessedData["HF"], ProcessedData["HF_TimeTicks"]))
    #print("Pf starting")
    pf.start()
    #print("Pf started")
    
    return houseData(q1.get(), q2.get(), qf.get(), ProcessedData['TaggingInfo'])

# let's turn this info into a pandaframe
def prepHFData(q, pdhf, pdhft):
    # with all relevant HF info inside ProcessedData, now we need to slice it into timesteps
    hfTimesteps = pd.DataFrame(pdhf, columns=None)
    hfTimesteps = hfTimesteps.transpose()
    hfTimesteps.set_index(pdhft.astype(int), inplace=True)
    
    # create midpoint list of frequency ranges
    l = list(range(122, 1000000, 244))
    # some trimming due to rounding concerns
    l = l[1:-1]
    cols = np.asarray(l)
    hfTimesteps.columns = cols
    
    print("HF Data prepped")
    
    q.put(hfTimesteps)
    return

def prepLineData(q, real, imag, app, pf, timeticks):
    ldFrame = pd.DataFrame([real, imag, app, pf], columns=None)
    ldFrame = ldFrame.transpose()
    ldFrame.set_index(timeticks.astype(int), inplace=True)
    cols = ["Real Power", "Imaginary Power", "Apparent Power", "Power Factor"]
    ldFrame.columns = cols
    
    print("Line Data prepped")
    
    q.put(ldFrame)
    return

# return a large, rather disperse dataframe of all appliance on times
# takes in a pandaframe of line or HF data plus relevant ticks and returns tagged appliance on/off (i.e. target) info for that range for all appliances 
def getApplianceData(data, taggingInfo):
    tagTable = pd.DataFrame(index=data.index)
    colSize = len(data.index)
    #we need to interate through all tagged devices
    for URdeviceID, URname, URon, URoff in taggingInfo:
        names = tagTable.columns.values.tolist()
        
        if name not in names:
            # get values out of this mess
            deviceID = int(URdeviceID[0][0])
            name = str(URname[0][0])[2:-2]
            on = URon[0][0]
            off = URoff[0][0]
            
            newCol = np.zeros(colSize, dtype=bool).transpose()
            uniqueID = ' '.join([str(name), str(deviceID)])
            newFrame = pd.DataFrame(newCol, columns=[uniqueID], index=data.index)
            newFrame[on:off] = ~newFrame[on:off]
            tagTable = pd.concat([tagTable, newFrame], axis=1)
            
        else:
            tagTable = tagTable.merge([newFrame], how='right', on=name)


    return tagTable        