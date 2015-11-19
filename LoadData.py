__author__ = 'sudhir'
import scipy.io as sio
import numpy as np
import cmath as cm


data = sio.loadmat('H3/Tagged_Training_07_30_1343631601.mat')

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
print ProcessedData["TaggingInfo"]