###------ 0. PREAMBLE ---------
### ===========================

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
n

### [NOTE: Use ALT+3 to comment // ALT+4 to uncomment]


###------ 1. LOAD .MAT FILE AND CONVERT IT TO .CSV FILE -------
### ===========================================================


## --- 1a. Load .mat file from desired directory
## ```````````````````````````````````````````````

# ^^^ octuple_tank_data_11_11_24.mat contains 4000 x 15 data; the first column being the index number (by MATLAB)

data = loadmat(r"D:\MatlabCodes\Octuple_Tank\octuple_tank_data_11_11_24.mat")
#data = loadmat(r"C:\Users\MAHE\AppData\Local\Programs\Python\Python313\octuple_tank_data_11_11_24.mat")
## mkm_laptop --> "D:\[] MKM_PhD_2023\MatlabCodes\Octuple_Tank\octuple_tank_data_11_11_24.mat"



## --- 1b. Convert .mat file to .csv and save in the current directory
## ````````````````````````````````````````````````````````````````````
for i in data:
        if '__' not in i and 'readme' not in i:
              np.savetxt(("Res.csv"),data[i],delimiter=',')

df = pd.read_csv('Res.csv')

##reader = csv.reader(open("Res.csv", "rb"), delimiter=",")
##x = list(reader)
##result = numpy.array(x).astype("float")

with open('Res.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader) 


## --- 1c. Read .csv file as Numpy Array
## ``````````````````````````````````````
    
data_array = np.array(data, dtype=float)
Nr_data = len(data_array)    ## = 4000
Mr_data = len(data_array[0]) ## = 15 (... the first col. contains index no. !! It must be removed)


Cdata = np.delete(data_array, 0, 1)## delete the first column of 'data_array'
## Caution: In python, indexing (row or column) starts from i,j = 0 !!



N_data = len(Cdata ) ### no. of observations in raw-data file 'data_array' [ROWS = 4000]
M_data = len(Cdata [0]) ### no. of data-vectors in raw-data file 'data_array' [COLUMNS]

print('No. of observations in raw-data files =', N_data)
print('\nNo. of data-vectors in raw-data files =', M_data)

# distributing the dataset into two components X and Y
Y = Cdata.iloc[:, 0:M_data].values
X = Cdata.iloc[:, M_data:].values


## --- 1d. Write Numpy Array into .csv file (Optional)
## ````````````````````````````````````````````````````
DF = pd.DataFrame(Cdata)
DF.to_csv("data_OG.csv")  ### not necessary




###------ 2. SPLIT DATA-FILE INTO TRAINING & TESTING DATA -------
### =============================================================

# # ---for training
n1= int(N_data/2)
m1= int(M_data/2)

# # ---for testing
n2=n1
m2=m1

####

DTrain = Cdata[0:n1,:]
print(np.size(DTrain, 0))
print(np.size(DTrain, 1))
Dtr = pd.DataFrame(DTrain)
Dtr.to_csv("TrainData_OG.csv")  ### not necessary

DTest = Cdata[n1:N_data,:]
print(np.size(DTest, 0))
print(np.size(DTest, 1))
Dts = pd.DataFrame(DTest)
Dts.to_csv("TestData_OG.csv")  ### not necessary


###------ 3. NORMALIZATION OF TRAINING DATASET -------
### =============================================================




xm = np.mean(DTrain, axis=0)
Sdm = np.std(DTrain, axis=0)

Xbar = (DTrain - np.array([xm,]*n1))/ (np.array([Sdm,]*n1)) 





