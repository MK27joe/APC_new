###------ 0. PREAMBLE ---------
### ===========================

from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig
import csv
### [NOTE: Use ALT+3 to comment // ALT+4 to uncomment]


def desc_sort(arr):
    temp = 0;      
    
    #Sort the array in descending order    
    for i in range(0, len(arr)):    
        for j in range(i+1, len(arr)):    
            if(arr[i] < arr[j]):    
                temp = arr[i];    
                arr[i] = arr[j];    
                arr[j] = temp;  



















###------ 1. LOAD .MAT FILE AND CONVERT IT TO .CSV FILE -------
### ===========================================================


## --- 1a. Load .mat file from desired directory
## ```````````````````````````````````````````````

# ^^^ octuple_tank_data_11_11_24.mat contains 4000 x 15 data; the first column being the index number (by MATLAB)
##data = loadmat(r"C:\Users\MAHE\AppData\Local\Programs\Python\Python313\octuple_tank_data_11_11_24.mat")
## mkm_laptop --> "D:\[] MKM_PhD_2023\MatlabCodes\Octuple_Tank\octuple_tank_data_11_11_24.mat"
data = loadmat(r"D:\MatlabCodes\Octuple_Tank\octuple_tank_data_11_11_24.mat")



## --- 1b. Convert .mat file to .csv and save in the current directory
## ````````````````````````````````````````````````````````````````````
for i in data:
        if '__' not in i and 'readme' not in i:
              np.savetxt(("Res.csv"),data[i],delimiter=',')

df = pd.read_csv('Res.csv')

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


## --- 1d. Write Numpy Array into .csv file (Optional)
## ````````````````````````````````````````````````````
DF = pd.DataFrame(Cdata) ## Cdata is now the OG dataset
DF.to_csv("data_OG.csv")  ### not necessary!!


###------ 2. SPLIT DATA-FILE INTO TRAINING & TESTING DATA -------
### =============================================================

# # ---for training
n1 = int(N_data/2) ## no. of training observtn.
m1 = int(M_data/2) ## no. of measurement vectors.


# # ---for testing
n2 = n1 ## no. of testing observtn.
m2 = m1 ## no. of measurement vectors.

####

DTrain = Cdata[0:n1,:] ## Extract data for training from OG dataset 
print(np.size(DTrain, 0))
print(np.size(DTrain, 1))
Dtr = pd.DataFrame(DTrain)
Dtr.to_csv("TrainData_OG.csv")  ### not necessary!!

DTest = Cdata[n1:N_data,:]
print(np.size(DTest, 0))
print(np.size(DTest, 1))
Dts = pd.DataFrame(DTest)
Dts.to_csv("TestData_OG.csv")  ### not necessary!!


###------ 3. NORMALIZATION OF TRAINING DATASET -------
### =============================================================




xm = np.mean(DTrain, axis=0) ## Find the mean of each col. in DTrain
Sdm = np.std(DTrain, axis=0) ## Find the std dev. of each col. in DTrain

Xbar = (DTrain - np.array([xm,]*n1))/ (np.array([Sdm,]*n1)) ## Normalize the DTrain data
XbarTr = pd.DataFrame(Xbar)
XbarTr.to_csv("NormTrainData.csv")  ### not necessary!!


xbar_trans = np.transpose(Xbar) ## ..else, we get an n1 x n1 CoVar Matrix !! 
CV = np.cov(xbar_trans) ## Obtain the m1 x m1 CoVar matrix
CVTr = pd.DataFrame(CV)
CVTr.to_csv("CVTrainData.csv")  ### not necessary!!

W, V = eig(CV) ## W --> Eig Values (LATENTS) && V --> Eig. Vectors

WW = desc_sort(W)

##arr = np.array([W])
####arr.sort(reverse = True) ## Sort in Descending order
##
##
##prompt = 98 ## Percentage of significance for CPV test (80-95 %)
##percent = prompt/100


###(!)### ()---> data cross-checked w/ MATLAB {as on Jan 20, 2025}
































##plt.plot(x, y, color='green', label='Sine')
##plt.legend()
##plt.xlabel("Time (t)")
##plt.ylabel("Ampliltude")
##plt.grid()
##plt.show()







