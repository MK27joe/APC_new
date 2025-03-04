###------ 0. PREAMBLE ---------
### ===========================
###
from scipy.io import loadmat
import scipy.stats as Statz
import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale
import plotly.express as px
from scipy.stats import norm


import os
import glob
import csv
from xlsxwriter.workbook import Workbook






###------ 1. LOAD DATASET  ---------
### =================================
###
###
### --- 1a. Load .mat file from desired directory
### ```````````````````````````````````````````````

data = loadmat(r"octuple_tank_data_11_11_24.mat")
###data = loadmat(r"D:\[] MKM_PhD_2023\py_codes\octuple_tank_data_11_11_24.mat")


### --- 1b. Convert .mat file to .csv and save in the current directory
### ````````````````````````````````````````````````````````````````````

for i in data:
        if '__' not in i and 'readme' not in i:
              np.savetxt(("Res.csv"),data[i],delimiter=',')

df = pd.read_csv('Res.csv')

with open('Res.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader) 


### --- 1c. Read .csv file as Numpy Array
###``````````````````````````````````````
    
data_array = np.array(data, dtype=float)
Nr_data = len(data_array)    ## = 4000
Mr_data = len(data_array[0]) ## = 15 (... the first col. contains index no. !! It must be removed)


Cdata = np.delete(data_array, 0, 1)## delete the first column of 'data_array'
### Caution: In python, indexing (row or column) starts from i,j = 0 !!

N_data = len(Cdata ) ### no. of observations in raw-data file 'data_array' [ROWS = 4000]
M_data = len(Cdata [0]) ### no. of data-vectors in raw-data file 'data_array' [COLUMNS]

print('\n\nNo. of observations in raw-data files =', N_data)
print('No. of data-vectors in raw-data files =', M_data)





###------ 2. SPLIT DATA-FILE INTO TRAINING & TESTING DATA -------
### =============================================================

### --- for training
n1 = int(N_data/2) ## no. of training observtn.
m1 = int(M_data) ## no. of measurement vectors.


### --- for testing
n2 = n1 ## no. of testing observtn.
m2 = m1 ## no. of measurement vectors.

###

DTrain = Cdata[0:n1,:] ## Extract data for training from OG dataset
dTrain = pd.DataFrame(DTrain)
dTrain.to_csv("DTrain_sheet.csv")
print('\nNo. of observations in training-data =', np.size(DTrain, 0))
print('No. of data-vectors in training-data =', np.size(DTrain, 1))

DTest = Cdata[n1:N_data,:]
dTest = pd.DataFrame(DTest)
dTest.to_csv("DTest_sheet.csv")
print('\nNo. of observations in testing-data =', np.size(DTest, 0))
print('No. of data-vectors in testing-data =', np.size(DTest, 1))





###------ 3. NORMALIZATION OF TRAINING DATASET && DEVELOPE PCA MODEL -------
### ========================================================================


### --- 3a. Normalize the Training Dataset
###`````````````````````````````````````````
 
Xbar = scale(DTrain)
xm = np.mean(DTrain, axis=0)
Sdm = np.std(DTrain, axis=0)
xbar_trans = np.transpose(Xbar) ## ..else, we get an n1 x n1 CoVar Matrix !! 
CV = np.cov(xbar_trans) ## Obtain the m1 x m1 CoVar matrix


plt.figure()
sns.heatmap(CV)
plt.title("Co-Variance Matrix (Heat-map) for Norm.TrainData")
plt.xlabel("Measurement Vector-Number")
plt.ylabel("Measurement Vector-Number")
plt.show()


### SVD method
###
U, S, V_T = np.linalg.svd(CV, full_matrices=True, compute_uv=True, hermitian=False)
SS = np.diag(S)

UU = pd.DataFrame(U)
UU.to_csv("U_Matrix.csv")  ### not necessary!! --> found same as U matrix in MATLAB

SS = pd.DataFrame(S)
SS.to_csv("S_Matrix.csv")  ### not necessary!! --> found same as S matrix in MATLAB

VT = pd.DataFrame(V_T)
VT.to_csv("VT_Matrix.csv")  ### not necessary!! --> found same as V^T matrix in MATLAB



### Extract ALL PCA Model informations
###
pca = decomposition.PCA(n_components = m1) ### Extract ALL info from ALL PCs

SCORE_PC = pca.fit_transform(Xbar) ### To extract the SCORE matrix of PCs
ww = pd.DataFrame(SCORE_PC)
ww.to_csv("ScoreMatrix.csv")  ### not necessary!!

coeff_matrix = pca.components_ ### To extract the COEFF matrix for PCs
coeff_matrix_trans = np.transpose(coeff_matrix) #### so as have exact match of COEFF w/ the MATLAB result
qq = pd.DataFrame(coeff_matrix_trans)
qq.to_csv("Coeff_Matrix.csv")  ### not necessary!!

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings) ### To extract the LOADING matrix of PCs
ww1 = pd.DataFrame(loading_matrix)
ww1.to_csv("LoadMatrix.csv")  ### not necessary!!


### CPV test for finding required no. of PCs
###
prompt = 95

percent = prompt/100

k = 0 ### Initial count = 0
i = 0

size_S = len(S )
alpha = np.empty([size_S,1], dtype=float, order='C')
for i in range(1, size_S):
    alpha[i] = sum(S[:i])/sum(S)
    if alpha[i] >= percent:
        k = i
        break
###princ = SCORE_PC(:,1:k); ### extract the first kth PCs
    
princ = SCORE_PC[:,0:k] ### Extract Scores for the first 'k' PCs
princc = pd.DataFrame(princ)
princc.to_csv("Firstk_PCS.csv")

exppl_per = S/sum(S); ## or, use exppl_per = pca.explained_variance_ratio_
CumExp = 100*np.cumsum(exppl_per)
####nPC = [range(1, m1)]
nPC = np.arange(1, m1+1, 1)

plt.figure()
plt.subplot(2, 1, 1)
plt.title("Variance contribution by i-th PCs")
plt.xlabel("PC#")
plt.ylabel("Variance (or, Eigen Values)")

plt.stem(S)
plt.grid()


plt.subplot(2, 1, 2)
plt.title("Cumulative Explained Variance (in %) vs. Number of PCs")
plt.xlabel("Number of PCs")
plt.ylabel("CPV")

plt.bar(nPC,100*np.cumsum(exppl_per))
plt.grid()
plt.show()





###------ 2. INTRODUCE BIAS FAULT INTO TEST DATASET; NORMALIZE WRT MODEL -------
### ============================================================================


lim = 1000;
IndX = 4; #### Tank# 4

FaultID = 'Bias'
Bias_value = 10
for i in range (0, n2):
        if i>lim:
                DTest[i,IndX-1] = DTest[i,IndX-1] + Bias_value


### Plot the Un-normalized Test-Dataset

plt.figure()          
plt.plot(DTest[:,IndX-1])
plt.xlim(0,n2)
plt.title("Un-normalized Faulty (i.e. Bias = 10) Test-Dataset vector, i.e. for Tank-4")
plt.xlabel("Observation")
### plt.ylabel("CPV")
### plt.axis([0, 10, 0, 20])
### plt.ylim(-2, 2)
plt.grid()
plt.show()


SY = (DTest - np.array([xm,]*n2))/ (np.array([Sdm,]*n1)) ### Normalizing DTest using the mean and STDEV of DTrain
SY_trans = np.transpose(SY) ## ..else, we get an n1 x n1 CoVar Matrix !! 
CV_test = np.cov(SY_trans)
####CVTEST = pd.DataFrame(CV_test)
####CVTEST.to_csv("CVTEST.csv")

XT1 = np.matmul(SY, coeff_matrix_trans[:,0:k])
Xp = np.matmul(XT1, np.transpose(coeff_matrix_trans[:,0:k])) ### Xp = estimation of original data using the PCA model, SY = Xp + E
e = SY - Xp ### Model Error aka Residual Space, E





###------ 3. ANALYZE THE T^2 AND SPE-Q SPACES
### ==========================================

### --- 3a. Hotelling's T^2 Analysis
###```````````````````````````````````
###
###
### -- 3a(i). Threshold
###`````````````````````
### print (SStat.f.isf(0.05, k, n1-k)) ### Reject Ho until 95% signification level
T2knbeta = (k*(len(Xbar )**2-1)/(len(Xbar ) *(len(Xbar ) - k))) * Statz.f.isf(0.05, k, len(Xbar ) -k)
print(f'\n(1) T^2 Beta = {T2knbeta:.4f}')
TS = T2knbeta * np.ones((SY.shape[0], 1)) ### Create an array of threshold value, so as to plot the same

### -- 3a(ii). Statistics
###```````````````````````
### test-code
SY = np.array(SY)
XT1 = np.array(XT1)

### Initialize ts1
ts1 = np.zeros(SY.shape[0])

DiagS = np.diag(S[0:k])
InvDiagS = np.linalg.inv(DiagS)
##ts1 = XT1[:,0:k] * InvDiagS[0:k] * XT1[:,0:k]

### Loop to compute ts1
for i in range(SY.shape[0]):
    ts1[i] = XT1[i, :k] @ InvDiagS[:k] @ XT1[i, :k].T



### Plot the T^2 Statistics vs Threshold
###
plt.figure()

plt.subplot(2, 1, 1)

plt.plot(ts1, 'b', label='T^2 Statistics')
###plt.hold(True)  ### Note: plt.hold() is deprecated in newer versions of Matplotlib
plt.plot(TS, 'r', label='T^2 Threshold')
####plt.axis([0, 10, 0, 20])
####plt.ylim(-2, 2)
plt.xlim(0,n2)
plt.title("PCA-T^2 Analysis")
plt.xlabel("Observation")
plt.ylabel("Statistics")
plt.legend()
plt.grid()



### --- 3b. SPE-Q Analysis
###````````````````````````
###
###
### -- 3b(i). Threshold
###`````````````````````
# Parameters
beta = 0.95  # takes values between 90% and 95%
theta = np.zeros(3)

# Assuming SY and LATENT are already defined as NumPy arrays
SY = np.array(SY)
Sar = np.array(S)


# Compute theta
for ii in range(3):
    for j in range(k, SY.shape[1]):
        theta[ii] += Sar[j] ** (ii + 1)

h0 = 1 - ((2 * theta[0] * theta[2]) / (3 * theta[1] ** 2))
ca = norm.ppf(0.95, loc=0, scale=1)  # ca is value for standard normal distribution for confidence level 95%
SPEbeta = theta[0] * ((ca * h0 * np.sqrt(2 * theta[1]) / theta[0]) + 1 + (theta[1] * h0 * (h0 - 1) / theta[0] ** 2)) ** (1 / h0)
print(f'\n(2) SPE-Q Beta = {SPEbeta:.4f}')
S1 = SPEbeta * np.ones(SY.shape[0])



### -- 3b(ii). Statistics
###````````````````````````
SPE = np.zeros(e.shape[0])
for i in range(e.shape[0]):
    SPE[i] = np.sum((e[i, :])**2)



### Plot the SPE-Q Statistics vs Threshold
###
plt.subplot(2, 1, 2)

plt.plot(SPE, 'b', label='SPE-Q Statistics')
###plt.hold(True)  ### Note: plt.hold() is deprecated in newer versions of Matplotlib
plt.plot(S1, 'r', label='SPE-Q Threshold')
plt.xlim(0,n2)
plt.title("PCA-SPE-Q Analysis")
plt.xlabel("Observation")
plt.ylabel("Statistics")
plt.legend()
plt.grid()
plt.show()


###########===> All codes are verified w.r.t. Matlab file: D:\MatlabCodes\Octuple_Tank\PCA_t2_spe_kld.m
###########===> .... as on Feb 11, 2025 1551 hrs. IST
