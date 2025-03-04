###------ 0a. PREAMBLE ---------
### ===========================
###
from scipy.io import loadmat
import scipy.stats as Statz
import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale
import plotly.express as px
from scipy.stats import norm



import os
import glob
import csv
from xlsxwriter.workbook import Workbook


t1 = time.perf_counter_ns()
##print(f"Start-time, T1 (in ns) : {t1}")



###------ 0b. Definition of User-defined Functions ---------
### ========================================================
###

### Fn(1)

def get_original_pairwise_affinities(X: np.ndarray, perplexity: int = 10) -> np.ndarray:
##    """
##    Function to obtain affinities matrix.
##
##    Parameters:
##    X (np.ndarray): The input data array.
##    perplexity (int): The perplexity value for the grid search.
##
##    Returns:
##    np.ndarray: The pairwise affinities matrix.
##    """

    n = len(X)

    print("Computing Pairwise Affinities....")

    p_ij = np.zeros(shape=(n, n))
    for i in range(0, n):
        # Equation 1 numerator
        diff = X[i] - X
        sigma_i = grid_search(diff, i, perplexity)  # Grid Search for σ_i
        norm = np.linalg.norm(diff, axis=1)
        p_ij[i, :] = np.exp(-(norm**2) / (2 * sigma_i**2))

        # Set p = 0 when j = i
        np.fill_diagonal(p_ij, 0)

        # Equation 1
        p_ij[i, :] = p_ij[i, :] / np.sum(p_ij[i, :])

    # Set 0 values to minimum numpy value (ε approx. = 0)
    eps = np.nextafter(0, 1)
    p_ij = np.maximum(p_ij, eps)

    print("Completed Pairwise Affinities Matrix. \n")

    return p_ij

#### CHECKED !!

### Fn(1) <---  Fn(2)

def grid_search(diff_i: np.ndarray, i: int, perplexity: int) -> float:
##    """
##    Helper function to obtain σ's based on user-specified perplexity.
##
##    Parameters:
##        diff_i (np.ndarray): Array containing the pairwise differences between data points.
##        i (int): Index of the current data point.
##        perplexity (int): User-specified perplexity value.
##
##    Returns:
##        float: The value of σ that satisfies the perplexity condition.
##    """

    result = np.inf  # Set first result to be infinity

    norm = np.linalg.norm(diff_i, axis=1)
    
    std_norm = np.std(norm)  # Use standard deviation of norms to define search space

    for sigmas_search in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
        # Equation 1 Numerator
        p = np.exp(-(norm**2) / (2 * sigmas_search**2))

        # Set p = 0 when i = j
        p[i] = 0

        # Equation 1 (ε -> 0)
        eps = np.nextafter(0, 1)
        p_new = np.maximum(p / np.sum(p), eps) ### error: RuntimeWarning: invalid value encountered in divide

        # Shannon Entropy
        H = -np.sum(p_new * np.log2(p_new))

        # Get log(perplexity equation) as close to equality
        if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
            result = np.log(perplexity) - H * np.log(2)
            sigmas = sigmas_search

    return sigmas


### Moving from Fn(1) to Fn(3)

def get_symmetric_p_ij(p_ij: np.ndarray) -> np.ndarray:
##    """
##    Function to obtain symmetric affinities matrix utilized in t-SNE.
##
##    Parameters:
##    p_ij (np.ndarray): The input affinity matrix.
##
##    Returns:
##    np.ndarray: The symmetric affinities matrix.
##
##    """
    print("Computing Symmetric p_ij matrix....")

    n = len(p_ij)
    p_ij_symmetric = np.zeros(shape=(n, n))
    
    for i in range(0, n):
        for j in range(0, n):
            p_ij_symmetric[i, j] = (p_ij[i, j] + p_ij[j, i]) / (2 * n)

    # Set 0 values to minimum numpy value (eps, i.e. ε approx. = 0)
    eps = np.nextafter(0, 1)
    p_ij_symmetric = np.maximum(p_ij_symmetric, eps)

    print("Completed Symmetric p_ij Matrix. \n")

    return p_ij_symmetric

#### CHECKED !!


### (4)

def initialization(
    X: np.ndarray, n_dimensions: int = 2, initialization: str = "random"
) -> np.ndarray:
##    """
##    Obtain initial solution for t-SNE either randomly or using PCA.
##
##    Parameters:
##        X (np.ndarray): The input data array.
##        n_dimensions (int): The number of dimensions for the output solution. Default is 2.
##        initialization (str): The initialization method. Can be 'random' or 'PCA'. Default is 'random'.
##
##    Returns:
##        np.ndarray: The initial solution for t-SNE.
##
##    Raises:
##        ValueError: If the initialization method is neither 'random' nor 'PCA'.
##    """

    # Sample Initial Solution
    if initialization == "random" or initialization != "PCA":
        y0 = np.random.normal(loc = 0, scale = 1e-4, size = (len(X), n_dimensions))
        
    elif initialization == "PCA":
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered)
        y0 = X_centered @ Vt.T[:, :n_dimensions]
        
    else:
        raise ValueError("Initialization must be 'random' or 'PCA'")

    return y0

#### CHECKED !!


### (5)

def get_low_dimensional_affinities(Y: np.ndarray) -> np.ndarray:
##    """
##    Obtain low-dimensional affinities.
##
##    Parameters:
##    Y (np.ndarray): The low-dimensional representation of the data points.
##
##    Returns:
##    np.ndarray: The low-dimensional affinities matrix.
##    """

    n = len(Y)
    q_ij = np.zeros(shape=(n, n))

    for i in range(0, n):
        # Equation 4 Numerator
        diff = Y[i] - Y
        norm = np.linalg.norm(diff, axis=1)
        q_ij[i, :] = (1 + norm**2) ** (-1)

    # Set p = 0 when j = i
    np.fill_diagonal(q_ij, 0)

    # Equation 4
    q_ij = q_ij / q_ij.sum()

    # Set 0 values to minimum numpy value (ε approx. = 0)
    eps = np.nextafter(0, 1)
    q_ij = np.maximum(q_ij, eps)

    return q_ij

#### CHECKED !!



### (6)

def get_gradient(p_ij: np.ndarray, q_ij: np.ndarray, Y: np.ndarray) -> np.ndarray:
##    """
##    Obtain gradient of cost function at current point Y.
##
##    Parameters:
##    p_ij (np.ndarray): The joint probability distribution matrix.
##    q_ij (np.ndarray): The Student's t-distribution matrix.
##    Y (np.ndarray): The current point in the low-dimensional space.
##
##    Returns:
##    np.ndarray: The gradient of the cost function at the current point Y.
##    """

    n = len(p_ij)

    # Compute gradient
    gradient = np.zeros(shape=(n, Y.shape[1]))
    for i in range(0, n):
        # Equation 5
        diff = Y[i] - Y
        A = np.array([(p_ij[i, :] - q_ij[i, :])])
        B = np.array([(1 + np.linalg.norm(diff, axis=1)) ** (-1)])
        C = diff
        gradient[i] = 4 * np.sum((A * B).T * C, axis=0)

    return gradient





### (7)

def tsne(
    X: np.ndarray,
    perplexity: int = 10,
    T: int = 1000,
    etaz: int = 200,
    early_exaggeration: int = 4,
    n_dimensions: int = 2,
) -> list[np.ndarray, np.ndarray]:
##    """
##    t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm implementation.
##
##    Args:
##        X (np.ndarray): The input data matrix of shape (n_samples, n_features).
##        perplexity (int, optional): The perplexity parameter. Default is 10.
##        T (int, optional): The number of iterations for optimization. Default is 1000.
##        η (int, optional): The learning rate for updating the low-dimensional embeddings. Default is 200.
##        early_exaggeration (int, optional): The factor by which the pairwise affinities are exaggerated
##            during the early iterations of optimization. Default is 4.
##        n_dimensions (int, optional): The number of dimensions of the low-dimensional embeddings. Default is 2.
##
##    Returns:
##        list[np.ndarray, np.ndarray]: A list containing the final low-dimensional embeddings and the history
##            of embeddings at each iteration.
##
##    """

    n = len(X)

    # Get original affinities matrix
    p_ij = get_original_pairwise_affinities(X, perplexity)
    p_ij_symmetric = get_symmetric_p_ij(p_ij)

    # Initialization
    Y = np.zeros(shape=(T, n, n_dimensions))
    Y_minus1 = np.zeros(shape=(n, n_dimensions))
    Y[0] = Y_minus1
    Y1 = initialization(X, n_dimensions)
    Y[1] = np.array(Y1)

    print("Optimizing Low Dimensional Embedding....")
    # Optimization
    for t in range(1, T - 1):
        # Momentum & Early Exaggeration
        if t < 250:
            alphaz = 0.5
            early_exaggeration = early_exaggeration
        else:
            alphaz = 0.8
            early_exaggeration = 1

        # Get Low Dimensional Affinities
        q_ij = get_low_dimensional_affinities(Y[t])

        # Get Gradient of Cost Function
        gradient = get_gradient(early_exaggeration * p_ij_symmetric, q_ij, Y[t])

        # Update Rule
        Y[t + 1] = Y[t] - etaz * gradient + alphaz * (Y[t] - Y[t - 1])  # Use negative gradient

        # Compute current value of cost function
        cost = np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))
        if t % 50 == 0 or t == 1:
##            cost = np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))
            print(f"Iteration {t}: Value of Cost Function is {cost}")

    print(
        f"Completed Low Dimensional Embedding: Final Value of Cost Function is {np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))}"
    )
    solution = Y[-1]

    return solution, Y








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
dTrain.to_csv("DTrain_sheetTSNE.csv")
print('\nNo. of observations in training-data =', np.size(DTrain, 0))
print('No. of data-vectors in training-data =', np.size(DTrain, 1))

DTest = Cdata[n1:N_data,:]
dTest = pd.DataFrame(DTest)
dTest.to_csv("DTest_sheetTSNE.csv")
print('\nNo. of observations in testing-data =', np.size(DTest, 0))
print('No. of data-vectors in testing-data =', np.size(DTest, 1))





###------ 3. NORMALIZATION OF TRAINING DATASET  -------
### ===================================================


### --- 3a. Normalize the Training Dataset
###`````````````````````````````````````````
 
Xbar = scale(DTrain)
xm = np.mean(DTrain, axis=0)
Sdm = np.std(DTrain, axis=0)

xbar_trans = np.transpose(Xbar) ## ..else, we get an n1 x n1 CoVar Matrix !! 
CV = np.cov(xbar_trans) ## Obtain the m1 x m1 CoVar matrix



### --- 3b. SVD ---
###`````````````````
### SVD method
###
U, S, V_T = np.linalg.svd(CV, full_matrices=True, compute_uv=True, hermitian=False)
SS = np.diag(S)

UU = pd.DataFrame(U)
UU.to_csv("U_MatrixTSNE.csv")  ### not necessary!! --> found same as U matrix in MATLAB

SS = pd.DataFrame(S)
SS.to_csv("S_MatrixTSNE.csv")  ### not necessary!! --> found same as S matrix in MATLAB

VT = pd.DataFrame(V_T)
VT.to_csv("VT_MatrixTSNE.csv")  ### not necessary!! --> found same as V^T matrix in MATLAB




###------ 4. INTRODUCE BIAS FAULT INTO TEST DATASET; THEN, NORMALIZE WRT MODEL -------
### ==================================================================================


lim = 1000;
IndX = 4; #### Tank# 4

FaultID = 'Bias'
Bias_value = 10
for i in range (0, n2):
        if i>lim:
                DTest[i,IndX-1] = DTest[i,IndX-1] + Bias_value

dTestF = pd.DataFrame(DTest)
dTestF.to_csv("DTestF_sheetTSNE.csv")



SY = (DTest - np.array([xm,]*n2))/ (np.array([Sdm,]*n1)) ### Normalizing DTest using the mean and STDEV of DTrain
SY_trans = np.transpose(SY) ## ..else, we get an n1 x n1 CoVar Matrix !! 
CV_test = np.cov(SY_trans)




###------ 5. t-SNE MODELS FOR NORMALIZED TRAINING DATASET  -------
### ===============================================================
###

print("\nt-SNE MODELING FOR NORM.TRAINING DATASET BEGINS::\n\n")

p_ij = get_original_pairwise_affinities(Xbar)
PIJ= pd.DataFrame(p_ij)
PIJ.to_csv("HDpairwise_affinities_Train.csv")

p_ij_symmetric = get_symmetric_p_ij(p_ij) 
PIJ_symm = pd.DataFrame(p_ij_symmetric)
PIJ_symm.to_csv("HDSYMpairwise_affinities_Train.csv")

y0 = initialization(Xbar)
Y_initial = pd.DataFrame(y0)
Y_initial.to_csv("Y_initial_Train.csv")

q_ij = get_low_dimensional_affinities(y0)
QIJ= pd.DataFrame(q_ij)
QIJ.to_csv("LDaffinities_Train.csv")

gradientZ = get_gradient(p_ij_symmetric, q_ij, y0)
Gradz = pd.DataFrame(gradientZ)
Gradz.to_csv("GradientMatrix_Train.csv")

solution1, Y1 = tsne(Xbar)
###### solution1 is the final 2-D mapping and
###### Y1 is the mapped 2-D values at 'each step' of the iteration == a 3D file !!.
###### Therefore, final_Y = Y1(-1)

Soln1z = pd.DataFrame(solution1)
Soln1z.to_csv("Train_LD_Sol.csv")

print("\n\nt-SNE MODELING FOR NORM.TRAINING DATASET ENDS::\n\n")

t2 = time.perf_counter_ns()


##print(f"\nStop-time, T2 (in s) : {tttimeSec}\n")
#### CHECKED !!



###------ 6. t-SNE MODELS FOR NORMALIZED TESTING DATASET  -------
### ===============================================================
###

print("\nt-SNE MODELING FOR NORM.TESTING DATASET BEGINS::\n\n")

p_ijTS = get_original_pairwise_affinities(SY)
PIJ = pd.DataFrame(p_ijTS)
PIJ.to_csv("HDpairwise_affinities_Test.csv")

p_ij_symmetric_TS = get_symmetric_p_ij(p_ijTS) 
PIJ_symm = pd.DataFrame(p_ij_symmetric_TS)
PIJ_symm.to_csv("HDSYMpairwise_affinities_Test.csv")

y0_TS = initialization(SY)
Y_initial = pd.DataFrame(y0_TS)
Y_initial.to_csv("Y_initial_Test.csv")

q_ij_TS = get_low_dimensional_affinities(y0_TS)
QIJ = pd.DataFrame(q_ij_TS)
QIJ.to_csv("LDaffinities_Test.csv")

gradientZ_TS = get_gradient(p_ij_symmetric_TS, q_ij_TS, y0_TS)
Gradz = pd.DataFrame(gradientZ_TS)
Gradz.to_csv("GradientMatrix_Test.csv")

solution2, Y2 = tsne(SY)
###### solution1 is the final 2-D mapping and
###### Y1 is the mapped 2-D values at 'each step' of the iteration == a 3D file !!.
###### Therefore, final_Y = Y1(-1)

Soln1z2 = pd.DataFrame(solution2)
Soln1z2.to_csv("Test_LD_Sol.csv")


print("\nt-SNE MODELING FOR NORM.TESTING DATASET ENDS::\n\n")

t3 = time.perf_counter_ns()


tttimeSec1 = (t2-t1)*1e-9
tttimeSec2 = (t3-t2)*1e-9
print(f"\nStop-time, T_train (in s) : {tttimeSec1}\n")
print(f"\nStop-time, T_test (in s) : {tttimeSec2}\n")


###------ 7. PLOT ALL RESULTS  -------
### ===================================
###

### Plot the Heatmap for CoVar Matrix of Norm.Train.Dataset
plt.figure()
sns.heatmap(CV)
plt.title("Co-Variance Matrix (Heat-map) for Norm.TrainData")
plt.xlabel("Measurement Vector-Number")
plt.ylabel("Measurement Vector-Number")
plt.show()


### Plot the Un-normalized Faulty Test-Dataset Measurement Vector

plt.figure()          
plt.plot(DTest[:,IndX-1])
plt.xlim(0,n2)
plt.title("Un-normalized Faulty (i.e. Bias = 10) Test-Dataset vector, i.e. for Tank-4")
plt.xlabel("Observation")
plt.grid()
plt.show()


### Plot the t-SNE Reduced Training-Dataset Measurement Vectors
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(Soln1z)
plt.xlim(0,len(Soln1z ))
plt.title("t-SNE reduced DTrain Lower Dimensional Space, Y = [Y1 Y2]")
plt.xlabel("Observation")
plt.grid()

### Plot the t-SNE Reduced Testing-Dataset Measurement Vectors

plt.subplot(2, 1, 2)
plt.plot(Soln1z2)
plt.xlim(0,len(Soln1z2 ))
plt.title("t-SNE reduced Faulty DTest Lower Dimensional Space, Y = [Y1 Y2]")
plt.xlabel("Observation")
plt.grid()
plt.show()








