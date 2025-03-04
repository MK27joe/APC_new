%% DISCLAIMER: This code-file is NOT developed for user-interactive mode

%%
close all; clc; clear; 


%% Load the datasheet
% load octuple_tank_data_28_08_24.mat
load octuple_tank_data_11_11_24.mat



%% 0. Extract your database
%%=======================%%

%%%% T == res
%%% Outputs --> Col 2 to 9
%%% Inputs --> Col 10 to 13
%%% Disturbances --> Col 14 to 15

xt = res(:,2:end);  
[M,N] = size(xt);



%% 1. Generate Residue from training data v/s PCA-model
%%===================================================%% 

%---- 1a. Extract dataset for training
Dtrain = xt(1:M/2,:);
[m1,n1] = size(Dtrain);


%---- 1b. Normalization 
% xm = mean(Dtrain);
% Sdm = std(Dtrain);
% Xbar = (Dtrain - xm(ones(m1,1),:)) ./ (Sdm(ones(m1,1),:));

[Xbar] = NormDat(Dtrain);
CV=cov(Xbar);
[U,S,V] = svd(CV); 

NDtr1 = Xbar(1:size(Xbar,1)/2,:);
NDtr2 = Xbar(1+size(Xbar,1)/2:end,:);


%---- 1c. PCA parameters using Dtr1
[Coeff1a, Score1a, Latent1a, TSQ1a, Explain1a, MUx] = pca(NDtr1); 
%%% 'Latent' = Eigen Values
%%% 'Score' contains PC vectors (column-vectored)
%%% 'Coeff' = Loading Matrix


%---- 1d. Find the desired no. of PCs (k) using CPV for 95% yield
prompt = 95;
percent = prompt/100;
k1=0; %%% Initial count = 0
for i = 1:size(Latent1a,1)                                                        
    alpha(i)=sum(Latent1a(1:i))/sum(Latent1a);
    if alpha(i)>=percent
            k1=i;
            break;
    end 
end
alpha,k1

%---- 1e. Find the Residue Space e1a using NDtr1
Xp1a = NDtr1*Coeff1a(:,1:k1)*Coeff1a(:,1:k1)';   % Xp is the estimation of original data using the PCA model, X=Xp+E
e1a = NDtr1 - Xp1a; %% Residual Space @ No-fault scenario

MU_e1 = mean(e1a);
Cov_e1a = cov(e1a);

%%%----- Next.. -----%%%



    





