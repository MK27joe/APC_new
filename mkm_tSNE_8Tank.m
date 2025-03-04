%% DISCLAIMER: This code-file is in developemenet stage.
%%% This code contains t-SNE developed from scratch

%%
close all; clc; clear; 


%% Load the datasheet
load octuple_tank_data_28_08_24.mat


%% 0. Extract your database
%%=======================%%

%%%% T == res
%%% Outputs --> Col 2 to 9
%%% Inputs --> Col 10 to 13
%%% Disturbances --> Col 14 to 15

xt = res(:,2:end);  
[M,N] = size(xt);

Dtrain = xt(1:M/2,:);
[m1,n1] = size(Dtrain);

Dtest = xt(1+(M/2):M,:);
[m2,n2] = size(Dtest);

%% 1A. NORMALIZATION OF TRAINING DATASET
%%==================================%%

xm = mean(Dtrain);
Sdm = std(Dtrain);

Xbar = (Dtrain - xm(ones(m1,1),:)) ./ (Sdm(ones(m1,1),:));

CV =cov(Xbar);
[U,S,V] = svd(CV); 


%% 1B. PCA PARAMETERS USING TRAINING DATASET
%%======================================%%

%%% In the file coeff = COEFF and X = SCORE
[coeff, X, LATENT, TSQUARED, EXPLAINED, MU] = pca(Xbar); %% LATENT (i.e. EigValues)


%% 2. t-SNE Coding
%%=================%%

perplexity = 200;
p_ij = get_original_pairwise_affinities(X, perplexity);

% p_ij_symmetric = get_symmetric_p_ij(p_ij);
% 
% y0 = initialization(X, 2, 'random');
% 
% q_ij = get_low_dimensional_affinities(y0);
% 
% gradient = get_gradient(p_ij, q_ij, y0);

n = size(X, 1);

% Get original affinities matrix
p_ij = get_original_pairwise_affinities(X, perplexity);
p_ij_symmetric = get_symmetric_p_ij(p_ij);

% Initialization

T=1000; 
n_dimensions=2; 
early_exaggeration=4;
eta=200;

Y = zeros(T, n, n_dimensions);
Y_minus1 = zeros(n, n_dimensions);
Y(1,:,:) = Y_minus1;
Y1 = initialization(X, n_dimensions);
Y(2,:,:) = Y1;
fprintf('Optimizing Low Dimensional Embedding....\n');

%%

for t = 2:T-1

    % Momentum & Early Exaggeration
    if t < 250
        alpha = 0.5;
        ee = early_exaggeration;
    else
        alpha = 0.8;
        ee = 1;
    end

% Get Low Dimensional Affinities
q_ij = get_low_dimensional_affinities(squeeze(Y(t,:,:)));

% Get Gradient of Cost Function
gradient = get_gradient(ee * p_ij_symmetric, q_ij, squeeze(Y(t,:,:)));

% % % Update Rule [WARNING: DIMENSIONAL ERROR found b/w Y (3D) and gradient (2D)]

Y(t+1,:,:) = Y(t,:,:) - eta * gradient + alpha * (Y(t,:,:) - Y(t-1,:,:)) ; % Use negative gradient

% Compute current value of cost function
if mod(t, 50) == 0 || t == 2
    cost = sum(p_ij_symmetric .* log(p_ij_symmetric ./ q_ij), 'all');
    fprintf('Iteration %d: Value of Cost Function is %f\n', t, cost);
end

end


% [solution, Y] = mtsne(X, 10, 1000, 200, 4, 2);


