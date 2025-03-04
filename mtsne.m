function [solution, Y] = mtsne(X, perplexity, T, eta, early_exaggeration, n_dimensions)
% t-SNE (t-Distributed Stochastic Neighbor Embedding) algorithm implementation.
%
% Inputs:
%>> X (matrix): The input data matrix of shape (n_samples, n_features).
%>> perplexity (integer, optional): The perplexity parameter. Default is 10.
%>> T (integer, optional): The number of iterations for optimization. Default is 1000.
%>> eta (integer, optional): The learning rate for updating the low-dimensional embeddings. Default is 200.
%>> early_exaggeration (integer, optional): The factor by which the pairwise affinities are exaggerated...during the early iterations of optimization. Default is 4.
%>> n_dimensions (integer, optional): The number of dimensions of the low-dimensional embeddings. Default is 2.
%
% Outputs:
%>> solution (matrix): The final low-dimensional embeddings.
%>> Y (3D matrix): The history of embeddings at each iteration.


% Set default values if not provided

if nargin < 2, perplexity = 10; end
if nargin < 3, T = 1000; end
if nargin < 4, eta = 200; end
if nargin < 5, early_exaggeration = 4; end
if nargin < 6, n_dimensions = 2; end

%
n = size(X, 1);

% Get original affinities matrix
p_ij = get_original_pairwise_affinities(X, perplexity);
p_ij_symmetric = get_symmetric_p_ij(p_ij);

% Initialization
Y = zeros(T, n, n_dimensions);
Y_minus1 = zeros(n, n_dimensions);
Y(1,:,:) = Y_minus1;
Y1 = initialization(X, n_dimensions);
Y(2,:,:) = Y1;
fprintf('Optimizing Low Dimensional Embedding....\n');

% Optimization
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

% Update Rule
Y(t+1,:,:) = Y(t,:,:) - eta * gradient + alpha * (Y(t,:,:) - Y(t-1,:,:)); % Use negative gradient

% Compute current value of cost function
if mod(t, 50) == 0 || t == 2
    cost = sum(p_ij_symmetric .* log(p_ij_symmetric ./ q_ij), 'all');
    fprintf('Iteration %d: Value of Cost Function is %f\n', t, cost);
end

end

%
final_cost = sum(p_ij_symmetric .* log(p_ij_symmetric ./ q_ij), 'all');
fprintf('Completed Low Dimensional Embedding: Final Value of Cost Function is %f\n', final_cost);
solution = squeeze(Y(end,:,:));
end

%%-----------------------%%
%%% Last updated on Nov 06, 2024 1046 hrs IST
%%-----------------------%%