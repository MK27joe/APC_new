function y0 = initialization(X, n_dimensions, initialization)
% Obtain initial solution for t-SNE either randomly or using PCA.
%
% Parameters:
% X (matrix): The input data array.
% n_dimensions (integer): The number of dimensions for the output solution. Default is 2.
% initialization (string): The initialization method. Can be 'random' or 'PCA'. Default is 'random'.
%
% Returns:Page 39 of 59
% y0 (matrix): The initial solution for t-SNE.
%
% Throws:
% MException: If the initialization method is neither 'random' nor 'PCA'.


    if nargin < 2 %% nargin means "Number of function input arguments"
    n_dimensions = 2;
    end
    
    if nargin < 3
    initialization = 'random';
    end
    
    % Sample Initial Solution
    %--- strcmp(S1,S2) ::  Compare strings or character vectors; =1 if TRUE
    if strcmp(initialization, 'random') || ~strcmp(initialization, 'PCA')
        y0 = 1e-4 * randn(size(X, 1), n_dimensions);
    elseif strcmp(initialization, 'PCA')
        X_centered = X - mean(X, 1);
        [~, ~, Vt] = svd(X_centered);
        y0 = X_centered * Vt(:, 1:n_dimensions);
    else
        error('Initialization must be either ''random'' or ''PCA''');
    end

    fprintf('\nInitial Solution of tSNE is completed.\n');
end




%%-----------------------%%
%%% Last updated on Nov 06, 2024 1046 hrs IST
%%-----------------------%%






















