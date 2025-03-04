function p_ij = get_original_pairwise_affinities(X, perplexity)
    % Function to obtain affinities-matrix p_ij in the Higher Dimension Space
    %
    % Parameters:
    % X (matrix): The input data array.
    % perplexity (int): The perplexity value for the grid search.
    %
    % Returns:
    % matrix: The pairwise affinities matrix.

    n = size(X, 1);
    disp('Computing Pairwise Affinities....');
    p_ij = zeros(n, n);

    for i = 1:n

        % Equation 1 numerator (refer Page 32 of 59 of tSNE w/ Py and Matlab.pdf)
        diff = X(i, :) - X;

        sigma_i = grid_search(diff, i, perplexity); % Grid Search for σ_i %% open the function file to edit

        norm = vecnorm(diff, 2, 2);
        p_ij(i, :) = exp(-(norm.^2) / (2 * sigma_i^2));

        % Set p = 0 when j = i
        p_ij(i, i) = 0;

        % Equation 1
        p_ij(i, :) = p_ij(i, :) / sum(p_ij(i, :));

    end

    % Set 0 values to minimum value (ε approx. = 0)
    epsilon = realmin('double');
    p_ij = max(p_ij, epsilon);
    disp('Completed Pairwise Affinities Matrix.');
end

%%-----------------------%%
%%% Last updated on Nov 06, 2024 1046 hrs IST
%%-----------------------%%