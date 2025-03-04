function [Y, costs] = tsne_custom_distance(X, no_dims, initial_dims, perplexity, theta, num_iterations, rand_seed, distance_func)
    if ~exist('distance_func','var') || isempty(distance_func)
        distance_func = @(x, y) sqrt(sum((x - y)).^2); %% Default = Eucledian Distance
    end

    % Compute pair-wise distances using user-defined function

    n = size(X, 1);
    D = zeros(n, n);

    for i = 1:n
        for j = 1:n
            D(i,j) = distance_func(X(i,:), X(j,:));
        end
    end

    % Compute the joint probabilities wrt the calculated distances
    P = d2p (D, perplexity, 1e-5); %% may require adjustments as per the tSNE version

    % Run the t-SNE algorithm with the 'P' values
    [Y, costs] = tsne(P, no_dims, initial_dims, perplexity, theta, num_iterations, rand_seed);

end

%%% Grok-2___Main function
%%% Feb , 2025