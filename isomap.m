function [Y, R] = isomap(X, k, d)
%ISOMAP Implements the Isomap algorithm for dimensionality reduction.
%
%   [Y, R] = isomap(X, k, d) performs Isomap on dataset X:
%   - X: N-by-D matrix of N points in D-dimensional space
%   - k: Number of nearest neighbors for graph construction
%   - d: Target dimensionality
%   - Y: N-by-d matrix of the embedded coordinates
%   - R: Distance matrix of geodesic distances

    % Step 1: Compute k-nearest neighbors
    [~, distances] = knnsearch(X, X, 'K', k+1); % +1 because each point is its own nearest neighbor
    distances = distances(:, 2:end); % Remove self-distance

    % Step 2: Construct the graph
    G = graph(min(distances, distances'), 'upper');
    
    % Step 3: Compute shortest paths (geodesic distances)
    R = distances(G.Edges.EndNodes); % Initialize with direct distances
    for i = 1:size(X, 1)
        [~, D] = shortestpath(G, i);
        R(i, :) = D;
    end
    
    % Step 4: Apply Classical MDS (Multidimensional Scaling)
    [V, E] = eig(R.^2, 'vector');
    [~, idx] = sort(E); % Sort eigenvalues
    V = V(:, idx); % Rearrange eigenvectors according to sorted eigenvalues
    
    % Use d+1 smallest non-zero eigenvalues for embedding
    Y = V(:, 2:d+1) * sqrt(diag(E(2:d+1)));
    Y = real(Y); % Ensure all values are real numbers
    
    % Normalize the coordinates
    Y = Y - mean(Y, 1);
    Y = Y ./ norm(Y);
end


%%%%%% Code from Grok. Feb 02, 2025