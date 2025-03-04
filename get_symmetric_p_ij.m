function p_ij_symmetric = get_symmetric_p_ij(p_ij)
% Function to obtain symmetric affinities matrix utilized in t-SNE.
%
% Parameters:
% p_ij (matrix): The input affinity matrix.
%
% Returns:
% matrix: The symmetric affinities matrix.
fprintf('Computing Symmetric p_ij matrix....\n');

n = size(p_ij, 1);

p_ij_symmetric = zeros(n, n); %% Initialization

for i = 1:n
    for j = 1:n
        p_ij_symmetric(i, j) = (p_ij(i, j) + p_ij(j, i)) / (2 * n);
    end
end

% Set 0 values to minimum MATLAB value (eps approx. = 0)
% eps(X) is the positive distance from ABS(X) to the next larger in...
% magnitude floating point number of the same precision as X.
p_ij_symmetric = max(p_ij_symmetric, eps);


fprintf('Completed Symmetric p_ij Matrix. \n');
end

%%-----------------------%%
%%% Last updated on Nov 06, 2024 1046 hrs IST
%%-----------------------%%