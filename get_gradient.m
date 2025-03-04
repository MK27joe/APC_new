function gradient = get_gradient(p_ij, q_ij, Y)
% Obtain gradient of cost function at current point Y.
%
% Parameters:
% p_ij: The joint probability distribution matrix.
% q_ij: The Student's t-distribution matrix.
% Y: The current point in the low-dimensional space.
%
% Returns:
% gradient: The gradient of the cost function at the current point Y.


n = size(p_ij, 1);

% Compute gradient
gradient = zeros(n, size(Y, 2));

for i = 1:n
% Equation 5
    diff = Y(i, :) - Y;
    A = p_ij(i, :) - q_ij(i, :);
    B = (1 + vecnorm(diff, 2, 2)).^(-1);
    C = diff;
    gradient(i, :) = 4 * sum((A' .* B) .* C, 1);
end

end

%%-----------------------%%
%%% Last updated on Nov 06, 2024 1046 hrs IST
%%-----------------------%%