function q_ij = get_low_dimensional_affinities(Y)
% Obtain low-dimensional affinities.
%
% Parameters:
% Y (matrix): The low-dimensional representation of the data points.Page 41 of 59
%
% Returns:
% q_ij (matrix): The low-dimensional affinities matrix.

n = size(Y, 1);
q_ij = zeros(n, n);

for i = 1:n
    % Equation 4 Numerator
    diff = Y(i, :) - Y;
    norm = vecnorm(diff, 2, 2);
    q_ij(i, :) = (1 + norm.^2).^(-1);
end

% Set p = 0 when j = i
q_ij(logical(eye(size(q_ij)))) = 0;

% Equation 4
q_ij = q_ij / sum(q_ij(:));

% Set 0 values to minimum MATLAB value (ε approx. = 0)
%----> eps(X) is the positive distance from ABS(X) to the next larger in
% magnitude floating point number of the same precision as X.
%----> eps, with no arguments, is the distance from 1.0 to the next larger double
% precision number, that is eps with no arguments returns 2^(-52).
epsilon = eps;
q_ij = max(q_ij, epsilon);

end

%%-----------------------%%
%%% Last updated on Nov 06, 2024 1046 hrs IST
%%-----------------------%%