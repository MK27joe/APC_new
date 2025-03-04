function [f, xi] = kdeG(data, bandwidth, nPoints)
%KDE Performs Kernel Density Estimation for a set of data points
%
%   [f, xi] = kde(data, bandwidth, nPoints) estimates the density:
%   - data: vector of data points
%   - bandwidth: bandwidth of the kernel (scalar or vector for adaptive)
%   - nPoints: number of points for the density estimation grid
%   - f: estimated density values
%   - xi: points at which the density is estimated

    if nargin < 3
        nPoints = 1000;
    end
    
    if nargin < 2
        % If no bandwidth is specified, use Silverman's rule of thumb
        bandwidth = 1.06 * std(data) * numel(data)^(-1/5);
    end
    
    % Ensure data is a column vector
    data = data(:);
    
    % Define the range of x for plotting or analysis
    minX = min(data);
    maxX = max(data);
    xi = linspace(minX, maxX, nPoints)';
    
    % Compute the density
    n = length(data);
    f = zeros(size(xi));
    
    % Gaussian kernel
    for i = 1:nPoints
        f(i) = sum(normpdf((xi(i) - data) / bandwidth)) / (n * bandwidth);
    end
end

