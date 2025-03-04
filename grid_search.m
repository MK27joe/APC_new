function sigma = grid_search(diff_i, i, perplexity)
    % Helper function to obtain σ's based on user-specified perplexity.
    %
    % Parameters:Page 34 of 59
    % diff_i (matrix): Array containing the pairwise differences between data points.
    % i (int): Index of the current data point.
    % perplexity (int): User-specified perplexity value.
    %
    % Returns:
    % float: The value of σ that satisfies the perplexity condition.
    result = inf; % Set first result to be infinity
    norm = vecnorm(diff_i, 2, 2);
    std_norm = std(norm); % Use standard deviation of norms to define search space

    for sigma_search = linspace(0.01 * std_norm, 5 * std_norm, 200)
        % Equation 1 Numerator
        p = exp(-(norm.^2) / (2 * sigma_search^2));

        % Set p = 0 when i = j
        p(i + 1) = 0; % MATLAB uses 1-based indexing

        % Equation 1 (ε -> 0)
         
        % % epsilon = nextafter(0, 1); %% replace 'nextafter'. It works
        % only in Python!!
        % epsilon = 5e-324; %% from online Xecutn of:>> print(math.nextafter(0, 1)) 
        epsilon = 5e-6;


        p_new = max(p / sum(p), epsilon);

        % Shannon Entropy
        H = -sum(p_new .* log2(p_new));

        % Get log(perplexity equation) as close to equality
        if abs(log(perplexity) - H * log(2)) < abs(result)
            result = log(perplexity) - H * log(2);
            sigma = sigma_search;
        end

    end
    % sigma
    % fprintf('\nSigma Value is determined based on used-defined perplexity\n');
end

%%-----------------------%%
%%% Last updated on Nov 06, 2024 1046 hrs IST
%%-----------------------%%