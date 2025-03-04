function H = compute_entropy(D, beta)
    P = exp(-D * beta);
    sumP = sum(P);
    H = log(sumP) + beta * sum(D .* P) / sumP;
end

%%% Grok-2___helper function to compute Entropy
%%% Feb , 2025