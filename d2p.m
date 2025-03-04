function P = d2p(D, PerplX, Tol)
    [n,~] =size(D);
    P =zeros(n,n);
    beta = ones(n,1);
    logU = log(PerplX);

    for i = 1:n
        betamin = -Inf;
        betamax = Inf;
        Di = D(i,:).^2;
        Di = Di/(sum(Di)/n);

        H = compute_entropy(Di, beta(i));
        Hdiff = H - logU;

        tries = 50;
        while abs(Hdiff) >Tol && tries >0

            if Hdiff >0
                betamin = beta(i);
                if isinf(betamax)
                    beta(i) = beta(i) * 2;
                else
                    beta(i) = (beta(i) + betamin) / 2;
                end

            else
                betamax = beta(i);
                if isinf(betamin)
                    beta(i) = beta(i) / 2;
                else
                    beta(i) = (beta(i) + betamin) /2;
                end
            end

            H = compute_entropy(Di, beta(i));
            Hdiff = H -logU;
            tries = tries -1;
        end
        
        P(i,:) = exp(-D(i,:).^2 *beta(i));
        P(i,i) = 0;
        P(i,:) = P(i,:) / sum(P(i,:));
    end
    P = (P + P')/(2*n);


end

%%% Grok-2___helper function to convert distances to probabilities
%%% Feb , 2025