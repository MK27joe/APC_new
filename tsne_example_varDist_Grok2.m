%%% Grok-2___helper function to convert distances to probabilities
%%% Feb , 2025

clear, clc;

%%

X = randn(1000,10);


%%
 no_dims = 2;
 perplexity = 30;


%%
custom_distance = @(x,y) sum(abs(x-y)); %% Manhattan Distance


%% = tsne_custom_distance(X, no_dims, initial_dims, perplexity, theta, num_iterations, rand_seed, distance_func)
[Y_custom, costs] = tsne_custom_distance(X, no_dims, [], perplexity, 0.5, 1000, [], custom_distance);
                    

%%
scatter(Y(:,1), Y(:,2),'c','filled'), grid