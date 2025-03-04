clear, clc;

%% RandN dataset

X = randn(1000,100);

%% Define your custom distance using function handles

%%% Method-A:
% myDistance = @(x,y) sum(abs(x - y)); %%% Manhattan Distance

% customDistance = @(XI, XJ) arrayfun(@(k) myDistance(XI, XJ(k,:)), 1:size(XJ,1))';
% % % --- XI is 1-by-n, XJ is m-by-n, output is m-by-1

%%% Method-B:
customDistance = @(XI, XJ) sum(abs(XJ- XI),2);  %%sqrt( (p21-p11)^2 + (p22-p12)^2 )  %%   sqrt(sum((XI - XJ))^2)



%% Compute Pairwise distance matrix using the defined distance

D_condensed = pdist(X, customDistance);
D = squareform(D_condensed);


%% Run tSNE

% Y = tsne(X, 'Distance', customDistance,'Perplexity',30, 'NumDimensions',2);
Y = tsne(D, 'Distance', customDistance,'Perplexity',30, 'NumDimensions',2);

%% Visualize the result

scatter(Y(:,1), Y(:,2),'c','filled'), grid
title('t-SNE with custom distance')
xlabel('Component-1')
ylabel('Component-2')

