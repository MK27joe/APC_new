%% DISCLAIMER: This code-file is in developemenet stage.
%%% This code contains t-SNE using MATLAB in-built functions


% tic %%% Begin the time-count

%%
close all; clc; clear; 


%% LOAD THE DATASHEET
%%=====================%%

% load octuple_tank_data_28_08_24.mat
load octuple_tank_data_11_11_24.mat



%% 0A. Extract your database
%%=======================%%

%%%% T == res
%%% Outputs --> Col 2 to 9
%%% Inputs --> Col 10 to 13
%%% Disturbances --> Col 14 to 15

xt = res(:,2:end);  
[M,N] = size(xt);

Dtrain = xt(1:M/2,:);
[m1,n1] = size(Dtrain);

Dtest = xt(1+(M/2):M,:);
[m2,n2] = size(Dtest);

%%%--- Enter the dimension (l) of the low-dimensional space Y:
NumDimensions = 2; 

%%%--- 
numNeighbors = 5; % Number of neighbors



%% 1A. NORMALIZATION OF TRAINING DATASET
%%=======================================%%

xm = mean(Dtrain);
Sdm = std(Dtrain);

Xbar = (Dtrain - xm(ones(m1,1),:)) ./ (Sdm(ones(m1,1),:));

CV =cov(Xbar);
[U,S,V] = svd(CV); 



%% 1C. t-SNE MODEL TRAINING DATASET 
% %=================================%%

% % % [Y, R] = isomap(Xbar, Perplexity_num, NumDimensions)
% % % %ISOMAP Implements the Isomap algorithm for dimensionality reduction.
% % % %
% % % %   [Y, R] = isomap(X, k, d) performs Isomap on dataset X:
% % % %   - X: N-by-D matrix of N points in D-dimensional space
% % % %   - Perplexity_num: Number of nearest neighbors for graph construction
% % % %   - NumDimensions: Target dimensionality
% % % %   - Y: N-by-d matrix of the embedded coordinates
% % % %   - R: Distance matrix of geodesic distances

%%%%%-----------------------------------------------------

% % % 
% % % % Compute the pairwise distance matrix
% % % D = pdist2(Xbar, Xbar);
% % % 
% % % % Find the k-nearest neighbors for each point
% % % [idx, ~] = knnsearch(Xbar, Xbar, 'K', numNeighbors + 1);
% % % 
% % % % Construct the adjacency matrix for the neighborhood graph
% % % adjMatrix = false(size(D));
% % % 
% % % for i = 1:size(D, 1)
% % %     [~, idx] = sort(D(i, :), 'ascend');
% % %     adjMatrix(i, idx(2:numNeighbors+1)) = true; % idx(1) is the point itself
% % % end
% % % 
% % % % Create the graph
% % % G = graph(adjMatrix, D); %%% <----- PROBLEM IS RIGHT HERE ??!
% % % 
% % % % Compute the shortest paths
% % % shortestPaths = distances(G);
% % % 
% % % % Perform classical MDS on the shortest path distances
% % % [~, eigVectors] = cmdscale(shortestPaths);
% % % 
% % % % Select the top dimensions
% % % reducedData = eigVectors(:, 1:numDimensions);
% % % 
% % % % Plot the reduced data
% % % scatter(reducedData(:, 1), reducedData(:, 2));
% % % title('Isomap Reduced Data');
% % % xlabel('Dimension 1');
% % % ylabel('Dimension 2');


%%%%%-----------------------------------------------------


% Compute the pairwise distance matrix
D = pdist2(Xbar, Xbar);

% Find the k-nearest neighbors for each point
[idx, ~] = knnsearch(Xbar, Xbar, 'K', numNeighbors + 1);

% Construct the adjacency matrix for the neighborhood graph
adjMatrix = false(size(D));
for i = 1:size(D, 1)
    adjMatrix(i, idx(i, 2:end)) = true; % idx(i, 1) is the point itself
end

% Make the adjacency matrix symmetric
adjMatrix = adjMatrix | adjMatrix';

% Create the graph
G = graph(adjMatrix, D); %%% <----- PROBLEM IS RIGHT HERE ??!



% %% 1D. DETERMINE THE CONTROL STATISTICS LIMIT
% %%============================================%%
% 
% %%%------ For T^2 Threshold compution, use... ------%%%
% 
% % % @INPROCEEDINGS{9213365,
% % %   author={Liu, Decheng and Guo, Tianxu and Chen, Maoyin},
% % %   booktitle={2019 CAA Symposium on Fault Detection, Supervision and Safety for Technical Processes (SAFEPROCESS)}, 
% % %   title={Fault Detection Based on Modified t-SNE}, 
% % %   year={2019},
% % %   volume={},
% % %   number={},
% % %   pages={269-273},
% % %   keywords={Principal component analysis;Fault detection;Feature extraction;Dimensionality reduction;Manifolds;Sensors;Euclidean distance;dimension reduction;fault detection;local structure;modified t-SNE;mahalanobis distance},
% % %   doi={10.1109/SAFEPROCESS45799.2019.9213365}}
% 
% %%% Low-Dim Projection martrix 'A' (m x l) via linear LS regression;
% %%% Here, prin --> in R(n x m) and l = 2;
% % A = inv(princ'*princ)*princ'*Y_tr 
% 
% 
% A = inv(Xbar'*Xbar)*Xbar'*Y_tr;
% A,
% 
% YA_tr = Xbar*A;
% 
% T2_tr = [];
% for i = 1:size(Xbar,1)
%     T2_tr(i) = Y_tr(i,:)*inv(YA_tr'*YA_tr/(size(Xbar,1)-1))*Y_tr(i,:)';
% end
% 
% % % % % [aera,pnt] = icalimit(T2_tr,1.49); 
% % % % % % % lim = Area limit which is between 1.4865 && 1.4950
% % % % % % % A = Statistical indicies calculated from the data operating in normal fault free conditions
% % % % % % % Aera = Area of the statistical index using kernel density estimation
% % % % % % % Pnt = Threshold point occupying 99% of the area = CONTROL STATISTICS LIMIT (?)
% % % % % T2knbeta = pnt;
% % % % % TS = pnt*ones(size(princ,1),1);
% % % % 
% % % % % [f,xi] = ksdensity(T2_tr); %% Kernel smoothing function estimate for univariate and bivariate data
% % % % %%%%% [f,xi] = ksdensity(x) returns a probability density estimate, f, for the sample data in the vector or two-column matrix x. ...
% % % % %%%%% The estimate is based on a normal kernel function,...
% % % % %%%%% and is evaluated at equally-spaced points, xi, that cover the range of the data in x. ...
% % % % %%%%% ksdensity estimates the density at 100 points for univariate data, or 900 points for bivariate data.
% % % % [f,xi] = kde(T2_tr); %% Kernel density estimate for univariate data
% %%%% [f,xf] = kde(a) estimates a probability density function (pdf) for the univariate data in the vector a and ...
% %%%% returns values f of the estimated pdf at the evaluation points xf. ...
% %%%% kde uses kernel density estimation to estimate the pdf. See Kernel Distribution for more information.
% 
% [f,xi] = kdeG(T2_tr, 0.1510, 2000);
% 
% figure(1)
% plot(xi,f, 'LineWidth',2), grid on, 
% xlabel('Data Values');
% ylabel('Density');
% title('Kernel Density Estimation');
% 
% Thresh1 = prctile(T2_tr, 95)
% 
% CumulDensty = cumtrapz(f, xi);
% T2knbeta = interp1(CumulDensty/max(CumulDensty),f, 0.95)
% % TS = T2knbeta * ones(size(princ,1),1);
% 
% %%%------ or... ------%%%
% 
% % T2knbeta = (k*(size(Y_tr,1)^2-1)/(size(Y_tr,1)*(size(Y_tr,1)-k))) * finv(0.95,k,size(Y_tr,1)-k);
% % fprintf('\n(**) T2knbeta = %0.4f\n', T2knbeta)
% % % % TS = T2knbeta*ones(size(Y_tr,1),1);
% 
% 
% 
% %% 2A. INTRODUCE FAULT INTO TEST DATASET
% %%========================================%%
% 
% fprintf('\n\n\n\n\n'); disp('//////// PART C:: FAULT DETAILS ///////'),fprintf('\n\n'),
% 
% 
% lim = 1000; %%input('\nEnter the instant of fault, lim =  ');
% fprintf('\nFault introduced at %d-th observation in Test dataset.', lim)
% 
% 
% IndX = 4; %%% Tank# 4
% fprintf('\nTank # Selection = %d\n',IndX);
% 
% 
% %%% ==== Uncomment only those (following) lines that need to be executed ====%%%
% 
% for xyz = 1
% 
% 
%     % % %-------------------- 1. Bias Fault -------------------------% %
%     % FaultID = 'Bias';
%     % Fvalue = 10;
%     % FvalueS = num2str(Fvalue);
%     % Limitz = lim;
%     % for i = 1:m2
%     %     if(i>Limitz)
%     %         Dtest(i,IndX) = Dtest(i,IndX) + Fvalue;
%     %     end
%     % end
% 
% 
%     % % -------------------- 2. Drift Fault -------------------------% %
%     FaultID = 'Drift';
%     Fvalue = 0.07; %% The slope of drift fault
%     FvalueS = num2str(Fvalue);
%     r = [];
%     Limitz = lim;
%     for i = 1:(m2-Limitz)
%         r(i)=i;
%     end
% 
%     for i=1:m2
%         if(i>Limitz)
%             Dtest(i,IndX) = Dtest(i,IndX) + Fvalue*r(i-Limitz);
%         end
%     end
% 
% 
%     % % % --------------- 3. Drift + Prec. Deg. Fault --------------------% %
%     % 
%     % FaultID = 'Drift+PD ';
%     % Dslope = 0.1;
%     % Mag_PD = 0.45;
%     % Limitz = lim;
%     % FvalueS = "Slope = " + num2str(Dslope) + "; Mag-PD = " + num2str(Mag_PD);
%     % 
%     % r = [];
%     % for i = 1:(m2-Limitz)
%     %     r(i)=i;
%     % end
%     % 
%     % for i=1:m2
%     %     if(i>Limitz)
%     %         Dtest(i,IndX) = Dtest(i,IndX) + Dslope*r(i-Limitz)*Mag_PD*rand(1);
%     %     end
%     % end
% 
% 
% 
%     % % % ------------------ 4. Freeze Fault --------------------% %
%     % 
%     % FaultID = 'Freeze';
%     % m2 = size(Dtest,1); 
%     % % lim = size(Dtest1,1)/2; 
%     % 
%     % idx = round(randi(numel(Dtest(lim+1:end, IndX)))/4); %%  Select a random cell index, idx, corr. to MFault_ID vector
%     % 
%     % idx,
%     % 
%     % Limitz = lim+idx;
%     % 
%     % A = Dtest(Limitz, IndX); %% Select that cell value corr. to idx
%     % 
%     % FvalueS = "at n = " + num2str(Limitz);
%     % for i =1:m2
%     %     if(i>Limitz)
%     %         Dtest(i,IndX) = 3*A; %% replace remaining cells as 'A' %%% Bias of 2-5
%     %     end
%     % end
% 
% 
%     % % % ------------------ 5. Intermittent Fault --------------------% %
%     % 
%     % FaultID = 'Intermittent';
%     % a= lim + 25;
%     % b= lim + 125;
%     % c= m2 - 250;
%     % d= m2 - 150;
%     % FvalueS = "b/w n= (" + num2str(a) + ", " + num2str(b) + ") and (" + num2str(c) + ", " + num2str(d) + ")";
%     % 
%     % for i=1:m2
%     %     if (((i > a) && (i < b)) || ((i > c) && (i < d))) %||((i>1050) && (i<1150)))
%     %         Dtest(i,IndX) = Dtest(i,IndX) + 10;
%     %     end
%     % end
% 
% end
% 
% 
% 
% %% 2B. NORMALIZATION OF FAULTY TEST DATASET
% %%===========================================%%
% 
% fprintf('\n\n\n'); disp('//////// PART D::  ///////'), 
% fprintf('\n')
% 
% SY1 = scale1(Dtest,xm,Sdm); %% Normalizing Dtest using the mean and STDEV of Dtrain
% 
% CV_test =cov(SY1);
% 
% 
% %% 2C. t-SNE FOR NORMALIZED FAULTY TEST DATASET
% %===============================================%%
% 
% % NumDimensions=2; %% 
% [Y_ts,loss2] = tsne(SY1,'Algorithm',Algos,'Distance',Dis_tSNE,InitialY = InitialY_val_tr, Perplexity = Perplexity_num,Exaggeration = Exaggeration_num, LearnRate = LearnRate_num); %% euclidean or mahalanobis
% 
% 
% 
% %% 2D. DETERMINE THE T^2 STATISTICS USING LOW-DIM. SPACE (FROM STEP # 2C.)
% %%=========================================================================%%
% 
% %%%--- Either ---%%%
% YA_ts = SY1*A;
% 
% T2_ts = [];
% 
% for i = 1:size(SY1,1)
%     T2_ts(i) = Y_ts(i,:)*inv(YA_ts'*YA_ts/(size(SY1,1)-1))*Y_ts(i,:)';
% end
% 
% 
% 
% f2 = figure(2);
%     set(gcf, 'WindowState', 'maximized');
% 
%     % plot(TS,'k--', 'LineWidth',3); 
%     % hold on;
%     plot(T2_ts,'b', 'LineWidth',1.2)
% 
%     box_col = [0 1 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(1,Limitz,"FaceColor",box_col,"FaceAlpha",0.1);
%     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(Limitz,size(T2_ts',1),"FaceColor",box_col,"FaceAlpha",0.1)
% 
%     F_Inst = "Fault Instance at n = " + num2str(Limitz);
%     xl = xline(Limitz,'--m',F_Inst,'LineWidth',3);
%     xl.LabelHorizontalAlignment = 'right';
%     xl.LabelVerticalAlignment = 'middle';
%     xl.Color = [00 0 0.40];
%     Thrsh = "Threshold = " + num2str(T2knbeta);
%     yl = yline(T2knbeta,'-.',Thrsh, 'LineWidth',3);
%     yl.LabelHorizontalAlignment = 'left';
%     yl.LabelVerticalAlignment = 'top';
%     yl.Color = [0.80 0 0.40];
%     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
%     xlim([0 size(Dtest,1)]);
%     % % legend({'T_{thres}^2','T_{test}^2'},'Location', 'best'),
%     legend({'T_{test}^2'},'Location', 'best'),
% 
%     grid
% 
% 
% f3 = figure(3); 
%     set(gcf, 'WindowState', 'maximized'); 
%     subplot(211), plot(Y_tr), hold on, title('Y_{tr}'), xlabel('Instance, n','fontweight','bold'), grid, 
%     subplot(212), plot(Y_ts), hold on, xline(Limitz,'--k',{'Fault Instance'}), title('Y_{ts}'), xlabel('Instance, n','fontweight','bold'), grid on,
%     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(Limitz,size(Y_ts,1),"FaceColor",box_col,"FaceAlpha",0.1)
% 
% 
% 
% f4 = figure(4); 
%     set(gcf, 'WindowState', 'maximized'); 
%     subplot(211), plot(Y_tr(:,1),'k', 'LineWidth',1.5), hold on, plot(Y_ts(:,1),'r', 'LineWidth',1.5), title('Y_1 = Y(:,1)'), xlabel('Instance, n','fontweight','bold'), grid,
%     F_Inst = "Fault Instance " ;
%     xl = xline(Limitz,'--m',F_Inst,'LineWidth',3);
%     xl.LabelHorizontalAlignment = 'right';
%     xl.LabelVerticalAlignment = 'top';
%     xl.LabelOrientation = 'horizontal';
%     xl.Color = [0.4660 0.6740 0.1880];
%     legend({'Y_{tr}(:,1)','Y_{ts}(:,1)'},'Location','best')
% 
%     subplot(212), plot(Y_tr(:,2),'k', 'LineWidth',1.5), hold on, plot(Y_ts(:,2),'r', 'LineWidth',1.5), title('Y_2 = Y(:,2)'), xlabel('Instance, n','fontweight','bold'), grid,
%     F_Inst = "Fault Instance " ;
%     xl = xline(Limitz,'--m',F_Inst,'LineWidth',3);
%     xl.LabelHorizontalAlignment = 'right';
%     xl.LabelVerticalAlignment = 'top';
%     xl.LabelOrientation = 'horizontal';
%     xl.Color = [0.4660 0.6740 0.1880];
%     legend({'Y_{tr}(:,2)','Y_{ts}(:,2)'},'Location','best')
%     % sgtitle('Note: Y = [Y_1  Y_2]','fontsize',14,'FontName', 'Arial','fontweight','bold') %% Global Title
% 
% 
% 
% f5 = figure(5); 
%     set(gcf, 'WindowState', 'maximized'); 
%     ee2 = [(Y_tr(:,1) - Y_ts(:,1)), (Y_tr(:,2) - Y_ts(:,2))];
%     subplot(211),plot(ee2(:,1)), title('E_1'), xlabel('Instance, n','fontweight','bold'), grid, 
%     xl = xline(Limitz,'--k',F_Inst,'LineWidth',1.5);
%     xl.LabelHorizontalAlignment = 'right';
%     xl.LabelVerticalAlignment = 'top';
%     xl.LabelOrientation = 'horizontal';
%     % xl.Color = [0.4660 0.6740 0.1880];
%     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(Limitz,size(Y_ts,1),"FaceColor",box_col,"FaceAlpha",0.1)
% 
%     subplot(212), plot(ee2(:,2)), title('E_2'), xlabel('Instance, n','fontweight','bold'), grid
%     xl = xline(Limitz,'--k',F_Inst,'LineWidth',1.5);
%     xl.LabelHorizontalAlignment = 'right';
%     xl.LabelVerticalAlignment = 'top';
%     xl.LabelOrientation = 'horizontal';
%     % xl.Color = [0.4660 0.6740 0.1880];
%     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(Limitz,size(Y_ts,1),"FaceColor",box_col,"FaceAlpha",0.1)


