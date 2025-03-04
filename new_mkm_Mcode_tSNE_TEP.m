%% DISCLAIMER: This code-file is in developemenet stage.
%%% This code contains t-SNE using MATLAB in-built functions


% % % " The data in the TE set consists of 22 different simulated running  
% % % data, with each sample in the TE set having 52 observation variables.
% % % In TEP, the training data set [d00.dat] is measured under normal
% % % condition, and contains 500 samples. It is worth noting that
% % % only normal data can be known. The simulation generates 21
% % % types of fault data sets. Each type of fault has 960 samples
% % % and the fault is introduced at the 161st sampling instant.
% % % Note that fault 1 can be easily detected using PCA,
% % % but fault 5 is hard to detect."

%%
close all; clc; clear; 


%% Export .dat file into array file
%%==================================%%

%--- 1(a) Training Dataset d00.dat

Tr = readtable('d00.dat');
T_00 = rows2vars(Tr);
T_00b = T_00(:,2:end);
DTrain = table2array(T_00b);
[mtr, ntr] = size(DTrain);

DTrain_meas = DTrain(:,1:22);
DTrain_xmv = DTrain(:,(ntr-10):end);
DTrain_new = [DTrain_meas,DTrain_xmv];
[m1, n1] = size(DTrain_new)


%--- 1(b) Testing Faulty Datasets dxy_te.dat ...
% % ... (Replace 'xy' as 01, 02, ..., or 21)
% % Here, Faulty datasets chosen: 01, 02, 05, 10, or 19.
% % As per RK sir: 02 to 06 are easier for fault detection

TF_table = readtable('d05_te.dat');
DTest = table2array(TF_table);
[mts, nts] = size(DTest);

DTest_meas = DTest(:,1:22);
DTest_xmv = DTest(:,(nts-10):end);
DTest_new = [DTest_meas,DTest_xmv];
[m2, n2] = size(DTest_new)


Algos = 'exact'; 

Dis_tSNE = 'mahalanobis';

NumDimensions = 2; 

Perplexity_num = 80;%200;

Exaggeration_num = 8;

LearnRate_num = 100;

% % randn('seed', 100);
% rng(1000);
InitialY_val_tr = 1e-4*randn(m1,NumDimensions);

% rng(1000);
InitialY_val_ts = 1e-4*randn(m2,NumDimensions);

Limitz = 161;

%%%--- Ref:: https://in.mathworks.com/help/stats/tsne.html#namevaluepairarguments


%% 1A. NORMALIZATION OF TRAINING DATASET
%%=======================================%%
xm = mean(DTrain_new);
Sdm = std(DTrain_new);
Xbar = (DTrain_new - xm(ones(m1,1),:)) ./ (Sdm(ones(m1,1),:));
CV =cov(Xbar);

%[U,S,V] = svd(CV); 



%% 1C. t-SNE MODEL TRAINING DATASET 
% %=============================================================%%
% % R^(m1 x k) to R^(m1 x 2) space ::: k--> 1st k PCs of X1 == princ

[Y_tr,loss1] = tsne(Xbar,'Algorithm',Algos,'Distance',Dis_tSNE,InitialY = InitialY_val_tr, Perplexity = Perplexity_num,Exaggeration = Exaggeration_num, LearnRate = LearnRate_num);



%% 1D. DETERMINE THE CONTROL STATISTICS LIMIT
%%============================================%%

%%%------ For T^2 Threshold compution, use... ------%%%

% % @INPROCEEDINGS{9213365,
% %   author={Liu, Decheng and Guo, Tianxu and Chen, Maoyin},
% %   booktitle={2019 CAA Symposium on Fault Detection, Supervision and Safety for Technical Processes (SAFEPROCESS)}, 
% %   title={Fault Detection Based on Modified t-SNE}, 
% %   year={2019},
% %   volume={},
% %   number={},
% %   pages={269-273},
% %   keywords={Principal component analysis;Fault detection;Feature extraction;Dimensionality reduction;Manifolds;Sensors;Euclidean distance;dimension reduction;fault detection;local structure;modified t-SNE;mahalanobis distance},
% %   doi={10.1109/SAFEPROCESS45799.2019.9213365}}

%%% Low-Dim Projection martrix 'A' (m x l) via linear LS regression;
%%% Here, prin --> in R(n x m) and l = 2;
% A = inv(princ'*princ)*princ'*Y_tr 


A = inv(Xbar' * Xbar) * Xbar' * Y_tr; %%% Find thr projection matrix, A
A, %%% Display Projection Matrix, A

YA_tr = Xbar * A;

T2_tr = [];
for i = 1:size(Xbar,1)
    % T2_tr(i) = Y_tr(i,:) * inv(YA_tr' * ...
    %             YA_tr/(size(Xbar,1)-1)) * Y_tr(i,:)';

    T2_tr(i) = YA_tr(i,:) * inv(Y_tr' * ...
                Y_tr/(size(Xbar,1)-1)) * YA_tr(i,:)';



end


% % % lim = Area limit which is between 1.4865 && 1.4950
% % % A = Statistical indicies calculated from the data operating in normal fault free conditions
% % % Aera = Area of the statistical index using kernel density estimation
% % % Pnt = Threshold point occupying 99% of the area = CONTROL STATISTICS LIMIT (?)
% [aera,pnt] = icalimit(T2_tr, 1.4865); %% 1.49 ?
% T2knbeta = pnt
% TS = pnt * ones(size(princ,1),1);

f1 = figure(1);
set(gcf, 'WindowState', 'maximized');

% % fprintf('\nT^2 limit: Using ksdensity fn.\n')
fprintf('`````````````````````````````````````\n')
%%%%% [f1,xi1] = ksdensity(T2_tr); %% Kernel smoothing function estimate for univariate and bivariate data
%%%%% [f1,xi1] = ksdensity(x) returns a probability density estimate, f, for the sample data in the vector or two-column matrix x. ...
%%%%% The estimate is based on a normal kernel function,...
%%%%% and is evaluated at equally-spaced points, xi, that cover the range of the data in x. ...
%%%%% ksdensity estimates the density at 100 points for univariate data, or 900 points for bivariate data.

[f1, xi1] = kdeG(T2_tr, 0.3, 5000);

%%%%% subplot(211)
plot(xi1,f1, 'k','LineWidth',2), grid on, 
% % xlim([0 size(xi1)]);
xlabel('Data Values');
ylabel('Density');
title('ksdensity (T^2_{tr})');

CumulDensty1 = cumtrapz(xi1,f1);
T2knbeta = interp1(CumulDensty1/max(CumulDensty1),xi1, 0.99)
TS = T2knbeta * ones(size(Xbar,1),1);

% % % % %%% 2 %%%
% % % % fprintf('\nT^2 limit: Using kde fn.\n')
% % % % fprintf('````````````````````````````````\n')
% % % % [f2,xi2] = kde(T2_tr); %% Kernel density estimate for univariate data
% % % % 
% % % % %%%% [f,xf] = kde(a) estimates a probability density function (pdf) for the univariate data in the vector a and ...
% % % % %%%% returns values f of the estimated pdf at the evaluation points xf. ...
% % % % %%%% kde uses kernel density estimation to estimate the pdf. See Kernel Distribution for more information.
% % % % subplot(212)
% % % % plot(f2, 'b','LineWidth',2), grid on, 
% % % % xlabel('Data Values');
% % % % ylabel('Density');
% % % % title('kde (T^2_{tr})');
% % % % 
% % % % CumulDensty2 = cumtrapz(xi2,f2);
% % % % T2knbeta2 = interp1(CumulDensty2/max(CumulDensty2),xi2, 0.99)
% % % % TS2 = T2knbeta2 * ones(size(princ,1),1);
% % % % 
% % % % %%% 3 %%%
% % % % fprintf('\nT^2 limit: Using prctile fn. on T^2_tr\n')
% % % % fprintf('```````````````````````````````````````````\n')
% % % % % % % Thresh1 = prctile(T2_tr, 95)
% % % % threshhh = prctile(T2_tr,99)




%% 2A. t-SNE FOR NORMALIZED FAULTY TEST DATASET
%================================================%%

SY1 = scale1(DTest_new,xm,Sdm); %% Normalizing DTest using the mean and STDEV of DTrain

% % % % CV_test =cov(SY);
% % % % XT1 = SY*COEFF;
% % % % Xp = SY*COEFF(:,1:k)*COEFF(:,1:k)';   % Xp is the estimation of original data using the PCA model, X=Xp+E
% % % % e = SY1 - Xp; 
% % % 
% % % 
% % % % % In the file coeff = COEFF and X = SCORE
% % % [COEFF2, X2, LATENT2, TSQUARED2, EXPLAINED2, MU2] = pca(SY1); %% LATENT (i.e. EigValued Matix)
% % % 
% % % % % kk=0; %%% Initial count = 0
% % % % % for i = 1:size(LATENT2,1)                                                        
% % % % %     alpha2(i)=sum(LATENT2(1:i))/sum(LATENT2);
% % % % %     if alpha2(i)>=percent
% % % % %             kk=i;
% % % % %             break;
% % % % %     end 
% % % % % end
% % % 
% % % SY = X2(:, 1:k); %% Use 'k'... meanwhile;  'k' = 36 vs. 'kk' = 16 !!

in_fault = SY1; %% SY1

NumDimensions=2; %% 

[Y_ts,loss2] = tsne(in_fault,'Algorithm',Algos,'Distance',Dis_tSNE,InitialY = InitialY_val_ts, Perplexity = Perplexity_num,Exaggeration = Exaggeration_num, LearnRate = LearnRate_num); %% euclidean or mahalanobis



%% 2D. DETERMINE THE T^2 STATISTICS USING LOW-DIM. SPACE (FROM STEP # 2C.)
%%=========================================================================%%



%%--- Either ---%%%
YA_ts = SY1 * A;

T2_ts = [];

for i = 1:size(SY1,1)

    % T2_ts(i) = Y_ts(i,:) * ...
    %             inv(YA_ts' * YA_ts/(size(SY1,1) - 1)) * ...
    %             Y_ts(i,:)';

    T2_ts(i) = YA_ts(i,:) * ...
                inv(Y_ts' * Y_ts/(size(SY1,1) - 1)) * ...
                YA_ts(i,:)';
end

TS2 = T2knbeta * ones(size(T2_ts, 2),1);

f2 = figure(2);
set(gcf, 'WindowState', 'maximized');
plot(T2_ts,'b','LineWidth',1.5),grid on, hold on,
plot(TS2,'r','LineWidth',2),grid on, 
xlim([0 size(TS2,1)]);
xlabel('Observation Instant')
ylabel('tSNE T^2')
legend({'T^2 Stat','Threshold'},'Location','best')

% % % f3 = figure(3);
% % % set(gcf, 'WindowState', 'maximized');
% % % plot(YA_ts,'LineWidth',1.5), grid on,
% % % xlim([0 size(YA_ts,1)]);
% % % xlabel('Observation Instant')
% % % ylabel('YA_{ts}')
% % % title('LD Test Dataset, YA_{ts} = SY1*A')
% % % legend({'YA_{ts_1}','YA_{ts_2}'},'Location','best')
% % % 
% % % f4 = figure(4);
% % % set(gcf, 'WindowState', 'maximized');
% % % plot(Y_ts,'LineWidth',1.5), grid on,
% % % xlim([0 size(YA_ts,1)]);
% % % xlabel('Observation Instant')
% % % ylabel('YA_{ts}')
% % % title('LD Test Dataset, Y_{ts}')
% % % legend({'Y_{ts_1}','Y_{ts_2}'},'Location','best')




% %%%--- or .. ---%%%
% 
% % % [Cof, Sco, Lat2Dts, T2sq, expl, mu_ts] = pca(Y_ts);
% % % for i=1:size(Y_ts,1)
% % %     T2_ts(i) = Y_ts(i,1:end) * diag( 1 ./ Lat2Dts(1:end) ) * Y_ts(i,1:end)';
% % % end
% 
% 
% 
% f2 = figure(2);
%     set(gcf, 'WindowState', 'maximized');
% 
%     plot(TS,'k--', 'LineWidth',3); 
%     hold on;
%     plot(T2_ts,'b', 'LineWidth',2)
% 
%     box_col = [0 1 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(1,Limitz,"FaceColor",box_col,"FaceAlpha",0.1);
%     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(Limitz,size(T2_ts',1),"FaceColor",box_col,"FaceAlpha",0.1)
% 
%     F_Inst = "Fault Instant at n = " + num2str(Limitz);
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
%     xlim([0 size(DTest,1)]);
%     % % legend({'T_{thres}^2','T_{test}^2'},'Location', 'best'),
%     legend({'T_{test}^2'},'Location', 'best'),
% 
%     grid on
% 
% 
% f3 = figure(3); 
%     set(gcf, 'WindowState', 'maximized'); 
%     subplot(211), plot(Y_tr), hold on, xlim([0, m1]), title('Y_{tr}'), xlabel('Instant, n','fontweight','bold'), grid, 
%     subplot(212), plot(Y_ts), hold on, xlim([0, m2]), xline(Limitz,'--k',{'Fault Instant'}), title('Y_{ts}'), xlabel('Instant, n','fontweight','bold'), grid on,
%     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(Limitz,size(Y_ts,1),"FaceColor",box_col,"FaceAlpha",0.1)
% 
% 
% 
% % % % f3 = figure(3); 
% % % %     set(gcf, 'WindowState', 'maximized'); 
% % % %     subplot(211), plot(Y_tr(:,1),'k', 'LineWidth',1.5), hold on, plot(Y_ts(:,1),'r', 'LineWidth',1.5), title('Y_1 = Y(:,1)'), xlabel('Instant, n','fontweight','bold'), grid,
% % % %     F_Inst = "Fault Instant " ;
% % % %     xl = xline(Limitz,'--m',F_Inst,'LineWidth',3);
% % % %     xl.LabelHorizontalAlignment = 'right';
% % % %     xl.LabelVerticalAlignment = 'top';
% % % %     xl.LabelOrientation = 'horizontal';
% % % %     xl.Color = [0.4660 0.6740 0.1880];
% % % %     legend({'Y_{tr}(:,1)','Y_{ts}(:,1)'},'Location','best')
% % % % 
% % % %     subplot(212), plot(Y_tr(:,2),'k', 'LineWidth',1.5), hold on, plot(Y_ts(:,2),'r', 'LineWidth',1.5), title('Y_2 = Y(:,2)'), xlabel('Instant, n','fontweight','bold'), grid,
% % % %     F_Inst = "Fault Instant " ;
% % % %     xl = xline(Limitz,'--m',F_Inst,'LineWidth',3);
% % % %     xl.LabelHorizontalAlignment = 'right';
% % % %     xl.LabelVerticalAlignment = 'top';
% % % %     xl.LabelOrientation = 'horizontal';
% % % %     xl.Color = [0.4660 0.6740 0.1880];
% % % %     legend({'Y_{tr}(:,2)','Y_{ts}(:,2)'},'Location','best')
% % % %     % sgtitle('Note: Y = [Y_1  Y_2]','fontsize',14,'FontName', 'Arial','fontweight','bold') %% Global Title
% % % % 
% % % % 
% % % % 
% % % % f4 = figure(4); 
% % % %     set(gcf, 'WindowState', 'maximized'); 
% % % %     ee2 = [(Y_tr(:,1) - Y_ts(:,1)), (Y_tr(:,2) - Y_ts(:,2))];
% % % %     subplot(211),plot(ee2(:,1)), title('E_1'), xlabel('Instant, n','fontweight','bold'), grid, 
% % % %     xl = xline(Limitz,'--k',F_Inst,'LineWidth',1.5);
% % % %     xl.LabelHorizontalAlignment = 'right';
% % % %     xl.LabelVerticalAlignment = 'top';
% % % %     xl.LabelOrientation = 'horizontal';
% % % %     % xl.Color = [0.4660 0.6740 0.1880];
% % % %     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
% % % %     xregion(Limitz,size(Y_ts,1),"FaceColor",box_col,"FaceAlpha",0.1)
% % % % 
% % % %     subplot(212), plot(ee2(:,2)), title('E_2'), xlabel('Instant, n','fontweight','bold'), grid
% % % %     xl = xline(Limitz,'--k',F_Inst,'LineWidth',1.5);
% % % %     xl.LabelHorizontalAlignment = 'right';
% % % %     xl.LabelVerticalAlignment = 'top';
% % % %     xl.LabelOrientation = 'horizontal';
% % % %     % xl.Color = [0.4660 0.6740 0.1880];
% % % %     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
% % % %     xregion(Limitz,size(Y_ts,1),"FaceColor",box_col,"FaceAlpha",0.1)
% 
% 
% [FP, TN, FN, TP] = FaultRatios(length(T2_ts), Limitz, T2_ts, T2knbeta)
% 
% fprintf('\n[*] True Positive Rate (TPR) (in percent) = %0.4f\n', (TP*100/(TP + FN))) %% = 1 - FNR
% fprintf('\n[*] False Positive Rate (FPR), or False Alarm Rate (FAR) (in percent) = %0.4f\n', (FP*100/(FP + TN))) %% = 1 - TNR
% fprintf('\n[*] False Discovery Rate (FDR) (in percent) = %0.4f\n', (FP*100/(FP + TP))) %% = 1 - PPV
% fprintf('\n[*] Accuracy (in percent) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
% fprintf('\n$$$$$$$$$$$$$$$\n\n')
% 



%% 1B. DIMENSION-REDUCTION OF (NORMALIZED) TRAINING DATASET USING PCA
%%====================================================================%%

%%% In the file coeff = COEFF and X = SCORE
[COEFF, X1, LATENT, TSQUARED, EXPLAINED, MU] = pca(Xbar); %% LATENT (i.e. EigValued Matix)

prompt = 90;
% % prompt = input('\n\nEnter the percentage of significance (b/w 80 to 95 %) = '); %%% self explanatory !!
% prompt = 95;
%percent = input(prompt);
percent = prompt/100;

k = 0; %%% Initial PC count = 0

for i = 1:size(LATENT,1)                                                        
    alpha(i)=sum(LATENT(1:i))/sum(LATENT);
    if alpha(i)>=percent
            k = i;
            break;
    end 
end

% %---------- Display the details of PCs -------------------%

for ij = 1
    fprintf('\n\n\n=================================\n')
    fprintf('||    TE Process: Test Results      ||\n')
    fprintf('======================================\n\n\n')

    disp('//////// PART A:: SYSTEM INFO ///////'), 

    fprintf('\n** Total Number of Observations (Training) = %d \n', m1)
    fprintf('\n** Total Number of Observations (Testing) = %d \n\n', m2)


    fprintf('\n\n\n'); disp('//////// PART B:: PCs INFO ///////'), 
    fprintf('\n')

    fprintf('\n==> The percentage of significance set for PC contribution = %0.4f', prompt)

    fprintf('\n\n(1) No. of PCs chosen = %d out of 33 observable vectors.\n',k)

    fprintf('\n(2) Cumul. PC contrib: \n   ********************\n '), alpha,         

    TotPCcontr = alpha(1,end) * 100;
    fprintf('## Total PC contribution computed = %0.4f\n', TotPCcontr)

end

princ = X1(:,1:k);
in_nf = princ;%% Xbar