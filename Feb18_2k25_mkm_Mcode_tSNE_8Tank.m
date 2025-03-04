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



%% 0B. GLOBAL SETTINGS FOR ...
%%=============================%%



%%%%---> [ii] t-SNE ALGORITHM
%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~



Algos = 'exact'; 


%%%--- Select the choice of distance measured in t-SNE algorithm:
% % 'euclidean' (default) | 

Dis_tSNE = 'mahalanobis';


%%%--- Enter the dimension (l) of the low-dimensional space Y:
NumDimensions = 2; 

%%%--- Enter the perplexity (= no. of neighbours around xi//yi) :
% %- The 'Perplexity' value cannot be greater than the number of rows(data)
Perplexity_num = 40;

%%%--- Enter the Exaggeration value :
Exaggeration_num = 4;

%%%--- Enter the Learning Rate :
LearnRate_num = 100;

%%%--- Enter the Initial Assumption of Lower Dimensional Space Y :
InitialY_val_tr = 1e-4*randn(m1,NumDimensions);
InitialY_val_ts = 1e-4*randn(m2,NumDimensions);

%%%--- Ref:: https://in.mathworks.com/help/stats/tsne.html#namevaluepairarguments


%% 1A. NORMALIZATION OF TRAINING DATASET
%%=======================================%%

xm = mean(Dtrain);
Sdm = std(Dtrain);

Xbar = (Dtrain - xm(ones(m1,1),:)) ./ (Sdm(ones(m1,1),:));

CV =cov(Xbar);
[U,S,V] = svd(CV); 





%% 1C. t-SNE MODEL TRAINING DATASET 
% %=============================================================%%
% % R^(m1 x k) to R^(m1 x 2) space ::: k--> 1st k PCs of X1 == princ

[Y_tr,loss1] = tsne(Xbar,'Algorithm',Algos,'Distance',Dis_tSNE,InitialY = InitialY_val_tr, Perplexity = Perplexity_num,Exaggeration = Exaggeration_num, LearnRate = LearnRate_num);



%% 1D. DETERMINE THE CONTROL STATISTICS LIMIT
%%============================================%%


A = inv(princ'*princ)*princ'*Y_tr;
A,

YA_tr = princ*A;

T2_tr = [];
for i = 1:size(princ,1)
    T2_tr(i) = Y_tr(i,:)*inv(YA_tr'*YA_tr/(size(princ,1)-1))*Y_tr(i,:)';
end

% [aera,pnt] = icalimit(T2_tr,1.49); 
% % % lim = Area limit which is between 1.4865 && 1.4950
% % % A = Statistical indicies calculated from the data operating in normal fault free conditions
% % % Aera = Area of the statistical index using kernel density estimation
% % % Pnt = Threshold point occupying 99% of the area = CONTROL STATISTICS LIMIT (?)
% T2knbeta = pnt;
% TS = pnt*ones(size(princ,1),1);

% [f,xi] = ksdensity(T2_tr); %% Kernel smoothing function estimate for univariate and bivariate data
%%%%% [f,xi] = ksdensity(x) returns a probability density estimate, f, for the sample data in the vector or two-column matrix x. ...
%%%%% The estimate is based on a normal kernel function,...
%%%%% and is evaluated at equally-spaced points, xi, that cover the range of the data in x. ...
%%%%% ksdensity estimates the density at 100 points for univariate data, or 900 points for bivariate data.

% % % [f,xi] = kde(T2_tr); %% Kernel density estimate for univariate data
[f,xi] = kdeG(T2_tr);
% [f,xi]=icalimit(T2_tr, 1.485)


%%%% [f,xf] = kde(a) estimates a probability density function (pdf) for the univariate data in the vector a and ...
%%%% returns values f of the estimated pdf at the evaluation points xf. ...
%%%% kde uses kernel density estimation to estimate the pdf. See Kernel Distribution for more information.
figure(1)
plot(xi,f, 'LineWidth',2), grid on, 
xlabel('Data Values');
ylabel('Density');
title('Kernel Density Estimation');

Thresh1 = prctile(T2_tr, 95)

CumulDensty = cumtrapz(f, xi);
T2knbeta = interp1(CumulDensty/max(CumulDensty),f, 0.95)
% TS = T2knbeta * ones(size(princ,1),1);






%% 2A. t-SNE FOR NORMALIZED NO-FAULT TEST DATASET
%%================================================%%

SY1 = scale1(Dtest,xm,Sdm); %% Normalizing Dtest using the mean and STDEV of Dtrain


%%% In the file coeff = COEFF and X = SCORE
[COEFF2, X2, LATENT2, TSQUARED2, EXPLAINED2, MU2] = pca(SY1); %% LATENT (i.e. EigValued Matix)

kk=0; %%% Initial count = 0
for i = 1:size(LATENT2,1)                                                        
    alpha2(i)=sum(LATENT2(1:i))/sum(LATENT2);
    if alpha2(i)>=percent
            kk=i;
            break;
    end 
end
SY = X2(:,1:k); %% k or kk?

% NumDimensions=2; %% 
[Y_ts0,loss20] = tsne(SY1,'Algorithm',Algos,'Distance',Dis_tSNE,InitialY = InitialY_val_tr, Perplexity = Perplexity_num,Exaggeration = Exaggeration_num, LearnRate = LearnRate_num);




%% 2B. INTRODUCE FAULT INTO TEST DATASET
%%========================================%%

fprintf('\n\n\n\n\n'); disp('//////// PART C:: FAULT DETAILS ///////'),fprintf('\n\n'),


lim = 1000; %%input('\nEnter the instant of fault, lim =  ');
fprintf('\nFault introduced at %d-th observation in Test dataset.', lim)


IndX = 4; %%% Tank# 4
fprintf('\nTank # Selection = %d\n',IndX);


%%% ==== Uncomment only those (following) lines that need to be executed ====%%%

for xyz = 1


    % %-------------------- 1. Bias Fault -------------------------% %
    FaultID = 'Bias';
    Fvalue = 25;
    FvalueS = num2str(Fvalue);
    Limitz = lim;
    for i = 1:m2
        if(i>Limitz)
            Dtest(i,IndX) = Dtest(i,IndX) + Fvalue;
        end
    end


    % % % -------------------- 2. Drift Fault -------------------------% %
    % FaultID = 'Drift';
    % Fvalue = 0.07; %% The slope of drift fault
    % FvalueS = num2str(Fvalue);
    % r = [];
    % Limitz = lim;
    % for i = 1:(m2-Limitz)
    %     r(i)=i;
    % end
    % 
    % for i=1:m2
    %     if(i>Limitz)
    %         Dtest(i,IndX) = Dtest(i,IndX) + Fvalue*r(i-Limitz);
    %     end
    % end


    % % % --------------- 3. Drift + Prec. Deg. Fault --------------------% %
    % 
    % FaultID = 'Drift+PD ';
    % Dslope = 0.1;
    % Mag_PD = 0.45;
    % Limitz = lim;
    % FvalueS = "Slope = " + num2str(Dslope) + "; Mag-PD = " + num2str(Mag_PD);
    % 
    % r = [];
    % for i = 1:(m2-Limitz)
    %     r(i)=i;
    % end
    % 
    % for i=1:m2
    %     if(i>Limitz)
    %         Dtest(i,IndX) = Dtest(i,IndX) + Dslope*r(i-Limitz)*Mag_PD*rand(1);
    %     end
    % end



    % % % ------------------ 4. Freeze Fault --------------------% %
    % 
    % FaultID = 'Freeze';
    % m2 = size(Dtest,1); 
    % % lim = size(Dtest1,1)/2; 
    % 
    % idx = round(randi(numel(Dtest(lim+1:end, IndX)))/4); %%  Select a random cell index, idx, corr. to MFault_ID vector
    % 
    % idx,
    % 
    % Limitz = lim+idx;
    % 
    % A = Dtest(Limitz, IndX); %% Select that cell value corr. to idx
    % 
    % FvalueS = "at n = " + num2str(Limitz);
    % for i =1:m2
    %     if(i>Limitz)
    %         Dtest(i,IndX) = 3*A; %% replace remaining cells as 'A' %%% Bias of 2-5
    %     end
    % end


    % % % ------------------ 5. Intermittent Fault --------------------% %
    % 
    % FaultID = 'Intermittent';
    % a= lim + 25;
    % b= lim + 125;
    % c= m2 - 250;
    % d= m2 - 150;
    % FvalueS = "b/w n= (" + num2str(a) + ", " + num2str(b) + ") and (" + num2str(c) + ", " + num2str(d) + ")";
    % 
    % for i=1:m2
    %     if (((i > a) && (i < b)) || ((i > c) && (i < d))) %||((i>1050) && (i<1150)))
    %         Dtest(i,IndX) = Dtest(i,IndX) + 10;
    %     end
    % end

end



%% 2B. NORMALIZATION OF FAULTY TEST DATASET
%%===========================================%%

fprintf('\n\n\n'); disp('//////// PART D::  ///////'), 
fprintf('\n')

SY1 = scale1(Dtest,xm,Sdm); %% Normalizing Dtest using the mean and STDEV of Dtrain

CV_test =cov(SY1);
XT1 = SY1*COEFF;
Xp = SY1*COEFF(:,1:k)*COEFF(:,1:k)';   % Xp is the estimation of original data using COEFF of training PCA model, X = Xp + E






%% 2C. t-SNE FOR NORMALIZED FAULTY TEST DATASET
%===============================================%%

% NumDimensions=2; %% 
[Y_ts,loss2] = tsne(SY1,'Algorithm',Algos,'Distance',Dis_tSNE,InitialY = InitialY_val_tr, Perplexity = Perplexity_num,Exaggeration = Exaggeration_num, LearnRate = LearnRate_num); %% euclidean or mahalanobis


%% 2D. DETERMINE THE T^2 STATISTICS USING LOW-DIM. SPACE (FROM STEP # 2C.)
%%=========================================================================%%

%%%--- Either ---%%%
YA_ts = SY*A;

T2_ts = [];

for i = 1:size(SY,1)
    T2_ts(i) = Y_ts(i,:)*inv(YA_ts'*YA_ts/(size(SY,1)-1))*Y_ts(i,:)';
end


f2 = figure(2);
    set(gcf, 'WindowState', 'maximized');

    % plot(TS,'k--', 'LineWidth',3); 
    % hold on;
    plot(T2_ts,'b', 'LineWidth',1.2)
    
    box_col = [0 1 0]; %%% Default = [0.4902 0.4902 0.4902]
    xregion(1,Limitz,"FaceColor",box_col,"FaceAlpha",0.1);
    box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
    xregion(Limitz,size(T2_ts',1),"FaceColor",box_col,"FaceAlpha",0.1)

    F_Inst = "Fault Instance at n = " + num2str(Limitz);
    xl = xline(Limitz,'--m',F_Inst,'LineWidth',3);
    xl.LabelHorizontalAlignment = 'right';
    xl.LabelVerticalAlignment = 'middle';
    xl.Color = [00 0 0.40];
    Thrsh = "Threshold = " + num2str(T2knbeta);
    yl = yline(T2knbeta,'-.',Thrsh, 'LineWidth',3);
    yl.LabelHorizontalAlignment = 'left';
    yl.LabelVerticalAlignment = 'top';
    yl.Color = [0.80 0 0.40];
    xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
    xlim([0 size(Dtest,1)]);
    % % legend({'T_{thres}^2','T_{test}^2'},'Location', 'best'),
    legend({'T_{test}^2'},'Location', 'best'),
    
    grid


f3 = figure(3); 
    set(gcf, 'WindowState', 'maximized'); 
    subplot(211), plot(Y_tr), hold on, title('Y_{tr}'), xlabel('Instance, n','fontweight','bold'), grid, 
    subplot(212), plot(Y_ts), hold on, xline(Limitz,'--k',{'Fault Instance'}), title('Y_{ts}'), xlabel('Instance, n','fontweight','bold'), grid on,
    box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
    xregion(Limitz,size(Y_ts,1),"FaceColor",box_col,"FaceAlpha",0.1)



f4 = figure(4); 
    set(gcf, 'WindowState', 'maximized'); 
    subplot(211), plot(Y_tr(:,1),'k', 'LineWidth',1.5), hold on, plot(Y_ts(:,1),'r', 'LineWidth',1.5), title('Y_1 = Y(:,1)'), xlabel('Instance, n','fontweight','bold'), grid,
    F_Inst = "Fault Instance " ;
    xl = xline(Limitz,'--m',F_Inst,'LineWidth',3);
    xl.LabelHorizontalAlignment = 'right';
    xl.LabelVerticalAlignment = 'top';
    xl.LabelOrientation = 'horizontal';
    xl.Color = [0.4660 0.6740 0.1880];
    legend({'Y_{tr}(:,1)','Y_{ts}(:,1)'},'Location','best')
    
    subplot(212), plot(Y_tr(:,2),'k', 'LineWidth',1.5), hold on, plot(Y_ts(:,2),'r', 'LineWidth',1.5), title('Y_2 = Y(:,2)'), xlabel('Instance, n','fontweight','bold'), grid,
    F_Inst = "Fault Instance " ;
    xl = xline(Limitz,'--m',F_Inst,'LineWidth',3);
    xl.LabelHorizontalAlignment = 'right';
    xl.LabelVerticalAlignment = 'top';
    xl.LabelOrientation = 'horizontal';
    xl.Color = [0.4660 0.6740 0.1880];
    legend({'Y_{tr}(:,2)','Y_{ts}(:,2)'},'Location','best')
    % sgtitle('Note: Y = [Y_1  Y_2]','fontsize',14,'FontName', 'Arial','fontweight','bold') %% Global Title






