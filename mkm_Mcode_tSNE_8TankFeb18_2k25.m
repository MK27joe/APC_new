
close all; clc; clear; 

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

Algos = 'exact'; 

Dis_tSNE = 'mahalanobis';

NumDimensions = 2; 

Perplexity_num = 40;

Exaggeration_num = 4;

LearnRate_num = 100;

InitialY_val_tr = 1e-4*randn(m1,NumDimensions);
InitialY_val_ts = 1e-4*randn(m2,NumDimensions);


%% 1A. NORMALIZATION OF TRAINING DATASET
%%=======================================%%

xm = mean(Dtrain);
Sdm = std(Dtrain);

Xbar = (Dtrain - xm(ones(m1,1),:)) ./ (Sdm(ones(m1,1),:));

CV =cov(Xbar);
[U,S,V] = svd(CV); 



%% 1C. t-SNE MODEL TRAINING DATASET 
% %===================================%%
% % R^(m1 x n1) to R^(m1 x 2) space 

[Y_tr,loss1] = tsne(Xbar,'Algorithm',Algos,'Distance',Dis_tSNE,InitialY = InitialY_val_tr, Perplexity = Perplexity_num,Exaggeration = Exaggeration_num, LearnRate = LearnRate_num);


A = inv(Xbar' * Xbar) * Xbar' * Y_tr;
A,

YA_tr = Xbar*A;

T2_tr = [];
for i = 1:size(Xbar,1)
    T2_tr(i) = YA_tr(i,:) * inv(Y_tr' * Y_tr/(size(Xbar,1)-1)) * YA_tr(i,:)';
end

%%%%% [aera,pnt] = icalimit(T2_tr,1.49); 
%%%%% lim = Area limit which is between 1.4865 && 1.4950
%%%%% A = Statistical indicies calculated from the data operating in normal fault free conditions
%%%%% Aera = Area of the statistical index using kernel density estimation
%%%%% Pnt = Threshold point occupying 99% of the area = CONTROL STATISTICS LIMIT (?)
%%%%% T2knbeta = pnt;
%%%%% TS = pnt*ones(size(princ,1),1);

%%%%% [f,xi] = ksdensity(T2_tr); %% Kernel smoothing function estimate for univariate and bivariate data
%%%%% [f,xi] = ksdensity(x) returns a probability density estimate, f, for the sample data in the vector or two-column matrix x. ...
%%%%% The estimate is based on a normal kernel function,...
%%%%% and is evaluated at equally-spaced points, xi, that cover the range of the data in x. ...
%%%%% ksdensity estimates the density at 100 points for univariate data, or 900 points for bivariate data.

%%%%% [f,xi] = kde(T2_tr); %% Kernel density estimate for univariate data
%%%%% [f,xi]=icalimit(T2_tr, 1.485)
[f, xi] = kdeG(T2_tr, 0.3, 5000); %%% kdeG(data, bandwidth, nPoints);; f: estimated density values;; xi: points at which the density is estimated



%%%%% [f,xf] = kde(a) estimates a probability density function (pdf) for the univariate data in the vector a and ...
%%%%% returns values f of the estimated pdf at the evaluation points xf. ...
%%%%% kde uses kernel density estimation to estimate the pdf. See Kernel Distribution for more information.
figure(1)
plot(xi,f, 'LineWidth',2), grid on, 
xlabel('Data Values');
ylabel('Density');
title('Kernel Density Estimation for 2D-DTrain');

CumulDensty = cumtrapz(f, xi);
T2knbeta = interp1(CumulDensty/max(CumulDensty),f, 0.95)
% TS = T2knbeta * ones(size(princ,1),1);

%%%%% Thresh1 = prctile(T2_tr, 95)
%%%%% TS = T2knbeta * ones(size(princ,1),1);

%%%%%------ or... ------%%%

%%%%% T2knbeta = (k*(size(Y_tr,1)^2-1)/(size(Y_tr,1)*(size(Y_tr,1)-k))) * finv(0.95,k,size(Y_tr,1)-k);
%%%%% fprintf('\n(**) T2knbeta = %0.4f\n', T2knbeta)
%%%%% TS = T2knbeta*ones(size(Y_tr,1),1);




%% 2B. INTRODUCE FAULT INTO TEST DATASET
%%========================================%%

fprintf('\n\n\n\n\n'); disp('//////// PART C:: FAULT DETAILS ///////'),fprintf('\n\n'),


lim = 1000; %%input('\nEnter the instant of fault, lim =  ');
fprintf('\nFault introduced at %d-th observation in Test dataset.', lim)


IndX = 4; %%% Tank# 4
fprintf('\nTank # Selection = %d\n',IndX);


%%% ==== Uncomment only those (following) lines that need to be executed ====%%%

for xyz = 1


    % % %-------------------- 1. Bias Fault -------------------------% %
    % FaultID = 'Bias';
    % Fvalue = 25;
    % FvalueS = num2str(Fvalue);
    % Limitz = lim;
    % for i = 1:m2
    %     if(i>Limitz)
    %         Dtest(i,IndX) = Dtest(i,IndX) + Fvalue;
    %     end
    % end


    % % -------------------- 2. Drift Fault -------------------------% %
    FaultID = 'Drift';
    Fvalue = 0.1; %% The slope of drift fault // 0.07
    FvalueS = num2str(Fvalue);
    r = [];
    Limitz = lim;
    for i = 1:(m2-Limitz)
        r(i)=i;
    end

    for i=1:m2
        if(i>Limitz)
            Dtest(i,IndX) = Dtest(i,IndX) + Fvalue*r(i-Limitz);
        end
    end


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
    % A11 = Dtest(Limitz, IndX); %% Select that cell value corr. to idx
    % 
    % FvalueS = "at n = " + num2str(Limitz);
    % for i =1:m2
    %     if(i>Limitz)
    %         Dtest(i,IndX) = 3*A11; %% replace remaining cells as 'A11' %%% Bias of 2-5
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



%% 2B. t-SNE MODEL FROM NORMALIZED FAULTY TEST DATASET
%%=====================================================%%

SY1 = scale1(Dtest,xm,Sdm); %% Normalizing Dtest using the mean and STDEV of Dtrain

[Y_ts,loss2] = tsne(SY1,'Algorithm',Algos,'Distance',Dis_tSNE,InitialY = InitialY_val_tr, Perplexity = Perplexity_num,Exaggeration = Exaggeration_num, LearnRate = LearnRate_num); %% euclidean or mahalanobis


%% 2D. DETERMINE THE T^2 STATISTICS USING LOW-DIM. SPACE (FROM STEP # 2C.)
%%=========================================================================%%

%%%--- Either ---%%%
YA_ts = SY1*A;

T2_ts = [];

for i = 1:size(SY1,1)
    T2_ts(i) = YA_ts(i,:) * inv(Y_ts' * Y_ts/(size(SY1,1)-1)) * YA_ts(i,:)';
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


% f3 = figure(3); 
%     set(gcf, 'WindowState', 'maximized'); 
%     subplot(211), plot(Y_tr), hold on, title('Y_{tr}'), xlabel('Instance, n','fontweight','bold'), grid, 
%     subplot(212), plot(Y_ts), hold on, xline(Limitz,'--k',{'Fault Instance'}), title('Y_{ts}'), xlabel('Instance, n','fontweight','bold'), grid on,
%     box_col = [1 0 0]; %%% Default = [0.4902 0.4902 0.4902]
%     xregion(Limitz,size(Y_ts,1),"FaceColor",box_col,"FaceAlpha",0.1)
