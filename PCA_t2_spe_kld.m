%% DISCLAIMER: This code-file is NOT developed for user-interactive mode

%%
close all; clc; clear; 


%% Load the datasheet
load octuple_tank_data_11_11_24.mat

%% BIFURCATE YOUR DATABASE INTO TRAINING AND TESTING SETS
%%=======================================================%%

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

%% NORMALIZATION OF TRAINING DATASET
%%==================================%%

xm = mean(Dtrain);
Sdm = std(Dtrain);

Xbar = (Dtrain - xm(ones(m1,1),:)) ./ (Sdm(ones(m1,1),:));

CV =cov(Xbar);
[U,S,V] = svd(CV); 


%% PCA PARAMETERS USING TRAINING DATASET
%%======================================%%

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(Xbar); %% LATENT (i.e. EigValues)


%% ANALYSIS OF PCS FROM PCA METHOD
%%================================%%


% prompt = input('\n\nEnter the percentage of significance (b/w 80 to 95 %) = '); %%% self explanatory !!
prompt = 95;%Default 98pc

%percent = input(prompt);
percent = prompt/100;

k=0; %%% Initial count = 0

for i = 1:size(LATENT,1)                                                        
    alpha(i)=sum(LATENT(1:i))/sum(LATENT);
    if alpha(i)>=percent
            k=i;
            break;
    end 
end

%---- Find the Residue Space (Training Dataset) %%% for KLD

Xp1 = Xbar*COEFF(:,1:k)*COEFF(:,1:k)';   % Xp is the estimation of original data using the PCA model, X=Xp+E
e1 = Xbar - Xp1; %% Residual Space @ No-fault scenario


%%

% %---------- Display the details of PCs -------------------%
 
for ij = 1
    fprintf('\n\n\n=================================\n')
    fprintf('|| Octuple-Tank system: Test Results ||\n')
    fprintf('=================================\n\n\n')

    disp('//////// PART A:: SYSTEM INFO ///////'), 
    
    fprintf('\n** Total Number of Observations (Database) = %d \n', M)
    fprintf('\n** Total Number of Observations (Training) = %d \n', m1)
    fprintf('\n** Total Number of Observations (Testing) = %d \n\n', m2)


    % disp('~~~~~~~~~~~~~~~')
    % fprintf('Obervations\n')
    % disp('~~~~~~~~~~~~~~~')

    fprintf('\n\n\n'); disp('//////// PART B:: PCs INFO ///////'), 
    fprintf('\n')

    fprintf('\n==> The percentage of significance set for PC contribution = %0.4f', prompt)

    fprintf('\n\n(1) No. of PCs chosen = %d out of 14 I/O vectors.\n',k)

    fprintf('\n(2) Cumul. PC contrib: \n   ********************\n '), alpha,         

    TotPCcontr = alpha(1,end) * 100;
    fprintf('## Total PC contribution computed = %0.4f\n', TotPCcontr)

end


%%
princ = SCORE(:,1:k); %%% upto first kth PCs
per = LATENT/sum(LATENT);

nnn = [1:length(EXPLAINED)];
CExpVar = zeros(1,length(nnn));
for xyz = 1:length(nnn)
    CExpVar(xyz+1) = CExpVar(xyz) + EXPLAINED(xyz);
end
CExpVar(1) = [];


%% Display E. values, CoVar Matrix and CPV


for lp = 1
    f1 = figure(1); % Plot the faulty test-data vector
    set(f1,'Position',get(0,'screensize'));
    cdata=CV;
    xvalues={'h1','h2','h3','h4','h5','h6','h7','h8','v1','v2','v3','v4','d1','d2'};
    yvalues={'h1','h2','h3','h4','h5','h6','h7','h8','v1','v2','v3','v4','d1','d2'};

    h=heatmap(xvalues,yvalues,cdata);
    colormap(parula)
    h.Title = 'Heatmap of the Co-variance (correlation) matrix of Training dataset Dtest';

    % %%%%% or,
    % VariableNames = {'h1','h2','h3','h4','h5','h6','h7','h8','v1','v2','v3','v4','d1','d2'};
    % h=corrplot(cdata,'varNames', VariableNames);



f2 = figure(2);
    set(f2,'Position',get(0,'screensize'));
    subplot(311)
    set(gcf, 'WindowState', 'maximized');
    plot(nnn,EXPLAINED,'Color',[0,0,0],'Marker','o','LineStyle', '-','LineWidth',2),grid; 
    title(['Cumulative Percentage Variance (CPV) Technique to find the required number of PCs (at a significance of ', num2str(prompt) ,'%).'])
    subtitle('(a)')
    ylabel('Eigen Value (i.e. Variance)','fontsize',14,'FontName', 'Arial');
    %%% xlim([1 length(nnn)]);

    subplot(312)
    set(gcf, 'WindowState', 'maximized');
    bar(EXPLAINED,'LineWidth',1.2); 
    subtitle('(b)')
    ylabel('Explained Var. (%)','fontsize',14,'FontName', 'Arial'), grid;

    subplot(313)
    set(gcf, 'WindowState', 'maximized');
    bar(CExpVar,'LineWidth',1.2); 
    subtitle('(c)')
    ylabel('Cumul. Explained Var. (%)','fontsize',14,'FontName', 'Arial'),grid;
    ylim([0,100]);

    xlabel('Principal Components','fontsize',12,'FontName', 'Arial');
end


%% INTRODUCE FAULTS: BIAS/DRIFT/PREC. DEGRADATION/FREEZING/INTERMITTENT
%%=====================================================================%%

fprintf('\n\n\n\n\n'); disp('//////// PART C:: FAULT DETAILS ///////'),fprintf('\n\n'),


lim = 1000; %%input('\nEnter the instant of fault, lim =  ');
fprintf('\nFault introduced at %d-th observation in Test dataset.', lim)


IndX = 4; %%% Tank# 4
fprintf('\nTank # Selection = %d\n',IndX);



%%% ==== Uncomment only those lines for faults that need to be executed ====%%%

for xyz = 1

    % % -------------------- 1. Bias Fault -------------------------% %
    FaultID = 'Bias';
    Bias_value = 10
    IntrmF = [0,0,0,0,0];

    for i = 1:m2
        if(i>lim)
            Dtest(i,IndX) = Dtest(i,IndX) + Bias_value;
        end
    end
    

  


    % % % -------------------- 2. Drift Fault -------------------------% %
    % FaultID = 'Drift';
    % Dslope = 0.07
    % IntrmF = [0,0,0,0,0];
    % r = [];
    % for i = 1:(m2-lim)
    %     r(i)=i;
    % end
    % 
    % for i=1:m2
    %     if(i>lim)
    %         Dtest(i,IndX) = Dtest(i,IndX) + Dslope*r(i-lim);
    %     end
    % end
    
   

    % % % 
    % % % --------------- 3. Drift + Prec. Deg. Fault --------------------% %
    % 
    % FaultID = 'Drift+PD';
    % IntrmF = [0,0,0,0,0];
    % Dslope = 0.07
    % Mag_PD = 0.45
    % r = [];
    % for i = 1:(m2-lim)
    %     r(i)=i;
    % end
    % 
    % for i=1:m2
    %     if(i>lim)
    %         Dtest(i,IndX) = Dtest(i,IndX) + Dslope*r(i-lim)*Mag_PD*rand(1);
    %     end
    % end
    



    % % % ------------------ 4. Freeze Fault --------------------% %
    % 
    % FaultID = 'Freeze';
    % IntrmF = [0,0,0,0,0];
    % m2 = size(Dtest,1); 
    % % lim = size(Dtest1,1)/2; 
    % 
    % idx = round(randi(numel(Dtest(lim+1:end, IndX)))/4); %%  Select a random cell index, idx, corr. to MFault_ID vector
    % Nidx = lim + idx
    % A = Dtest(Nidx, IndX); %% Select that cell value corr. to idx
    % 
    % for i =1:m2
    %     if(i>Nidx)
    %         Dtest(i,IndX) = 2*A; %% replace remaining cells as 'A' %%% Bias of 2
    %     end
    % end
    % lim = Nidx;
    % 
    


    % % % ------------------ 5. Intermittent Fault --------------------% %
    % 
    % FaultID = 'Intermittent';
    % a = lim + 25
    % b = lim + 125
    % c = m2 - 250
    % d = m2 - 150
    % 
    % for i=1:m2
    %     if (((i > a) && (i < b)) || ((i > c) && (i < d))) %%%||((i>1050) && (i<1150)))
    %         Dtest(i,IndX) = Dtest(i,IndX) + 10;
    %     end
    % end
    % IntrmF = [1,a,b,c,d];  

end

%% NORMALIZATION OF TESTING DATASET
%%=================================%%

fprintf('\n\n\n'); disp('//////// PART D:: PCA MODEL-TEST INFO ///////'), 
fprintf('\n')


SY = scale1(Dtest,xm,Sdm); %% Normalizing Dtest using the mean and STDEV of Dtrain
CV_test = cov(SY);

XT1 = SY*COEFF;

Xp2 = SY*COEFF(:,1:k)*COEFF(:,1:k)';   % Xp is the estimation of original data using the PCA model, X=Xp+E


%---- Find the Residue Space (Testing Dataset)
e2 = SY - Xp2; %% Residual Space @ Actual scenario




%% COMPUTE T^2 STATISTICS 
%%=======================%%



%%--------------------- 1. T^2 threshold computation ------------------%

T2knbeta = (k*(size(Xbar,1)^2-1)/(size(Xbar,1)*(size(Xbar,1)-k))) * finv(0.95,k,size(Xbar,1)-k);
fprintf('\n(5) T2knbeta = %0.4f\n', T2knbeta)
TS = T2knbeta*ones(size(SY,1),1);



%%--------------------- 2. T^2 statistic computation ------------------%

for i=1:size(SY,1)
    ts1(i) = XT1(i,1:k) * diag( 1 ./ LATENT(1:k) ) * XT1(i,1:k)';
end



%%--------------------- 3. T^2 Confusion Matrix  ------------------%

fprintf('\n\n$$$$$$$$$$$$$$$\n')
disp('A. Using T^2: ')
fprintf('\n$$$$$$$$$$$$$$$\n\n')


% [FP, TN, FN, TP] = FaultRatios(m2, lim, ts1, T2knbeta)
%[FP, TN, FN, TP] = ConFusion(m2, lim, ts1, T2knbeta, aa,bb,cc,dd)

Win_size = 0;
ValsIn = [m2, lim, T2knbeta, Win_size];
[TT] = ConFusion2(ts1, IntrmF, ValsIn)

% fprintf('\n[*] Precision, or Positive Predictive Value (PPV) (in pc) = %0.4f\n', (TP*100 / (TP + FP)))
% fprintf('\n[*] Recall, or Sensitivity, or True Positive Rate (TPR) (in pc) = %0.4f\n', (TP*100 / (TP + FN)))
% fprintf('\n[*] False Alarm Rate (FAR) or False Positive Rate (FPR) (in pc) = %0.4f\n', (FP*100 / (FP + TN))) %% = 1 - TNR
% fprintf('\n[*] F1 Score = %0.4f\n', (2*TP / (2*TP + FP + FN)))
% fprintf('\n[*] Accuracy (in pc) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
% fprintf('\n[*] Matthews Correlation Coefficient = %0.4f\n', (TP*TN - FP*FN) / sqrt((TP+FP)*(TN+FN)*(FP+TN)*(TP+FN)))
% fprintf('\n$$$$$$$$$$$$$$$\n\n')
% 
% % % % % ConfusionM = ["FP = " + num2str(FP), "TN = " + num2str(TN); "FN = " + num2str(FN), "TP = " + num2str(TP)]
% % % % % 
% % % % % fig = uifigure; uit = uitable(fig,"Data",ConfusionM,'FontSize',18)
% 
% % % % format shortG;
% % % % TitleS = ['FP', 'TN', 'FN', 'TP', strcat('Precision',{' '},'(','%',')'), ...
% % % %     strcat('Recall',{' '},'(','%',')'), strcat('FAR',{' '},'(','%',')'), ...
% % % %     strcat('F1',{' '},'Score'),strcat('Accuracy',{' '},'(','%',')'),'MCC']'; 
% % % % format shortG;
% % % % ConfuTable = [TitleS, TT]
% % % % 
% % % % strcat('Precision',{' '},'(','%',')')
% 


%% COMPUTE SPE/Q STATISTICS AND THRESHOLD
%%=======================================%%



%%--------------------- 1. SPE/Q threshold computation ----------------%


 beta = 0.95;      % takes values between 90% and 95%
 theta = zeros(3,1);
 for ii = 1:3
     for j = k+1:size(SY,2)
         theta(ii) = theta(ii) + LATENT(j)^(ii);
     end
 end
h0 = 1-((2*theta(1)*theta(3))/(3*theta(2)^2));
ca = norminv(0.95, 0, 1);% ca is value for standard normal distribution for confidence level 95%
SPEbeta = theta(1)*((ca*h0*sqrt(2*theta(2))/theta(1))+1+(theta(2)*h0*(h0-1)/theta(1)^2))^(1/h0);
fprintf('\n(4) SPE-Q Beta = %0.4f\n', SPEbeta)
S1 = SPEbeta*ones(size(SY,1),1);


%%--------------------- 2. SPE/Q statistic computation ----------------%

for i = 1:size(SY,1)
    SPE(i) = sum((SY(i,:) - Xp2(i,:)).^2); % Sum of Predicted Errors
end


%%--------------------- 3. SPE/Q Confusion Matrix   ------------------%

fprintf('\n$$$$$$$$$$$$$$$\n')
disp('B. Using SPE\Q: ')
fprintf('\n$$$$$$$$$$$$$$$\n\n')

% [FP, TN, FN, TP] = FaultRatios(m2, lim, SPE, SPEbeta)
%[FP, TN, FN, TP] = ConFusion(m2, lim, SPE, SPEbeta, aa,bb,cc,dd)
Win_size = 0;
ValsIn = [m2, lim, SPEbeta, Win_size];
[TT] = ConFusion2(SPE, IntrmF, ValsIn)

% 
% fprintf('\n[*] Precision, or Positive Predictive Value (PPV) (in pc) = %0.4f\n', (TP*100 / (TP + FP)))
% fprintf('\n[*] Recall, or Sensitivity, or True Positive Rate (TPR) (in pc) = %0.4f\n', (TP*100 / (TP + FN)))
% fprintf('\n[*] False Alarm Rate (FAR) or False Positive Rate (FPR) (in pc) = %0.4f\n', (FP*100 / (FP + TN))) %% = 1 - TNR
% fprintf('\n[*] F1 Score = %0.4f\n', (2*TP / (2*TP + FP + FN)))
% fprintf('\n[*] Accuracy (in pc) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
% fprintf('\n[*] Matthews Correlation Coefficient = %0.4f\n', (TP*TN - FP*FN) / sqrt((TP+FP)*(TN+FN)*(FP+TN)*(TP+FN)))
% fprintf('\n$$$$$$$$$$$$$$$\n\n')
% 




%%  COMPUTE KLD STATISTICS AND THRESHOLD
%%=======================================%%


%%--------------------- 1. KLD threshold, h ----------------%

%----- Convert 1000x14 to 1000x1 vector
NormMat_NF= [];
for ii = 1:size(e1,1)
    NormMat_NF(ii,1) = norm(e1(ii,:));
end


fprintf('\n$$$$$$$$$$$$$$$\n')
disp('C. Using KLD: ')
fprintf('\n$$$$$$$$$$$$$$$\n\n')

mu0 = mean(NormMat_NF)
var0 = var(NormMat_NF)
h = mu0 + 3*sqrt(var0)
hLine = h*ones(size(e1,1),1);


%%--------------------- 2. KLD statistic, J  ----------------%

%----- Convert 1000x14 to 1000x1 vector
NormMat_F= [];
for ii = 1:size(e2,1)
    NormMat_F(ii,1) = norm(e2(ii,:));
end

Win_size = 70; %% Moving window size, N = 50 (say)
SubDatSize = size(NormMat_F,1)/Win_size;

J = zeros(1, length(NormMat_F) - Win_size);

for i = 1:length(NormMat_F) - Win_size
    J(i) = (mean(NormMat_F(i:(i+Win_size-1))) - mu0).^2 / var0;
end
J=J';
dum = zeros(Win_size,1);
Jt = [dum;J];

%%--------------------- 3. KLD Confusion Matrix   ------------------%


% [FP, TN, FN, TP] = FaultRatios(m2, lim, Jt, h)
%[FP, TN, FN, TP] = ConFusion(m2, lim, Jt, h, aa,bb,cc,dd)

ValsIn = [m2, lim, h, Win_size];
[TT] = ConFusion2(Jt, IntrmF, ValsIn)

% % fprintf('\n[*] Precision, or Positive Predictive Value (PPV) (in pc) = %0.4f\n', (TP*100 / (TP + FP)))
% % fprintf('\n[*] Recall, or Sensitivity, or True Positive Rate (TPR) (in pc) = %0.4f\n', (TP*100 / (TP + FN)))
% % fprintf('\n[*] False Alarm Rate (FAR) or False Positive Rate (FPR) (in pc) = %0.4f\n', (FP*100 / (FP + TN))) %% = 1 - TNR
% % fprintf('\n[*] F1 Score = %0.4f\n', (2*TP / (2*TP + FP + FN)))
% % fprintf('\n[*] Accuracy (in pc) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
% % fprintf('\n[*] Matthews Correlation Coefficient = %0.4f\n', (TP*TN - FP*FN) / sqrt((TP+FP)*(TN+FN)*(FP+TN)*(TP+FN)))
% % fprintf('\n$$$$$$$$$$$$$$$\n\n')



%% Statistics Comparison
TTTT = res(1:length(Dtest),1)';

f3 = figure(3);
    set(gcf, 'WindowState', 'maximized');

    subplot(3,1,1)

    % plot(TTTT(1,1:lim), ts1(1,1:lim), 'g','LineWidth',2); hold on;
    % plot(TTTT(1,lim:end), ts1(1,lim:end), 'r','LineWidth',2);
    plot(TTTT, ts1, 'b','LineWidth',2);

    ylabel('PCA-T^2','fontsize',14,'FontName', 'Arial');
    hold on;
    plot(TS,'r--', 'LineWidth',3); 
    xlabel('Observation Number, n','fontsize',12,'FontName', 'Arial');
    xlim([0 size(Dtest,1)]);
    legend({'T_{stat}^2','T_{Thres}^2'},'Location', 'best'),
    grid


    subplot(3,1,2)

    % plot(TTTT(1,1:lim), SPE(1,1:lim), 'g','LineWidth',2); hold on;
    % plot(TTTT(1,lim:end), SPE(1,lim:end), 'r','LineWidth',2);
    plot(TTTT, SPE, 'b','LineWidth',2);

    ylabel('PCA-SPE-Q','fontsize',14,'FontName', 'Arial');
    hold on;
    plot(S1,'r--', 'LineWidth',3);  
    xlabel('Observation Number, n','fontsize',12,'FontName', 'Arial');
    legend({'SPE-Q_{stat}','Q-Thres.'},'Location', 'best')
    grid;

    subplot(3,1,3)

    % plot(TTTT(1,1:lim), Jt(1:lim,1), 'g','LineWidth',2); grid,hold on;
    % plot(TTTT(1,lim:end), Jt(lim:end,1), 'r','LineWidth',2),grid on;
    plot(TTTT, Jt, 'b','LineWidth',2),grid on;
    hold on
    plot(hLine,'r--','LineWidth',2); grid on,
    xlabel('Observation Number, n','fontsize',12,'FontName', 'Arial'),
    ylabel('PCA-KLD','fontsize',14,'FontName', 'Arial');
    legend({'KLD_{Stat}','KLD_{Thresh}'},'Location', 'best'),




