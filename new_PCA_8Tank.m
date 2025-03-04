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

CV = cov(Xbar);
[U,S,V] = svd(CV); 

pp = n1;
Vr = V(:,1:pp);
zz =  Vr'*Dtrain';
zzT = zz';
Xhat = (Vr*zz)';

%% PCA PARAMETERS USING TRAINING DATASET
%%======================================%%

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(Xbar); %% LATENT (i.e. EigValues)


%% ANALYSIS OF PCS FROM PCA METHOD
%%================================%%


% prompt = input('\n\nEnter the percentage of significance (b/w 80 to 95 %) = '); %%% self explanatory !!
prompt = 95;

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
% 
% % %---------- Display the details of PCs -------------------%
% % 
% % for ij = 1
% %     fprintf('\n\n\n=================================\n')
% %     fprintf('|| Octuple-Tank system: Test Results ||\n')
% %     fprintf('=================================\n\n\n')
% % 
% %     disp('//////// PART A:: SYSTEM INFO ///////'), 
% % 
% %     fprintf('\n** Total Number of Observations (Database) = %d \n', M)
% %     fprintf('\n** Total Number of Observations (Training) = %d \n', m1)
% %     fprintf('\n** Total Number of Observations (Testing) = %d \n\n', m2)
% % 
% % 
% %     % disp('~~~~~~~~~~~~~~~')
% %     % fprintf('Obervations\n')
% %     % disp('~~~~~~~~~~~~~~~')
% % 
% %     fprintf('\n\n\n'); disp('//////// PART B:: PCs INFO ///////'), 
% %     fprintf('\n')
% % 
% %     fprintf('\n==> The percentage of significance set for PC contribution = %0.4f', prompt)
% % 
% %     fprintf('\n\n(1) No. of PCs chosen = %d out of 14 I/O vectors.\n',k)
% % 
% %     fprintf('\n(2) Cumul. PC contrib: \n   ********************\n '), alpha,         
% % 
% %     TotPCcontr = alpha(1,end) * 100;
% %     fprintf('## Total PC contribution computed = %0.4f\n', TotPCcontr)
% % 
% % end
% % 
princ = SCORE(:,1:k); %%% upto first kth PCs
per = LATENT/sum(LATENT);

nnn = [1:length(EXPLAINED)];
CExpVar = zeros(1,length(nnn));
for xyz = 1:length(nnn)
    CExpVar(xyz+1) = CExpVar(xyz) + EXPLAINED(xyz);
end
CExpVar(1) = [];
% % 
% % f1 = figure(1); % Plot the faulty test-data vector
% %     set(f1,'Position',get(0,'screensize'));
% %     cdata=CV;
% %     xvalues={'h1','h2','h3','h4','h5','h6','h7','h8','v1','v2','v3','v4','d1','d2'};
% %     yvalues={'h1','h2','h3','h4','h5','h6','h7','h8','v1','v2','v3','v4','d1','d2'};
% %     h=heatmap(xvalues,yvalues,cdata);
% %     colormap(parula)
% %     h.Title = 'Heatmap of the Co-variance (correlation) matrix of Training dataset Dtest';
% % 
% % 
% % f2 = figure(2);
% %     set(f2,'Position',get(0,'screensize'));
% %     subplot(311)
% %     set(gcf, 'WindowState', 'maximized');
% %     plot(nnn,EXPLAINED,'Color',[0,0,0],'Marker','o','LineStyle', '-','LineWidth',2),grid; 
% %     title(['Cumulative Percentage Variance (CPV) Technique to find the required number of PCs (at a significance of ', num2str(prompt) ,'%).'])
% %     subtitle('(a)')
% %     ylabel('Eigen Value (i.e. Variance)','fontsize',14,'FontName', 'Arial');
% %     %%% xlim([1 length(nnn)]);
% % 
% %     subplot(312)
% %     set(gcf, 'WindowState', 'maximized');
% %     bar(EXPLAINED,'LineWidth',1.2); 
% %     subtitle('(b)')
% %     ylabel('Explained Var. (%)','fontsize',14,'FontName', 'Arial'), grid;
% % 
% %     subplot(313)
% %     set(gcf, 'WindowState', 'maximized');
% %     bar(CExpVar,'LineWidth',1.2); 
% %     subtitle('(c)')
% %     ylabel('Cumul. Explained Var. (%)','fontsize',14,'FontName', 'Arial'),grid;
% %     ylim([0,100]);
% % 
% %     xlabel('Principal Components','fontsize',12,'FontName', 'Arial');
% % 
% % 
%% INTRODUCE FAULTS: BIAS/DRIFT/PREC. DEGRADATION/FREEZING/INTERMITTENT
%%=====================================================================%%

% % fprintf('\n\n\n\n\n'); disp('//////// PART C:: FAULT DETAILS ///////'),fprintf('\n\n'),
% % 
% % 
lim = 1000; %%input('\nEnter the instant of fault, lim =  ');
% % fprintf('\nFault introduced at %d-th observation in Test dataset.', lim)
% % 
% % 
IndX = 4; %%% Tank# 4
% % fprintf('\nTank # Selection = %d\n',IndX);
% % 
% % %     %%% FaultID = 'Drift';
% % %      %%% Ageing slope
% % %     Bias = 0; Mag_PD =0; idx=0; A=0;
% % %     Dtest = FDrift(Dtest,lim,IndX,Dslope);
% % % 
% % %     %%% FaultID = 'Bias';
% % %       %% How to introduce 10% of total variations ??
% % %     Dslope = 0; Mag_PD = 0; idx=0; A=0;
% % %     Dtest = FBias(Dtest,lim,IndX,Bias);
% % % 
% % %     %%% FaultID = 'Freeze';
% % %     Bias=0;Dslope =0; Mag_PD =0;
% % %     [Dtest,idx,A] = FFreeze(Dtest,lim,IndX);
% % % 
% % %     %%% FaultID = 'Intermittent';
% % %     Bias=0;Dslope =0; Mag_PD =0; idx=0; A=0; lim=300;
% % %     [Dtest] = FIntermit(Dtest,lim,IndX);
% % % 
% % %     %%% FaultID = 'Precision-Degradation';
% % %     Dslope = 0.07; %% Enter the slope of Drift fault (w/ Precision Degradation)
% % %     Mag_PD = 0.3; %% Enter the level of degradation within the interval (0, 1)
% % %     Bias=0; idx=0; A=0;
% % %     Dtest = FDPD(Dtest,lim,IndX,Dslope,Mag_PD);
% % % 
% % 
% % 
% % 
% % %%% ==== Uncomment only those lines that need to be executed ====%%%
% % 
% % for xyz = 1
% % 
    % % -------------------- 1. Bias Fault -------------------------% %
    FaultID = 'Bias';
    Bias_value = 10
    for i = 1:m2
        if(i>lim)
            Dtest(i,IndX) = Dtest(i,IndX) + Bias_value;
        end
    end
    


% %     % % % -------------------- 2. Drift Fault -------------------------% %
% %     % FaultID = 'Drift';
% %     % Dslope = 0.07;
% %     % r = [];
% %     % for i = 1:(m2-lim)
% %     %     r(i)=i;
% %     % end
% %     % 
% %     % for i=1:m2
% %     %     if(i>lim)
% %     %         Dtest(i,IndX) = Dtest(i,IndX) + Dslope*r(i-lim);
% %     %     end
% %     % end
% % 
% % 
% %     % % % --------------- 3. Drift + Prec. Deg. Fault --------------------% %
% %     % 
% %     % FaultID = 'Drift+PD';
% %     % Dslope = 0.09;
% %     % Mag_PD = 0.45;
% %     % r = [];
% %     % for i = 1:(m2-lim)
% %     %     r(i)=i;
% %     % end
% %     % 
% %     % for i=1:m2
% %     %     if(i>lim)
% %     %         Dtest(i,IndX) = Dtest(i,IndX) + Dslope*r(i-lim)*Mag_PD*rand(1);
% %     %     end
% %     % end
% % 
% % 
% % 
% %     % % % ------------------ 4. Freeze Fault --------------------% %
% %     % 
% %     % FaultID = 'Freeze';
% %     % m2 = size(Dtest,1); 
% %     % % lim = size(Dtest1,1)/2; 
% %     % 
% %     % idx = round(randi(numel(Dtest(lim+1:end, IndX)))/4) %%  Select a random cell index, idx, corr. to MFault_ID vector
% %     % 
% %     % A = Dtest(lim+idx, IndX); %% Select that cell value corr. to idx
% %     % 
% %     % for i =1:m2
% %     %     if(i>lim+idx)
% %     %         Dtest(i,IndX) = 2*A; %% replace remaining cells as 'A' %%% Bias of 2
% %     %     end
% %     % end
% % 
% % 
% %     % % % ------------------ 5. Intermittent Fault --------------------% %
% %     % 
% %     % FaultID = 'Intermittent';
% %     % a= lim + 25;
% %     % b= lim + 125;
% %     % c= m2 - 250;
% %     % d= m2 - 150;
% %     % 
% %     % 
% %     % for i=1:m2
% %     %     if (((i > a) && (i < b)) || ((i > c) && (i < d))) %||((i>1050) && (i<1150)))
% %     %         Dtest(i,IndX) = Dtest(i,IndX) + 10;
% %     %     end
% %     % end
% % 
% % end
% % 
% % %% NORMALIZATION OF TESTING DATASET
% % %%=================================%%
% % 
% % fprintf('\n\n\n'); disp('//////// PART D:: PCA MODEL-TEST INFO ///////'), 
% % fprintf('\n')
% % 
% % 
SY = scale1(Dtest,xm,Sdm); %% Normalizing Dtest using the mean and STDEV of Dtrain
CV_test =cov(SY);

XT1 = SY*COEFF;

Xp = SY*COEFF(:,1:k)*COEFF(:,1:k)';   % Xp is the estimation of original data using the PCA model, X=Xp+E

e = SY - Xp; %% Residual Space
% % 
% % 
% % %% COMPUTE T^2 STATISTICS 
% % %%=======================%%
% % 
% % 
% % 
%%--------------------- 1a. T^2 threshold computation ------------------%

T2knbeta = (k*(size(Xbar,1)^2-1)/(size(Xbar,1)*(size(Xbar,1)-k))) * finv(0.95,k,size(Xbar,1)-k);
fprintf('\n(5) T2knbeta = %0.4f\n', T2knbeta)
TS = T2knbeta*ones(size(Xbar,1),1);
% % 
% % 
% % % %%--------------------- 1b. Mahalanobis Distance (D) computation ------------------%
% % % 
% % % T2_MD = (size(Xbar,2) * (size(Xbar,1)^2-1)/(size(Xbar,1)*(size(Xbar,1) - size(Xbar,2)))) * finv(0.95,k,size(Xbar,1) - size(Xbar,2));
% % % fprintf('\n(5) T2_MD = %0.4f\n', T2_MD)
% % % TS_md = T2_MD*ones(size(SY,1),1);
% % 
% % 
% % %%--------------------- 2. T^2 statistic computation ------------------%
% % 
% % for i=1:size(SY,1)
% %     ts1(i) = XT1(i,1:k) * diag( 1 ./ LATENT(1:k) ) * XT1(i,1:k)';
% % end
% % 
% % % % nft2 = ts1(1,1:lim-1);
% % % % ZZ = zeros(1,lim);
% % % % ft2 = [ZZ,ts1(1,lim:end)];
% % 
% % % %  FINV  : Inverse of the F cumulative distribution function.
% % % %     X=FINV(beta,k,size(X,1)-k) returns the inverse of the F distribution 
% % % %     function with k and size(X,1)-k degrees of freedom, at the values in beta.
% % 
% % 
% % %% COMPUTE SPE/Q STATISTICS AND THRESHOLD
% % %%=======================================%%
% % 
% % 
% % 
% % %%--------------------- 1. SPE/Q threshold computation ----------------%
% % 
% % 
% %  beta = 0.95;      % takes values between 90% and 95%
% %  theta = zeros(3,1);
% %  for ii = 1:3
% %      for j = k+1:size(SY,2)
% %          theta(ii) = theta(ii) + LATENT(j)^(ii);
% %      end
% %  end
% % h0 = 1-((2*theta(1)*theta(3))/(3*theta(2)^2));
% % ca = norminv(0.95, 0, 1);% ca is value for standard normal distribution for confidence level 95%
% % SPEbeta = theta(1)*((ca*h0*sqrt(2*theta(2))/theta(1))+1+(theta(2)*h0*(h0-1)/theta(1)^2))^(1/h0);
% % fprintf('\n(4) SPE-Q Beta = %0.4f\n', SPEbeta)
% % S1 = SPEbeta*ones(size(SY,1),1);
% % 
% % 
% % %%--------------------- 2. SPE/Q statistic computation ----------------%
% % 
% % for i = 1:size(SY,1)
% %     SPE(i) = sum((SY(i,:) - Xp(i,:)).^2); % Sum of Predicted Errors
% % end
% % 
% % 
% % %% FAR and FDR COMPUTATION
% % %%========================%%
% % 
% % %%--------------------- 1. T^2  ------------------%
% % 
% % fprintf('\n\n$$$$$$$$$$$$$$$\n')
% % disp('A. Using T^2: ')
% % fprintf('\n$$$$$$$$$$$$$$$\n\n')
% % 
% % [FP, TN, FN, TP] = FaultRatios(m2, lim, ts1, T2knbeta)
% % 
% % fprintf('\n[*] True Positive Rate (TPR) (in percent) = %0.4f\n', (TP*100/(TP + FN))) %% = 1 - FNR
% % fprintf('\n[*] False Positive Rate (FPR), or False Alarm Rate (FAR) (in percent) = %0.4f\n', (FP*100/(FP + TN))) %% = 1 - TNR
% % fprintf('\n[*] False Discovery Rate (FDR) (in percent) = %0.4f\n', (FP*100/(FP + TP))) %% = 1 - PPV
% % fprintf('\n[*] Accuracy (in percent) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
% % fprintf('\n$$$$$$$$$$$$$$$\n\n')
% % 
% % % %%------ 2. Mahalanobis Distance (Global T^2)  -------%
% % % 
% % % fprintf('\n\n$$$$$$$$$$$$$$$\n')
% % % disp('B. Using Mahalanobis Distance: ')
% % % fprintf('\n$$$$$$$$$$$$$$$\n\n')
% % % 
% % % [FP, TN, FN, TP] = FaultRatios(m2, lim, ts1, T2_MD)
% % % 
% % % fprintf('\n[*] True Positive Rate (TPR) (in percent) = %0.4f\n', (TP*100/(TP + FN))) %% = 1 - FNR
% % % fprintf('\n[*] False Positive Rate (FPR), or False Alarm Rate (FAR) (in percent) = %0.4f\n', (FP*100/(FP + TN))) %% = 1 - TNR
% % % fprintf('\n[*] False Discovery Rate (FDR) (in percent) = %0.4f\n', (FP*100/(FP + TP))) %% = 1 - PPV
% % % fprintf('\n[*] Accuracy (in percent) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
% % % fprintf('\n$$$$$$$$$$$$$$$\n\n')
% % 
% % 
% % %%--------------------- 3. SPE-Q  ------------------%
% % 
% % fprintf('\n$$$$$$$$$$$$$$$\n')
% % disp('C. Using SPE\Q: ')
% % fprintf('\n$$$$$$$$$$$$$$$\n\n')
% % 
% % [FP, TN, FN, TP] = FaultRatios(m2, lim, SPE, SPEbeta)
% % 
% % fprintf('\n[*] True Positive Rate (TPR) (in percent) = %0.4f\n', (TP*100/(TP + FN))) %% = 1 - FNR
% % fprintf('\n[*] False Positive Rate (FPR), or False Alarm Rate (FAR) (in percent) = %0.4f\n', (FP*100/(FP + TN))) %% = 1 - TNR
% % fprintf('\n[*] False Discovery Rate (FDR) (in percent) = %0.4f\n', (FP*100/(FP + TP))) %% = 1 - PPV
% % fprintf('\n[*] Accuracy (in percent) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
% % fprintf('\n$$$$$$$$$$$$$$$\n\n')
% % 
% % 
% % %% RESULT PLOT
% % %%=============%%
% % 
% % TTTT = res(1:length(Dtest),1)';
% % 
% % f3 = figure(3); % Plot Test Dataset 
% %     set(f3,'Position',get(0,'screensize'));
% % 
% %     plot(Dtest(:,1),'Color',[1 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
% %     plot(Dtest(:,2),'Color',[0 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
% %     plot(Dtest(:,3),'Color',[0 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
% %     plot(Dtest(:,4),'Color',[1 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid; hold on;
% % 
% %     plot(Dtest(:,5),'Color',[0 0.4470 0.7410],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(Dtest(:,6),'Color',[0.8500 0.3250 0.0980],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(Dtest(:,7),'Color',[0 1 0],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(Dtest(:,8),'Color',[0.9290 0.6940 0.1250],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(Dtest(:,9),'Color',[0.4940 0.1840 0.5560],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(Dtest(:,10),'Color',[0.4660 0.6740 0.1880],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(Dtest(:,11),'Color',[0.3010 0.7450 0.9330],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(Dtest(:,12),'Color',[0.6350 0.0780 0.1840],'Marker','.','LineStyle', '-','LineWidth',2); grid; 
% % 
% % 
% %     xlim([0 size(Dtest,1)])
% %     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
% %     ylabel('Test-Dataset Vectors','fontsize',14,'FontName', 'Arial');
% %     % title(['Test dataset vectors for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
% %     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
% %     legend({'h1','h2','h3','h4','h5','h6','h7','h8','V1','V2','V3','V4'},'location','bestoutside')
% %     grid;
% % 
% % 
% %     %%%------%%%
% % 
% % 
% % f4 = figure(4); % Plot the complete residue space
% %     set(f4,'Position',get(0,'screensize'));
% % 
% %     plot(e(:,1),'Color',[1 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
% %     plot(e(:,2),'Color',[0 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
% %     plot(e(:,3),'Color',[0 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
% %     plot(e(:,4),'Color',[1 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid; hold on;
% % 
% %     plot(e(:,5),'Color',[0 0.4470 0.7410],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(e(:,6),'Color',[0.8500 0.3250 0.0980],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(e(:,7),'Color',[0 1 0],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(e(:,8),'Color',[0.9290 0.6940 0.1250],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(e(:,9),'Color',[0.4940 0.1840 0.5560],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(e(:,10),'Color',[0.4660 0.6740 0.1880],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(e(:,11),'Color',[0.3010 0.7450 0.9330],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
% %     plot(e(:,12),'Color',[0.6350 0.0780 0.1840],'Marker','.','LineStyle', '-','LineWidth',2); grid; 
% % 
% %     xlim([0 size(Dtest,1)])
% %     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
% %     ylabel('Residual Space','fontsize',14,'FontName', 'Arial');
% %     % title(['Model Errors for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
% %     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
% %     legend({'h1','h2','h3','h4','h5','h6','h7','h8','V1','V2','V3','V4'},'location','bestoutside')
% %     grid;
% % 
% %     %%%-----%%%
% % 
% %  f5 = figure(5); % Plot the faulty test-data vector
% %     set(f5,'Position',get(0,'screensize'));
% %     %plot(e(:,IndX),'k','LineWidth',2)
% % 
% %     EE = e(:,IndX)';
% %     plot(TTTT(1,1:lim), EE(1,1:lim), 'g','LineWidth',2); hold on;
% %     plot(TTTT(1,lim:end), EE(1,lim:end), 'r','LineWidth',2);
% % 
% %     xlim([0 size(Dtest,1)])
% %     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
% %     ylabel('Error Values','fontsize',14,'FontName', 'Arial');
% %     % title(['Error Plot (Faulty vector) for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
% %     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
% %     grid;
% % 
% %     %%%------%%
% % 
% % f6 = figure(6);
% %     % set(f6,'Position',get(0,'screensize'));
% %     subplot(2,1,1)
% %     % figure(6)
% %     set(gcf, 'WindowState', 'maximized');
% %     % plot(ts1, 'b','LineWidth',2); hold on;
% % 
% %     plot(TTTT(1,1:lim), ts1(1,1:lim), 'g','LineWidth',2); hold on;
% %     plot(TTTT(1,lim:end), ts1(1,lim:end), 'r','LineWidth',2);
% %     ylabel('PCA-T^2','fontsize',14,'FontName', 'Arial');
% %     hold on;
% %     plot(TS,'b--', 'LineWidth',3); 
% %     % hold on;
% %     % plot(TS_md,'k--', 'LineWidth',3)
% %     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
% %     xlim([0 size(Dtest,1)]);
% %     % title(['FD for 8-Tank system: T^2 limits, ', FaultID,' Fault ','of value = ', num2str(Bias_value)]);
% %     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(IndX)]);
% %     % legend({'T_{No-fault}^2','T_{Faulty}^2','T_{Thres}^2','T_{MD-Thres}^2'},'Location', 'best'),
% %     legend({'T_{No-fault}^2','T_{Faulty}^2','T_{Thres}^2'},'Location', 'best'),
% %     grid
% % 
% % 
% %     subplot(2,1,2)
% % 
% %     set(gcf, 'WindowState', 'maximized');
% %     % plot(SPE,'k','LineWidth',2); 
% %     plot(TTTT(1,1:lim), SPE(1,1:lim), 'g','LineWidth',2); hold on;
% %     plot(TTTT(1,lim:end), SPE(1,lim:end), 'r','LineWidth',2);
% %     % xlim([0 size(Dtest,1)]);
% %     %ylim([-50000 290000])
% %     ylabel('PCA-SPE-Q','fontsize',14,'FontName', 'Arial');
% %     hold on;
% %     plot(S1,'b--', 'LineWidth',3);  
% %     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
% %     % title(['FD for 8-Tank system: SPE/Q limits, ', FaultID,' Fault ','of value = ', num2str(Bias_value)]);
% %     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(IndX)]);
% %     legend({'SPE_{No-fault}','SPE_{Faulty}','Q-Thres.'},'Location', 'best')
% %     grid;
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
