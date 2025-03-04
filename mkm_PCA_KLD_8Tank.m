%% DISCLAIMER: This code-file is NOT developed for user-interactive mode

%%
close all; clc; clear; 


%% Load the datasheet
load octuple_tank_data_11_11_24.mat


%% 0. Extract your database
%%=======================%%

%%%% T == res
%%% Outputs --> Col 2 to 9
%%% Inputs --> Col 10 to 13
%%% Disturbances --> Col 14 to 15

xt = res(:,2:end);  
[M,N] = size(xt);



%% 1. Generate Residue from training data v/s PCA-model
%%===================================================%% 

%---- 1a. Extract dataset for training
Dtrain = xt(1:M/2,:);
[m1,n1] = size(Dtrain);


%---- 1b. Normalization 
xm = mean(Dtrain);
Sdm = std(Dtrain);
Xbar = (Dtrain - xm(ones(m1,1),:)) ./ (Sdm(ones(m1,1),:));
CV =cov(Xbar);
[U,S,V] = svd(CV); 


%---- 1c. PCA parameters using Xbar
[COEFF1, SCORE1, LATENT1, TSQUARED1, EXPLAINED1, MU1] = pca(Xbar); 
%%% 'Latent' = Eigen Values
%%% 'Score' contains PC vectors (column-vectored)
%%% 'Coeff' = Loading Matrix


%---- 1d. FInd the desired no. of PCs (k) using CPV for 98% yield
prompt = 98;
percent = prompt/100;
k1=0; %%% Initial count = 0
for i = 1:size(LATENT1,1)                                                        
    alpha(i)=sum(LATENT1(1:i))/sum(LATENT1);
    if alpha(i)>=percent
            k1=i;
            break;
    end 
end


%---- 1e. Find the Residue Space (Training Dataset) 
Xp1 = Xbar*COEFF1(:,1:k1)*COEFF1(:,1:k1)';   % Xp is the estimation of original data using the PCA model, X=Xp+E
e1 = Xbar - Xp1; %% Residual Space @ No-fault scenario



%% 2. Generate Residue from testing data v/s PCA-model
%%===================================================%% 

%---- 2a. Extract dataset for testing
Dtest = xt(1+(M/2):M,:);
[m2,n2] = size(Dtest);


%---- 2b(i). Introduce a fault at instant (n = lim) in Tank (# = IndX)
lim = 300; 
% fprintf('\nFault introduced at %d-th observation in Test dataset.', lim)
IndX = 4; %%% Tank# 4
% fprintf('\nTank # Selection = %d\n',IndX);


%---- 2b(ii). Choose the nature of fault (Bias/Drift/Drift+Deg/Freeze/Intermittent)
for xyz = 1

    % % -------------------- 1. Bias Fault -------------------------% %
    % FaultID = 'Bias';
    % Bias_value = 5;
    % for i = 1:m2
    %     if(i>lim)
    %         Dtest(i,IndX) = Dtest(i,IndX) + Bias_value;
    %     end
    % end


    % % % -------------------- 2. Drift Fault -------------------------% %
    % FaultID = 'Drift';
    % Dslope = 0.04;
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


    % % --------------- 3. Drift + Prec. Deg. Fault --------------------% %

    FaultID = 'Drift+PD';
    Dslope = 0.09;
    Mag_PD = 0.45;
    r = [];
    for i = 1:(m2-lim)
        r(i)=i;
    end

    for i=1:m2
        if(i>lim)
            Dtest(i,IndX) = Dtest(i,IndX) + Dslope*r(i-lim)*Mag_PD*rand(1);
        end
    end



    % % % ------------------ 4. Freeze Fault --------------------% %
    % 
    % FaultID = 'Freeze';
    % m2 = size(Dtest,1); 
    % % lim = size(Dtest1,1)/2; 
    % 
    % idx = round(randi(numel(Dtest(lim+1:end, IndX)))/4); %%  Select a random cell index, idx, corr. to MFault_ID vector
    % 
    % A = Dtest(lim+idx, IndX); %% Select that cell value corr. to idx
    % 
    % for i =1:m2
    %     if(i>lim+idx)
    %         Dtest(i,IndX) = 2*A; %% replace remaining cells as 'A' %%% Bias of 2
    %     end
    % end


    % % % ------------------ 5. Intermittent Fault --------------------% %
    % 
    % FaultID = 'Intermittent';
    % a= lim + 25;
    % b= lim + 125;
    % c= m2 - 250;
    % d= m2 - 150;
    % 
    % 
    % for i=1:m2
    %     if (((i > a) && (i < b)) || ((i > c) && (i < d))) %||((i>1050) && (i<1150)))
    %         Dtest(i,IndX) = Dtest(i,IndX) + 10;
    %     end
    % end

end


%---- 2c. Normalization
SY = scale1(Dtest,xm,Sdm); %% Normalizing Dtest using the mean and STDEV of training data
CV_test =cov(SY);


%---- 2d. Find the Residue Space (Testing Dataset)
Xp = SY*COEFF1(:,1:k1)*COEFF1(:,1:k1)';   % Xp is the estimation of original data using the training PCA model, X=Xp+E
e2 = SY - Xp; %% Residual Space @ Actual scenario
XT1 = SY*COEFF1;

%---- 1f. Display the results (obtained so far)
 
% for ij = 1
%     fprintf('\n\n\n=================================\n')
%     fprintf('|| Octuple-Tank system: Test Results ||\n')
%     fprintf('=================================\n\n\n')
% 
%     disp('//////// PART A:: SYSTEM INFO ///////'), 
% 
%     fprintf('\n** Total Number of Observations (Database) = %d \n', M)
%     fprintf('\n** Total Number of Observations (Training) = %d \n', m1)
%     fprintf('\n** Total Number of Observations (Testing) = %d \n\n', m2)
% 
%     % disp('~~~~~~~~~~~~~~~')
%     % fprintf('Obervations\n')
%     % disp('~~~~~~~~~~~~~~~')
% 
%     fprintf('\n\n\n'); disp('//////// PART B:: PCs INFO ///////'), 
%     fprintf('\n')
% 
%     fprintf('\n==> The percentage of significance set for PC contribution = %0.4f', prompt)
% 
%     fprintf('\n\n(1) No. of PCs chosen = %d out of 14 I/O vectors.\n',k1)
% 
%     fprintf('\n(2) Cumul. PC contrib: \n   ********************\n '), alpha,         
% 
%     TotPCcontr = alpha(1,end) * 100;
%     fprintf('## Total PC contribution computed = %0.4f\n\n\n', TotPCcontr)
% 
% %---%
% princ = SCORE1(:,1:k1); %%% upto first kth PCs
% per = LATENT1/sum(LATENT1);
% nnn = [1:length(EXPLAINED1)];
% CExpVar = zeros(1,length(nnn));
% for xyz = 1:length(nnn)
%     CExpVar(xyz+1) = CExpVar(xyz) + EXPLAINED1(xyz);
% end
% 
% CExpVar(1) = [];
% 
% f1 = figure(1); % Plot the faulty test-data vector
%     set(f1,'Position',get(0,'screensize'));
%     cdata=CV;
%     xvalues={'h1','h2','h3','h4','h5','h6','h7','h8','v1','v2','v3','v4','d1','d2'};
%     yvalues={'h1','h2','h3','h4','h5','h6','h7','h8','v1','v2','v3','v4','d1','d2'};
%     h=heatmap(xvalues,yvalues,cdata);
%     colormap(parula)
%     h.Title = 'Heatmap of the Co-variance (correlation) matrix of Training dataset Dtest';
% 
% 
% f2 = figure(2);
%     set(f2,'Position',get(0,'screensize'));
%     subplot(311)
%     set(gcf, 'WindowState', 'maximized');
%     plot(nnn,EXPLAINED1,'Color',[0,0,0],'Marker','o','LineStyle', '-','LineWidth',2),grid; 
%     title(['Cumulative Percentage Variance (CPV) Technique to find the required number of PCs (at a significance of ', num2str(prompt) ,'%).'])
%     subtitle('(a)')
%     ylabel('Eigen Value (i.e. Variance)','fontsize',14,'FontName', 'Arial');
%     %%% xlim([1 length(nnn)]);
% 
%     subplot(312)
%     set(gcf, 'WindowState', 'maximized');
%     bar(EXPLAINED1,'LineWidth',1.2); 
%     subtitle('(b)')
%     ylabel('Explained Var. (%)','fontsize',14,'FontName', 'Arial'), grid;
% 
%     subplot(313)
%     set(gcf, 'WindowState', 'maximized');
%     bar(CExpVar,'LineWidth',1.2); 
%     subtitle('(c)')
%     ylabel('Cumul. Explained Var. (%)','fontsize',14,'FontName', 'Arial'),grid;
%     ylim([0,100]);
% 
%     xlabel('Principal Components','fontsize',12,'FontName', 'Arial');
% 
% 
% TTTT = res(1:length(Dtest),1)';
% f3 = figure(3); % Plot Test Dataset 
%     set(f3,'Position',get(0,'screensize'));
% 
%     plot(Dtest(:,1),'Color',[1 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(Dtest(:,2),'Color',[0 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(Dtest(:,3),'Color',[0 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(Dtest(:,4),'Color',[1 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid; hold on;
% 
%     plot(Dtest(:,5),'Color',[0 0.4470 0.7410],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(Dtest(:,6),'Color',[0.8500 0.3250 0.0980],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(Dtest(:,7),'Color',[0 1 0],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(Dtest(:,8),'Color',[0.9290 0.6940 0.1250],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(Dtest(:,9),'Color',[0.4940 0.1840 0.5560],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(Dtest(:,10),'Color',[0.4660 0.6740 0.1880],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(Dtest(:,11),'Color',[0.3010 0.7450 0.9330],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(Dtest(:,12),'Color',[0.6350 0.0780 0.1840],'Marker','.','LineStyle', '-','LineWidth',2); grid; 
% 
% 
%     xlim([0 size(Dtest,1)])
%     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
%     ylabel('Dataset Vectors','fontsize',14,'FontName', 'Arial');
%     % title(['Test dataset vectors for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
%     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
%     legend({'h1','h2','h3','h4','h5','h6','h7','h8','V1','V2','V3','V4'},'location','bestoutside')
%     grid;
% 
% 
% f4 = figure(4); % Plot the complete residue space e1
% set(f4,'Position',get(0,'screensize'));
% 
%     plot(e1(:,1),'Color',[1 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(e1(:,2),'Color',[0 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(e1(:,3),'Color',[0 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(e1(:,4),'Color',[1 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid; hold on;
%     plot(e1(:,5),'Color',[0 0.4470 0.7410],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e1(:,6),'Color',[0.8500 0.3250 0.0980],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e1(:,7),'Color',[0 1 0],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e1(:,8),'Color',[0.9290 0.6940 0.1250],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e1(:,9),'Color',[0.4940 0.1840 0.5560],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e1(:,10),'Color',[0.4660 0.6740 0.1880],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e1(:,11),'Color',[0.3010 0.7450 0.9330],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e1(:,12),'Color',[0.6350 0.0780 0.1840],'Marker','.','LineStyle', '-','LineWidth',2); grid; 
% 
%     xlim([0 size(Dtest,1)])
%     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
%     ylabel('Residue Space_{Training}','fontsize',14,'FontName', 'Arial');
%     % title(['Model Errors for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
%     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
%     legend({'h1','h2','h3','h4','h5','h6','h7','h8','V1','V2','V3','V4'},'location','bestoutside')
%     grid;
% 
% %%%-----%%%
% 
% f5 = figure(5); % Plot the complete residue space e2
% set(f5,'Position',get(0,'screensize'));
% 
%     plot(e2(:,1),'Color',[1 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(e2(:,2),'Color',[0 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(e2(:,3),'Color',[0 0 0],'Marker','.','LineStyle', '-','LineWidth',1); grid, hold on;
%     plot(e2(:,4),'Color',[1 0 1],'Marker','.','LineStyle', '-','LineWidth',1); grid; hold on;
%     plot(e2(:,5),'Color',[0 0.4470 0.7410],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e2(:,6),'Color',[0.8500 0.3250 0.0980],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e2(:,7),'Color',[0 1 0],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e2(:,8),'Color',[0.9290 0.6940 0.1250],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e2(:,9),'Color',[0.4940 0.1840 0.5560],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e2(:,10),'Color',[0.4660 0.6740 0.1880],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e2(:,11),'Color',[0.3010 0.7450 0.9330],'Marker','.','LineStyle', '-','LineWidth',2); grid; hold on;
%     plot(e2(:,12),'Color',[0.6350 0.0780 0.1840],'Marker','.','LineStyle', '-','LineWidth',2); grid; 
% 
%     xlim([0 size(Dtest,1)])
%     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
%     ylabel('Residue Space_{Testing}','fontsize',14,'FontName', 'Arial');
%     % title(['Model Errors for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
%     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
%     legend({'h1','h2','h3','h4','h5','h6','h7','h8','V1','V2','V3','V4'},'location','bestoutside')
%     grid;
% 
% %%%-----%%%
% 
% f6 = figure(6); % Plot the faulty test-data vector
% set(f6,'Position',get(0,'screensize'));
%     %plot(e(:,IndX),'k','LineWidth',2)
% 
%     EE = e2(:,IndX)';
%     plot(TTTT(1,1:lim), EE(1,1:lim), 'g','LineWidth',2); hold on;
%     plot(TTTT(1,lim:end), EE(1,lim:end), 'r','LineWidth',2);
% 
%     xlim([0 size(Dtest,1)])
%     xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
%     ylabel('Faulty Measurement Vector','fontsize',14,'FontName', 'Arial');
%     % title(['Error Plot (Faulty vector) for ', num2str(T1),'-Tank system [SNR = [] and Fault = ', FaultID,' and of value = ', num2str(Bias),']']);
%     % subtitle(['Fault at n = ', num2str(lim),' for Tank # ', num2str(TankNum)]);
%     grid;
% 
% end



%% 3. Analysis of Residues
%%======================%%
%-- Ref: https://www.sciencedirect.com/science/article/abs/pii/S0950423016302273?via%3Dihub 
%-- Kullback-Leibler distance-based enhanced detection of incipient anomalies
%-- Fouzi Harrou, Ying Sun, Muddu Madakyaru (2016)


% %---- 3a. Determine the mean and Stdev. of residues e1 and e2
% mu0 = mean(e1);
% sigma0 = std(e1); 
% var0 = sigma0.*sigma0; %%% = sigma0^2.. You can also use the code:: var(e1) 
% 
% mu1 = mean(e2);%%% Assuming:: var(1) = var(e2) [..approx.]
% % sigma1 = std(e2); 
% % var1 = sigma1.*sigma1;
% 
% 
% %---- 3b. Determine the threshold, h
% L = 3; %%% width of the control limits that determines the confidence limits, usually specified in practice as 3 (i.e Three-sigma rule) for a false alarm rate of 0.27%
% h =  mu0 + L*sigma0;
% 
% 
% %---- 3c. Compute the (symmetrized version) Kullback-Leibler distance, J
% J = ((mu1 - mu0).^2) ./var0;
% 
% 
% %---- 3d. Compare J and h
% 
% %--- 3d(i). Display J vs h Table
% 
% disp('/////////////////////////////////////////')
% fprintf('\n///// PART C:: Kullback-Leibler Distance INFO ///\n')
% disp('/////////////////////////////////////////')
% 
% ANSW =[J',h'];
% KLD_result = array2table(ANSW,'VariableNames',{'J','h'})
% 
% %--- 3d(ii). Generate the Decision on FD
% %%% Method #1
% fprintf('\nKLD_Method #1: Element-wise Comparison\n')
% count = 0;
% for rr = 1:size(h,2) %% .. or size(J,2). Both are the same.
%     if J(rr) > h(rr)
%         count = count + 1;
%     else
%         continue;
%     end
% end
% 
% if count > 0
%     fprintf('\n** Fault Detected from Test-data\n')
% else
%     fprintf('\n** No anomalies Detected from Test-data\n')
% end
% 
% %%% Method #2
% fprintf('\nKLD_Method #2: Vector-norm comparison\n')
% if norm(J) > norm(h)
%     fprintf('\n** Fault Detected from Test-data\n')
% else
%     fprintf('\n** No anomalies Detected from Test-data\n')
% end




%%
% 
% figure(1), plot(e1), figure(2), plot(e2)



%%

%-- Convert 1000x14 to 1000x1 vector
NormMat_NF= [];
for ii = 1:size(e1,1)
    NormMat_NF(ii,1) = norm(e1(ii,:));
end

%-- KLD Threshold
mu0 = mean(NormMat_NF)
var0 = var(NormMat_NF)
h = mu0 + 3*sqrt(var0);
hLine = h*ones(size(e1,1),1);

plot(hLine,'r','LineWidth',2);
hold on

%%

%-- Convert 1000x14 to 1000x1 vector
NormMat_F= [];
for ii = 1:size(e2,1)
    NormMat_F(ii,1) = norm(e2(ii,:));
end

Win_size = 70; %% Moving window size, N = 50 (say)
SubDatSize = size(NormMat_F,1)/Win_size;


J = zeros(1, length(NormMat_F) - Win_size);;

for i = 1:length(NormMat_F) - Win_size
    J(i) = (mean(NormMat_F(i:(i+Win_size-1))) - mu0).^2 / var0;
end
J=J';
dum = zeros(Win_size,1);
Jt = [dum;J];


plot(Jt,'b','LineWidth',2); grid
legend({'KLD_{Threshold}','KLD_{Statistics}'},'Location','best')
xlabel('Observation Number','fontsize',12,'FontName', 'Arial');
ylabel('KLD Statistics, J','fontsize',14,'FontName', 'Arial');
title('Fault Detection using KL Divergence method');

% %% COMPUTE T^2 STATISTICS 
% %%=======================%%
% 
% 
% 
% %%--------------------- 1a. T^2 threshold computation ------------------%
% 
% T2knbeta = (k1*(size(Xbar,1)^2-1)/(size(Xbar,1)*(size(Xbar,1)-k1))) * finv(0.95,k1,size(Xbar,1)-k1);
% fprintf('\n(5) T2knbeta = %0.4f\n', T2knbeta)
% TS = T2knbeta*ones(size(SY,1),1);
% 
% 
% % %%--------------------- 1b. Mahalanobis Distance (D) computation ------------------%
% % 
% % T2_MD = (size(Xbar,2) * (size(Xbar,1)^2-1)/(size(Xbar,1)*(size(Xbar,1) - size(Xbar,2)))) * finv(0.95,k,size(Xbar,1) - size(Xbar,2));
% % fprintf('\n(5) T2_MD = %0.4f\n', T2_MD)
% % TS_md = T2_MD*ones(size(SY,1),1);
% 
% 
% %%--------------------- 2. T^2 statistic computation ------------------%
% 
% for i=1:size(SY,1)
%     ts1(i) = XT1(i,1:k1) * diag( 1 ./ LATENT1(1:k1) ) * XT1(i,1:k1)';
% end
% 
% 
% %% COMPUTE SPE/Q STATISTICS AND THRESHOLD
% %%=======================================%%
% 
% 
% 
% %%--------------------- 1. SPE/Q threshold computation ----------------%
% 
% 
%  beta = 0.95;      % takes values between 90% and 95%
%  theta = zeros(3,1);
%  for ii = 1:3
%      for j = k1+1:size(SY,2)
%          theta(ii) = theta(ii) + LATENT1(j)^(ii);
%      end
%  end
% h0 = 1-((2*theta(1)*theta(3))/(3*theta(2)^2));
% ca = norminv(0.95, 0, 1);% ca is value for standard normal distribution for confidence level 95%
% SPEbeta = theta(1)*((ca*h0*sqrt(2*theta(2))/theta(1))+1+(theta(2)*h0*(h0-1)/theta(1)^2))^(1/h0);
% fprintf('\n(4) SPE-Q Beta = %0.4f\n', SPEbeta)
% S1 = SPEbeta*ones(size(SY,1),1);
% 
% 
% %%--------------------- 2. SPE/Q statistic computation ----------------%
% 
% for i = 1:size(SY,1)
%     SPE(i) = sum((SY(i,:) - Xp(i,:)).^2); % Sum of Predicted Errors
% end
% 
% plot(ts1,'r','LineWidth',2); hold on; plot(S1,'b','LineWidth',2); hold on; plot(Jt,'k','LineWidth',2); grid on



%%
[FP, TN, FN, TP] = FaultRatios(m2, lim, Jt, h)

fprintf('\n[*] True Positive Rate (TPR) (in percent) = %0.4f\n', (TP*100/(TP + FN))) %% = 1 - FNR
fprintf('\n[*] False Positive Rate (FPR), or False Alarm Rate (FAR) (in percent) = %0.4f\n', (FP*100/(FP + TN))) %% = 1 - TNR
fprintf('\n[*] False Discovery Rate (FDR) (in percent) = %0.4f\n', (FP*100/(FP + TP))) %% = 1 - PPV
fprintf('\n[*] Accuracy (in percent) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
fprintf('\n$$$$$$$$$$$$$$$\n\n')





    





