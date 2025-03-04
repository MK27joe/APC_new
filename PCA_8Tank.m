
close all; clc; clear; 

% global T1 FaultID;
% 
% diary on

% fprintf('\n#####################################\n')
% Date = string(datetime('now'),'yyyy-MM-dd_HH:mm:ss')
% fprintf('\n#####################################\n')

% % %% CHOOSE YOUR SYSTEM FOR DATABASE EXTRACTION
% % %%===========================================%%
% % 
% % for ij = 1
% %     fprintf('\n');
% %     disp('Choose the tank-system for analysis:')
% %     IPi = input('\n\nPress NumKey "0" for Quadruple-Tank System, or, "1" for Octuple-Tank System:  ');
% % 
% %     if IPi == 0
% %         T = readmatrix("Data4Tank.xlsx"); 
% %         Tname = 'Quadruple'; 
% %         T1 = 4; 
% %         IPs = 2; 
% %         OPs = 4;
% % 
% %     elseif IPi == 1
% %         T = readmatrix("Data8Tank.xlsx"); 
% %         Tname = 'Octuple'; 
% %         T1 = 8; 
% %         IPs = 4; 
% %         OPs = 8;
% % 
% %     else 
% %         fprintf('\n\n** WARNING: Your input is wrong :( ... Considering Octuple tank system as the DEFAULT !! **\n')
% %         disp('===================================================================================================')
% % 
% %         T = readmatrix("Data8Tank.xlsx"); 
% %         Tname = 'Octuple'; 
% %         T1 = 8; 
% %         IPs = 4; 
% %         OPs = 8;
% %     end
% % 
% %     fprintf('\nYou pressed "%d" for %s-Tank System.\n', IPi,Tname)
% % end

load octuple_tank_data_28_08_24.mat

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
[U,S,V] = svd(CV); %% Top Max EigValues of CV (in S)


%% PCA PARAMETERS USING TRAINING DATASET
%%======================================%%

[COEFF, SCORE, LATENT, TSQUARED] = pca(Xbar); %% LATENT (i.e. EigValues)


%% ANALYSIS OF PCS FROM PCA METHOD
%%================================%%


prompt = input('\n\nEnter the percentage of significance (b/w 80 to 95 %) = '); %%% self explanatory !!
% prompt = 80;

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

% %---------- Display the details of PCs -------------------%
% 
for ij = 1
    fprintf('\n\n\n=================================\n')
    fprintf('|| Octuple-Tank system: Test Results ||\n')
    fprintf('=================================\n\n\n')

    disp('//////// PART A:: SYSTEM INFO ///////'), 
    % fprintf('\n')
    % fprintf('\n>==> Number of Inputs = %d \n', 4)
    % fprintf('\n>==> Number of Outputs = %d \n\n', 8)

    %SysDetail()

    fprintf('\n** Total Number of Observations (Database) = %d \n', M)
    fprintf('\n** Total Number of Observations (Training) = %d \n', m1)
    fprintf('\n** Total Number of Observations (Testing) = %d \n\n', m2)


    % disp('~~~~~~~~~~~~~~~')
    % fprintf('Obervations\n')
    % disp('~~~~~~~~~~~~~~~')

    fprintf('\n\n\n'); disp('//////// PART B:: PCs INFO ///////'), 
    fprintf('\n')

    fprintf('\n==> The percentage of significance set for PC contribution = %0.4f', prompt)

    fprintf('\n\n(1) No. of PCs chosen = %d \n\n',k)

    fprintf('(2) Cumul. PC contrib: \n   ********************\n '), alpha         

    fprintf('## Total PC contribution computed = %0.4f\n\n', (alpha(1,end) * 100))

end

princ=SCORE(:,1:k); %%% upto first kth PCs
per=LATENT/sum(LATENT);


%% INTRODUCE FAULTS: BIAS/DRIFT/PREC. DEGRADATION/FREEZING/INTERMITTENT
%%=====================================================================%%

pause() %% Disable / Comment THIS line if not required !!

fprintf('\n\nLimit-instance can range from observation count n = 1 to %d.\n', m2)
lim = input('\nEnter the instant of fault, lim =  ');
%lim = size(Dtest,1)/2;

for ij = 1
    fprintf('\n\n'); disp('//////// PART C:: FAULT DETAILS ///////'), 
    fprintf('\n\n')

    TankNum = input('Tank # Selection = ');
    IndX = TankNum;

    fprintf('\nFault introduced at %d-th observation in Test dataset.', lim)


    fprintf('\n\nIntroduce ANY ONE fault of your choice:') 
    fprintf(['\n* Press key "a" for "Ageing (or, Drift)" ' ...
        '\n* Press key "b" for "Bias" ' ...
        '\n* Press key "f" for "Freezing" ' ...
        '\n* Press key "i" for "Intermittent" ' ...
        '\n* Press key "p" for "Precision-degradation"\n'])

    Fi = input('\nEnter your choice of fault:  ','s');
    fprintf('\nYou pressed "%s".\n\n', Fi)


    if Fi == 'a'
        FaultID = 'Drift';
        Dslope = input('\nEnter the slope of Ageing (i.e. Drift) fault:  ');
        Bias = 0; Mag_PD =0; idx=0; A=0;
        Dtest = FDrift(Dtest,lim,IndX,Dslope);


    elseif Fi == 'b'
        FaultID = 'Bias';
        Bias = input('\nEnter Bias-fault value:  ');
        Dslope = 0; Mag_PD = 0; idx=0; A=0;
        Dtest = FBias(Dtest,lim,IndX,Bias);



    elseif Fi == 'f'
        FaultID = 'Freeze';
        Bias=0;Dslope =0; Mag_PD =0;
        [Dtest,idx,A] = FFreeze(Dtest,lim,IndX);



    elseif Fi == 'i'
        FaultID = 'Intermittent';
        Bias=0;Dslope =0; Mag_PD =0; idx=0; A=0; lim=300;
        [Dtest] = FIntermit(Dtest,lim,IndX);

    else 
        % % Fi == 'p'
        FaultID = 'Precision-Degradation';
        Dslope = input('\nEnter the slope of Drift fault (w/ Precision Degradation):  ');
        Mag_PD = input('\nEnter the level of degradation within the interval (0, 1):  ');
        Bias=0; idx=0; A=0;
        Dtest = FDPD(Dtest,lim,IndX,Dslope,Mag_PD);
    end

end


%% NORMALIZATION OF TESTING DATASET
%%=================================%%


fprintf('\n\n\n'); disp('//////// PART D:: PCA MODEL-TEST INFO ///////'), 
fprintf('\n')


SY = scale1(Dtest,xm,Sdm); %% Normalizing Dtest using the mean and STDEV of Dtrain
XT1 = SY*COEFF;

Xp = SY*COEFF(:,1:k)*COEFF(:,1:k)';   % Xp is the estimation of original data using the PCA model, X=Xp+E

e = SY - Xp; %% Residual Space



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

ca = norminv(0.99, 0, 1);% ca is value for standard normal distribution for confidence level 95%

SPEbeta = theta(1)*((ca*h0*sqrt(2*theta(2))/theta(1))+1+(theta(2)*h0*(h0-1)/theta(1)^2))^(1/h0);
fprintf('\n\n(4) SPE-Q Beta = %0.4f\n\n', SPEbeta)

S1 = SPEbeta*ones(size(SY,1),1);



%%--------------------- 2. SPE/Q statistic computation ----------------%

for i = 1:size(SY,1)
    SPE(i) = sum((SY(i,:) - Xp(i,:)).^2); % Sum of Predicted Errors
end



%% COMPUTE T^2 STATISTICS 
%%=======================%%



%%--------------------- 1. T^2 threshold computation ------------------%

T2knbeta = (k*(size(Xbar,1)^2-1)/(size(Xbar,1)*(size(Xbar,1)-k))) * finv(0.95,k,size(Xbar,1)-k);
fprintf('\n\n(5) T2knbeta = %0.4f\n\n', T2knbeta)


TS = T2knbeta*ones(size(SY,1),1);


%%--------------------- 2. T^2 statistic computation ------------------%

for i=1:size(SY,1)
    ts1(i) = XT1(i,1:k) * diag( 1 ./ LATENT(1:k) ) * XT1(i,1:k)';
end


% %  FINV  : Inverse of the F cumulative distribution function.
% %     X=FINV(beta,k,size(X,1)-k) returns the inverse of the F distribution 
% %     function with k and size(X,1)-k degrees of freedom, at the values in beta.


%% FAR and FDR COMPUTATION
%%========================%%

%%--------------------- 1. T^2  ------------------%

fprintf('\n\n$$$$$$$$$$$$$$$\n')
disp('A. Using T^2: ')
fprintf('\n$$$$$$$$$$$$$$$\n\n')

[FP, TN, FN, TP] = FaultRatios(m2, lim, ts1, T2knbeta)

fprintf('\n\n[*] True Positive Rate (TPR) (in percent) = %0.4f\n', (TP*100/(TP + FN))) %% = 1 - FNR
% fprintf('\n[b] False Negative Rate (FNR) (in percent) = %0.4f\n', (FN*100/(FN + TP))) %% = 1 - TPR
% fprintf('\n[c] True Negative Rate (TNR) (in percent) = %0.4f\n', (TN*100/(TN + FP))) %% = 1 - FPR
fprintf('\n[*] False Positive Rate (FPR), or False Alarm Rate (FAR) (in percent) = %0.4f\n', (FP*100/(FP + TN))) %% = 1 - TNR
% fprintf('\n[e] Positive Predictive Value (PPV) (in percent) = %0.4f\n', (TP*100/(TP + FP))) %% = 1 - FDR
fprintf('\n[*] False Discovery Rate (FDR) (in percent) = %0.4f\n', (FP*100/(FP + TP))) %% = 1 - PPV
% fprintf('\n[g] Negative Predictive Value (NPV) (in percent) = %0.4f\n', (TN*100/(TN + FN))) %% = 1 - FOR
% fprintf('\n[h] False Omission Rate (FOR) (in percent) = %0.4f\n', (FN*100/(FN + TN))) %% = 1 - NPV
% fprintf('\n[i] Threat Score (TS), or Critical Success Index (CSI) (in percent) = %0.4f\n', (TP*100/(TP + FN + FP)))
fprintf('\n[*] Accuracy (in percent) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
fprintf('\n$$$$$$$$$$$$$$$\n\n')


%%--------------------- 2. SPE-Q  ------------------%

fprintf('\n$$$$$$$$$$$$$$$\n')
disp('B. Using SPE\Q: ')
fprintf('\n$$$$$$$$$$$$$$$\n\n')

[FP, TN, FN, TP] = FaultRatios(m2, lim, SPE, SPEbeta)

fprintf('\n\n[*] True Positive Rate (TPR) (in percent) = %0.4f\n', (TP*100/(TP + FN))) %% = 1 - FNR
% fprintf('\n[b] False Negative Rate (FNR) (in percent) = %0.4f\n', (FN*100/(FN + TP))) %% = 1 - TPR
% fprintf('\n[c] True Negative Rate (TNR) (in percent) = %0.4f\n', (TN*100/(TN + FP))) %% = 1 - FPR
fprintf('\n[*] False Positive Rate (FPR), or False Alarm Rate (FAR) (in percent) = %0.4f\n', (FP*100/(FP + TN))) %% = 1 - TNR
% fprintf('\n[e] Positive Predictive Value (PPV) (in percent) = %0.4f\n', (TP*100/(TP + FP))) %% = 1 - FDR
fprintf('\n[*] False Discovery Rate (FDR) (in percent) = %0.4f\n', (FP*100/(FP + TP))) %% = 1 - PPV
% fprintf('\n[g] Negative Predictive Value (NPV) (in percent) = %0.4f\n', (TN*100/(TN + FN))) %% = 1 - FOR
% fprintf('\n[h] False Omission Rate (FOR) (in percent) = %0.4f\n', (FN*100/(FN + TN))) %% = 1 - NPV
% fprintf('\n[i] Threat Score (TS), or Critical Success Index (CSI) (in percent) = %0.4f\n', (TP*100/(TP + FN + FP)))
fprintf('\n[*] Accuracy (in percent) = %0.4f\n', ((TP + TN)*100/(TP + TN + FP + FN)))
fprintf('\n$$$$$$$$$$$$$$$\n\n')
% 
% 
% 
% %% RESULT PLOT
% %%=============%%
% 
% 
% diary('DailySummaries.txt')  %%% Contains Command window results
% diary off
% 
% 
% 
% %%%%%% WARNING: Activate appropriate function file (given below) before execution %%%%%%%%
% %%%%%%%%%%%%%%%%% Generalized Code is still in progress !! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
FigPlotBias_PCA(8,TankNum,IndX,FaultID,Bias,lim,ts1,TS,SPE,S1,Dtest,e) 
% 
% % FigPlotAge_PCA(8,TankNum,IndX,FaultID,Dslope,lim,ts1,TS,SPE,S1,Dtest,e)
% 
% % FigPlotFRZ_PCA(8,TankNum,IndX,FaultID,lim,idx,A,ts1,TS,SPE,S1,Dtest,e)
% %%%%% Freezing --> not working ??
% 
% % FigPlotPD_PCA(8,TankNum,IndX,FaultID,lim,Dslope,Mag_PD,ts1,TS,SPE,S1,Dtest,e)
% 
% % FigPlotInt_PCA(8,TankNum,IndX,FaultID,lim,Dslope,Mag_PD,ts1,TS,SPE,S1,Dtest,e)
% 
% %%%%%%%%%%%%%%%%% All individ. function files ==> Working successfully !! %%%%%%%%%%%%%%%%%%
% 
% 
% 
% %%%%%% This code-section is under test !! dtd. June 28, 2024 %%%%%%%%%%%%%
% 
% 
% 
% % % % mydoc = publish("MKM_PCA_new.m","pdf");
% % % % winopen(mydoc)
% 
% 
