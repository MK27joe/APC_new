function [ConfMatrx] = ConFusion2(Stats, IntrmF, ValsIn)

% % IntrmF = [0,a,b,c,d];
PariF = IntrmF(1); %%% Parity flag: 0 = No Intermitt ; 1 = Intermitt.
a = IntrmF(2); 
b = IntrmF(3); 
c = IntrmF(4); 
d = IntrmF(5);

% % ValsIn = [m2, lim, T2knbeta, Win_size];
ObsvCount = ValsIn(1);
lim = ValsIn(2);
Thresh = ValsIn(3);
Win_size = ValsIn(4);

lim = ValsIn(2) + ValsIn(4);

FP = 0; FN = 0; TN = 0; TP = 0;

if PariF == 0     %%% (A == lim) && (B == 0) && (C == 0) && (D == 0) %% for Bias/Age/PD/Freeze

        for i = 1:ObsvCount          %% no. of observations in Dtest
    
            if i < lim               %% within the no-fault region
                if Stats(i) > Thresh %% False-Alarm
                    FP = FP + 1;
                    TN = TN;
                else
                    FP = FP;
                    TN = TN + 1;
                end
            else                     %% within the faulty region
                if Stats(i) < Thresh %% Miss Alarm
                    FN = FN + 1;
                    TP = TP;
                else
                    FN = FN;
                    TP = TP + 1;
                end
            end
        end
  

else %% during twin-intermittent faults
     a = IntrmF(2) + Win_size; b = IntrmF(3) + Win_size; c = IntrmF(4) + Win_size; d = IntrmF(5) + Win_size;

    for i = 1:ObsvCount          %% no. of observations in Dtest

        if (((i > a) && (i < b)) || ((i > c) && (i < d))) %% within the fault regions

            if Stats(i) < Thresh %% Miss Alarm
                        FN = FN + 1;
                        TP = TP;
            else
                        FN = FN;
                        TP = TP + 1;
            end
            

       else                     %% within the no-fault regions
            if Stats(i) > Thresh %% False-Alarm
                FP = FP + 1;
                TN = TN;
            else
                FP = FP;
                TN = TN + 1;
            end

       end
        

    end

end
Precision_pc = TP*100 / (TP + FP);
Recall_pc = TP*100 / (TP + FN);
FAR_pc = FP*100 / (FP + TN); %% = 1 - TNR
F1Score = 2*TP / (2*TP + FP + FN);
Accuracy_pc = (TP + TN)*100/(TP + TN + FP + FN);
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TN+FN)*(FP+TN)*(TP+FN));
ConfMatrx = [];
ConfMatrx = [FP, TN, FN, TP,Precision_pc,Recall_pc,FAR_pc,F1Score,Accuracy_pc,MCC]';

end



            
            

