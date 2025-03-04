function [FP, TN, FN, TP] = FaultRatios(ObsvCount, lim, Stats, Thresh)

FP = 0; FN = 0; TN = 0; TP = 0;

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
        if Stats(i) < Thresh %% False No-fault
            FN = FN + 1;
            TP = TP;
        else
            FN = FN;
            TP = TP + 1;
        end
    end

end

end
