function [Dtest1,idx,A] = FFreeze(Dtest1,lim,MFault_ID)

m2 = size(Dtest1,1); 
% lim = size(Dtest1,1)/2; 

idx = round(randi(numel(Dtest1(lim+1:end, MFault_ID)))/4) %%  Select a random cell index, idx, corr. to MFault_ID vector

A = Dtest1(lim+idx, MFault_ID); %% Select that cell value corr. to idx

for i =1:m2
    if(i>lim+idx)
        Dtest1(i,MFault_ID) = 4*A; %% replace remaining cells as 'A' %%% Bias of 4
    end
end

end