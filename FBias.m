function [Dtest1] = FBias(Dtest1,lim,MFault_ID,Bias_value)

m2 = size(Dtest1,1); 
for i=1:m2
    if(i>lim)
        Dtest1(i,MFault_ID) = Dtest1(i,MFault_ID) + Bias_value;
    end
end

end

