function [Dtest1] = FDrift(Dtest1,lim,MFault_ID,drift_slope_value)

m2 = size(Dtest1,1); 
r = [];
for i = 1:(m2-lim)
    r(i)=i;
end


for i=1:m2
    if(i>lim)
        Dtest1(i,MFault_ID) = Dtest1(i,MFault_ID) + drift_slope_value*r(i-lim);
    end
end

end