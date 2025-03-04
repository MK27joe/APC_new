function [Dtest1] = FIntermit(Dtest1,lim,MFault_ID)

m2 = size(Dtest1,1); 

a= lim + 25;
b= lim + 125;
c= m2 - 250;
d= m2 - 150;


for i=1:m2
    if (((i > a) && (i < b)) || ((i > c) && (i < d))) %||((i>1050) && (i<1150)))
        Dtest1(i,MFault_ID) = Dtest1(i,MFault_ID) + 2;
    end
end

end
