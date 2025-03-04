function [aera,pnt]=icalimit(A,lim)

%% Function to calculate the control limit for ICA based fault detection indicators.

%% Inputs

%% A = Statistical indicies calculated from the data operating in normal fault free conditions
%% lim = Area limit which is between 1.4865 && 1.4950

%% Outputs

%% Aera = Area of the statistical index using kernel density estimation
%% Pnt = Threshold point occupying 99% of the area

[bw,den,msh,cf] = kde(A,4096); %% 4096 points
%plot(msh,den);
x = msh;y = den;    
aera(1) = 0;
for i = 2:length(y) 
    dx(i-1) = x(i)-x(i-1);
    meanY(i-1) = y(i-1)+y(i)/2;
    aera(i) = aera(i-1)+ (dx(i-1)*meanY(i-1));
end
%[l]=Ind(aera(1.485));
kid = find(abs(aera-lim) < 0.0001); %% find - Finds indices and values of nonzero elements
pnt = x(kid(1));