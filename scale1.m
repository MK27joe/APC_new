function sx = scale1(x,mon,sd)

%  scale retuns the data with zero mean and unit variance
%  Inputs:
% x ------ Input data
% mx ------mean of input data
% stdx ----Standard input data

% Output:
% rx ---- Scaled data

[m,n] = size(x);
for r=1:n
    XX1=(x(:,r)-mon(:,r))/sd(:,r);
    X1(:,r)=XX1;
end
sx = X1;