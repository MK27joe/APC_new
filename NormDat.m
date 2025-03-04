function [X_bar] = NormDat(Data)
    r = size(Data,1); % no. of observations aka. Rows
    xm = mean(Data);
    Sdm = std(Data);
    X_bar = (Data - xm(ones(r,1),:)) ./ (Sdm(ones(r,1),:));
end
