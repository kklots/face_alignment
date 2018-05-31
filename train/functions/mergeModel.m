function [ models ] = mergeModel( models,pcamodels )
ITERTIMES = size(models,2);
for iter=1:ITERTIMES
     M = pcamodels{1,iter}.meanfeature;
     V = pcamodels{1,iter}.varshape;
     W1 = pcamodels{1,iter}.coeff;
     W2 = models{1,iter}.W;
     W_temp = [W1;zeros(1,size(W1,2))];
     W_temp = [W_temp,zeros(size(W_temp,1),1)];
     W_temp(end,end)=1;
     W_need = W_temp*W2;
     models{iter}.W = W_need;
     models{iter}.V=V;
     models{iter}.M=M;
end
end

