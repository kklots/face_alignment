function [ pred ] = predProcess(model, img, pred,layer,pcamodel)
ptsSize = size(pred, 1) / 2;
ptsSize2 = ptsSize * 2;

predRslt = zeros(1, ptsSize2);

feature = extract_hog_feature(img,pred);
meanFeature = pcamodel.meanfeature;
feature = feature - meanFeature;
varshape = pcamodel.varshape;
feature = feature./varshape;
coeff = pcamodel.coeff;
feature = feature' * coeff;
feature = feature';
newfeature = feature;
for i = 1:ptsSize
    predRslt(1, i) = [newfeature' 1] * model{layer}.W(:, i);
    predRslt(1, i + ptsSize)=[newfeature' 1] * model{layer}.W(:, i + ptsSize);
end
if(1==layer)
    pred = (pred + predRslt')*2;
elseif(2==layer)
    pred = (pred + predRslt')*2;
elseif(3==layer)
    pred = pred + predRslt';
end

end





