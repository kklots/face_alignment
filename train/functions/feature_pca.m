function [ pcamodel ] = feature_pca( AllTrainX )
meanFeature = mean(AllTrainX,2);
AllTrainX = AllTrainX - repmat(meanFeature, 1, size(AllTrainX,2));
varshape = std(AllTrainX, 0, 2);
AllTrainX = AllTrainX./repmat(varshape, 1, size(AllTrainX,2));

[coeff, score, latent] = pca(AllTrainX');
pcamodel.coeff = coeff;
pcamodel.meanfeature = meanFeature;
pcamodel.varshape = varshape;
pcamodel.latent=latent;
end

