function [ newAllTrainX ] = getPCAfeature( AllTrainX,pcamodel)
meanFeature = pcamodel.meanfeature;
AllTrainX = AllTrainX - repmat(meanFeature, 1, size(AllTrainX,2));
varshape = pcamodel.varshape;
AllTrainX = AllTrainX./repmat(varshape, 1, size(AllTrainX,2));
coeff = pcamodel.coeff;
newAllTrainX = AllTrainX' * coeff;
newAllTrainX = newAllTrainX';
end

