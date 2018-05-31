function [ meanVector, varVector,w ] = lmsProcessForVal( AllTrainX, AllTrainY )
meanVector = mean(AllTrainX, 2);
data = (AllTrainX - repmat(meanVector, 1, size(AllTrainX, 2)));
varVector = zeros(size(meanVector, 1), 1);

for i = 1:size(meanVector, 1)
    varValue = var(data(i, :));
    varVector(i, 1) = sqrt(varValue);
end

TrainX = AllTrainX;
TrainX = TrainX - repmat(meanVector, 1, size(AllTrainX, 2));
TrainX = TrainX ./ repmat(varVector, 1, size(AllTrainX, 2));
TrainY = AllTrainY;

A = [TrainX; ones(1, size(TrainX, 2))]';
pinvMat = pinv(A);
b = TrainY(1, :)';
w = pinvMat * b;
end



