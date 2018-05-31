function [ meanVector, varVector,w ] = lmsProcess( AllTrainX, AllTrainY )
ptsSize = size(AllTrainY, 1) / 2;
meanVector = mean(AllTrainX, 2);

data = (AllTrainX - repmat(meanVector, 1, size(AllTrainX, 2)));
varVector = zeros(size(meanVector, 1), 1);

for i = 1:size(meanVector, 1)
    varValue = var(data(i, :));
    varVector(i, 1) = sqrt(varValue);
end

TrainX = AllTrainX;
TrainY = AllTrainY;

x1 = zeros(size(TrainX, 1) + 1, ptsSize);
x2 = zeros(size(TrainX, 1) + 1, ptsSize);

A = [TrainX; ones(1, size(TrainX, 2))]';
pinvMat = pinv(A);

for i = 1:ptsSize;
    b1 = TrainY(i, :)';
    b2 = TrainY(i + ptsSize, :)';
    x1(:, i) = pinvMat * b1;
    x2(:, i) = pinvMat * b2;
end

w = [x1 x2];
end

