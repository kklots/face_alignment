function trainData = shape_pca(shapes, meanshape)
shapes = permute(shapes,[3,1,2]);
[ssize, ptsSize, dim] = size(shapes);

if(dim ~= 2)
    fprintf('shape error\n');
end

ptsMat = zeros(ptsSize * 2, ssize);
meanshape = meanshape(:);
dp = meanshape;
pts = zeros(ptsSize, 2);

for i = 1:ssize    
    pts(:, :) = shapes(i, :, :);
    sp = pts;
    
    T = CalcAffineCo(sp, dp);
    
    newpts = pts;
    newpts(:, 1) = pts(:, 1) * T(1, 1) + pts(:, 2) * T(2, 1) + T(3, 1);
    newpts(:, 2) = pts(:, 1) * T(1, 2) + pts(:, 2) * T(2, 2) + T(3, 2);
    
    ptsMat(:, i) = reshape(newpts(:), ptsSize * 2, 1);
end

% meanshape = mean(ptsMat, 2);
% 
% dp = meanshape;

for i = 1:ssize    
    pts(:, :) = shapes(i, :, :);
    sp = pts;
    
    T = CalcAffineCo(sp, dp);
    
    newpts = pts;
    newpts(:, 1) = pts(:, 1) * T(1, 1) + pts(:, 2) * T(2, 1) + T(3, 1);
    newpts(:, 2) = pts(:, 1) * T(1, 2) + pts(:, 2) * T(2, 2) + T(3, 2);
    
    ptsMat(:, i) = reshape(newpts(:), ptsSize * 2, 1);
end

ptsMat = ptsMat - repmat(meanshape, 1, ssize);
varshape = std(ptsMat, 0, 2);
ptsMat = ptsMat./repmat(varshape, 1, ssize);

[coeff, score, latent] = pca(ptsMat');

trainData.meanshape = meanshape;
trainData.latent = latent;
trainData.varshape = varshape;
trainData.coeff = coeff;

end