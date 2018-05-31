function X = extract_hog_feature(img, pts)
img = single(img);
[ptsSize, dim] = size(pts);
scale = 1;
global orientations
global feat_dim;
global iter;
if(1==iter)
    HALF_SIZE = 12;
    CELLSIZE  = 12;
    featStride = 1;
elseif(2==iter)
    HALF_SIZE = 12;
    CELLSIZE  = 12;
    featStride = 1;
else
    HALF_SIZE = 12;
    CELLSIZE  = 12;
    featStride = 1;
end

if(dim == 1)
    ptsSize = ptsSize / 2;
    pts = reshape(pts, ptsSize, 2);
end
X = zeros(feat_dim, ptsSize);

for i = 1 : ptsSize
    x = pts(i, 1) * scale;
    y = pts(i, 2) * scale;
    
    if(x - HALF_SIZE < 1 ||x + HALF_SIZE-1  > size(img, 2) || y - HALF_SIZE < 1 || y + HALF_SIZE-1 > size(img, 1))
        src_x1 = ceil( max(x-HALF_SIZE,1)); src_y1 = ceil(max(y-HALF_SIZE,1));
        src_x2 = floor(min(x+HALF_SIZE-1, size(img, 2)));src_y2= floor(min(y+HALF_SIZE-1, size(img, 1)));
        dst_x1 = ceil(max(2+HALF_SIZE-x,1));
        dst_y1 = ceil(max(2+HALF_SIZE-y,1));
        roiImg = zeros(HALF_SIZE*2,HALF_SIZE*2,'single');
        roiImg (dst_y1:dst_y1+src_y2-src_y1,dst_x1:dst_x1+src_x2-src_x1)=img(src_y1:src_y2,src_x1:src_x2);
        roiImg = permute(roiImg,[2,1,3]);
        width = size(roiImg,2);
        height = size(roiImg,1);      
    else
        roiImg = img(floor(y)-HALF_SIZE:floor(y)+HALF_SIZE-1, floor(x)-HALF_SIZE:floor(x)+HALF_SIZE-1,:);
        roiImg = permute(roiImg,[2,1,3]);
        width = size(roiImg,2);
        height = size(roiImg,1);
    end
    mode=0;
    numChannels = size(roiImg,3);
    X(:, i) = Mex_feature(orientations, width, height, numChannels,CELLSIZE,featStride,mode,roiImg);
end

X = reshape(X, feat_dim * ptsSize, 1);
X = mapstd(X',0,1)';
end