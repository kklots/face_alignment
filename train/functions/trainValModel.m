function [ validateModel ] = trainValModel( FOLDER_NAME )
addpath('functions');
global orientations;
global feat_dim;
global CELLSIZE;
global HALF_SIZE;

orientations = 8;
feat_dim = 4*(3*orientations+1);
CELLSIZE = 12;
HALF_SIZE = 12;

PTS = [1:68];
PTS_NUM = size(PTS,2);
FEATURE_LENGTH = feat_dim*PTS_NUM;

img_infos = dir(['./' FOLDER_NAME '/*.jpg']);
imgMat = cell(size(img_infos,1),1);
shpMat = zeros(PTS_NUM,2,size(img_infos,1),'single');
AllTrainX = zeros(FEATURE_LENGTH,size(img_infos,1),'single');
AllTrainY = zeros(1,size(img_infos,1),'single');
count = 0;
for idx = 1:size(img_infos,1)
    fprintf(1, repmat('\b',1,count));
    count=fprintf(1,'idx = %d',idx);
    img = imread(['./' FOLDER_NAME '/' img_infos(idx).name]);
    if(3==size(img,3))
        img = rgb2gray(img);
    end
    img=medfilt2(img);
    imgMat{idx,1} =imresize(img,2);
    load(['./' FOLDER_NAME '/' img_infos(idx).name(1:end-4) '.mat']);
    pts = pts*2;
    shpMat(:,:,idx) = single(pts(PTS,:));
end
fprintf(1,'\n');
fprintf(1,'train face validator\n');
count = 0;
for jpg_idx = 1:size(img_infos,1)
    OFF_SET = rand(1)*80;
    fprintf(1, repmat('\b',1,count));
    count=fprintf(1,'jpg_idx = %d',jpg_idx);
    img = imgMat{jpg_idx,1};
    x_setoff = OFF_SET*(0.5-rand(1));
    y_setoff = OFF_SET*(0.5-rand(1));
    true_shape = shpMat(:,:,jpg_idx);
    shape(:,1) = true_shape(:,1)+x_setoff;
    shape(:,2) = true_shape(:,2)+y_setoff;
%         showpoints2(img,shape);
    dist = sqrt(x_setoff^2+y_setoff^2);
    AllTrainY(:,jpg_idx) = min(dist/64,1);
    AllTrainX(:,jpg_idx) = extract_hog_feature(img,shape);   
end
fprintf(1,'\n');
[M, V, W] = lmsProcessForVal( AllTrainX, AllTrainY );
validateModel.M=M;
validateModel.V=V;
validateModel.W=W;
end

