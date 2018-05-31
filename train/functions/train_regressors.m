function [ models,pcamodels,meanpts ] = train_regressors( FOLDER_NAME )
global orientations;
global feat_dim;
orientations = 8;
feat_dim = 4*(3*orientations+1);
PTS = [1:68];
PTS_NUM = size(PTS,2);
FEATURE_LENGTH = feat_dim*PTS_NUM;
ITER_TIMES = 3;
OFF_SET = 13;
SMOOTH_PARAM = 0.8;
disp('calculate shape pca');
imgs_info = dir([FOLDER_NAME,'/*.jpg']);
shapesMat = zeros(PTS_NUM,2,size(imgs_info,1));
for idx=1:size(imgs_info,1)
    load([FOLDER_NAME '/' imgs_info(idx).name(1:end-4) '.mat']);
    shapesMat(:,:,idx)=pts;
    if(mod(idx,1000)==1)
        fprintf('.');
    end
end
fprintf('\n');
meanShape = mean(shapesMat,3);
pcadata = shape_pca(shapesMat,meanShape);

imgMat = cell(size(imgs_info,1),1);
shpMat = zeros(PTS_NUM,2,size(imgs_info,1),'single');
predMat =  zeros(PTS_NUM,2,size(imgs_info,1),'single'); 
AllTrainX = zeros(FEATURE_LENGTH,size(imgs_info,1),'single');
AllTrainY = zeros(PTS_NUM*2,size(imgs_info,1),'single');

for iter = 1:ITER_TIMES
    fprintf(1,'iteration: %d\n',iter);
    disp('  load images and pts');
    meanpts = single(reshape(pcadata.meanshape,68,2)); 
    middle_mean_shape_x = (max(meanpts(:,1))+min(meanpts(:,1)))/2;
    middle_mean_shape_y = (max(meanpts(:,2))+min(meanpts(:,2)))/2;
    count = 0;
    if(1==iter)
        mean_shape = meanpts(PTS,:)*0.25;
        middle_mean_shape_x = middle_mean_shape_x*0.25;
        middle_mean_shape_y = middle_mean_shape_y*0.25;
        for idx = 1:size(imgs_info,1)
            fprintf(1, repmat('\b',1,count));
            count=fprintf(1,'  idx = %d',idx);
            img = imread(['./' FOLDER_NAME './' imgs_info(idx).name]);
            if(3==size(img,3))
                img = rgb2gray(img);
            end
            imgMat{idx,1} =imresize(img,0.25);
            load(['./' FOLDER_NAME './' imgs_info(idx).name(1:end-4) '.mat']);
            pts = pts*0.25;
            shpMat(:,:,idx) = single(pts(PTS,:));
        end
    end
    if(2==iter) 
        mean_shape = meanpts(PTS,:)*0.5;
        middle_mean_shape_x = middle_mean_shape_x*0.5;
        middle_mean_shape_y = middle_mean_shape_y*0.5;
        for idx = 1:size(imgs_info,1)
            fprintf(1, repmat('\b',1,count));
            count=fprintf(1,'  idx = %d',idx);
            img = imread(['./' FOLDER_NAME './' imgs_info(idx).name]);
            if(3==size(img,3))
                img = rgb2gray(img);
            end
            imgMat{idx,1} =imresize(img,0.5);
            load(['./' FOLDER_NAME './' imgs_info(idx).name(1:end-4) '.mat']);
            pts = pts*0.5;
            shpMat(:,:,idx) = single(pts(PTS,:));
        end
    end
    if(3==iter)
        mean_shape = meanpts(PTS,:);
        for idx = 1:size(imgs_info,1)
            fprintf(1, repmat('\b',1,count));
            count=fprintf(1,'  idx = %d',idx);
            img = imread(['./' FOLDER_NAME './' imgs_info(idx).name]);
            if(3==size(img,3))
                img = rgb2gray(img);
            end
            imgMat{idx,1} =img;
            load(['./' FOLDER_NAME './' imgs_info(idx).name(1:end-4) '.mat']);
            shpMat(:,:,idx) = single(pts(PTS,:));
        end
    end
    fprintf('\n');
    disp('  extract hog-like features');
    count = 0;
    for jpg_idx = 1:size(imgs_info,1)
        fprintf(1, repmat('\b',1,count));
        count=fprintf(1,'  jpg_idx = %d',jpg_idx);
        img = imgMat{jpg_idx,1};
        true_shape = shpMat(:,:,jpg_idx);
        if(1==iter)            
            middle_true_shape_x = (max(true_shape(:,1))+min(true_shape(:,1)))/2;
            middle_true_shape_y = (max(true_shape(:,2))+min(true_shape(:,2)))/2;
            x_setoff = OFF_SET*(0.5-rand(1));
            y_setoff = OFF_SET*(0.5-rand(1));
            predMat(:,1,jpg_idx) = mean_shape(:,1) +(middle_true_shape_x-middle_mean_shape_x) + x_setoff;
            predMat(:,2,jpg_idx) = mean_shape(:,2) +(middle_true_shape_y-middle_mean_shape_y) + y_setoff;
       end
        init_shape(:,1) =  predMat(:,1,jpg_idx);
        init_shape(:,2) =  predMat(:,2,jpg_idx);
        AllTrainY(:,jpg_idx) = true_shape(:) - init_shape(:);
        AllTrainX(:,jpg_idx) = extract_hog_feature(img,init_shape);   
    end
    fprintf('\n');
    disp('  calculate feature pca');
    pcamodel = feature_pca( AllTrainX );
    totalLatent = sum(pcamodel.latent);
    for XTH=1:size(pcamodel.latent,1)
        score = sum(pcamodel.latent(1:XTH));
        if(score/totalLatent>SMOOTH_PARAM)
            break;
        end
    end
    disp('  calculate linear regressors');
    pcamodel.coeff = pcamodel.coeff(:,1:XTH);
    newAllTrainX = getPCAfeature(AllTrainX,pcamodel);
    [M, V, W] = lmsProcess( newAllTrainX, AllTrainY );
    models{iter}.M=M;
    models{iter}.V=V;
    models{iter}.W=W;
    pcamodels{iter} = pcamodel;
%     save(['./model_' FOLDER_NAME],'models','pcamodels');
    fprintf(1,'  predict the results of layer: %d for next iteration \n',iter);
    count = 0;
    for jpg_idx = 1:size(imgs_info,1)
        fprintf(1, repmat('\b',1,count));
        count=fprintf(1,'  jpg_idx = %d',jpg_idx);
        img = imgMat{jpg_idx,1};
        pred = reshape(predMat(:,:,jpg_idx),PTS_NUM *2,1);
        pred = predProcess(models, img, pred,iter,pcamodel);
        pred = reshape(pred,PTS_NUM,2);
%         showpoints(img,pred/2);
        predMat(:,:,jpg_idx) = pred;
    end
    fprintf(1,'\n');
end

end

