clear; close all; clc;
addpath functions;
load shp3d;
[ val0 ] = trainValModel( 'data0' );
[ val1 ] = trainValModel( 'data1' );
[ val2 ] = trainValModel( 'data2' );
[ models0,pcamodels0,meanpts0 ] = train_regressors( 'data0' );
models0=mergeModel(models0,pcamodels0);
[ models1,pcamodels1,meanpts1 ] = train_regressors( 'data1' );
models1=mergeModel(models1,pcamodels1);
[ models2,pcamodels2,meanpts2 ] = train_regressors( 'data2' );
models2=mergeModel(models2,pcamodels2);
writeBin(val0,val1,val2,models0,models1,models2,shp3d,meanpts0,meanpts1,meanpts2);

