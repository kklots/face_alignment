function [ ] = writeBin(val0,val1,val2,models0,models1,models2,shp3d,meanpts0,meanpts1,meanpts2)
PTS = [1:68];
for i=1:size(models0,2)
    final_models{1,i}=models0{1,i};
end
final_models{1,size(models0,2)+1} = meanpts0(PTS,:);

for i=1:size(models1,2)
    final_models{2,i}=models1{1,i};
end
final_models{2,size(models1,2)+1} = meanpts1(PTS,:);

for i=1:size(models2,2)
    final_models{3,i}=models2{1,i};
end
final_models{3,size(models2,2)+1} = meanpts2(PTS,:);

final_models{4,1} = shp3d;
final_models{5,1} = val0;
final_models{6,1} = val1;
final_models{7,1} = val2;
write_model('./model/', 'trackingmodel.bin', final_models);
fprintf('\n finished! the model is saved in "./model/"\n');
end

