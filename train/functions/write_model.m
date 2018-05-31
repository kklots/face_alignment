function write_model(outdir, filename, models)

if(outdir(end) ~= '/')
    outdir = [outdir, '/'];
end
if(~exist(outdir,'dir')) 
    mkdir(outdir);
end
fid = fopen([outdir, filename], 'wb');
stage = size(models,2)-1;
fwrite(fid, stage, 'uint32');
for i = 1:3
    for j=1:size(models,2)-1
        write_matrix(fid, models{i,j}.M);
        write_matrix(fid,  models{i,j}.V);
        write_matrix_w(fid, models{i,j}.W);
    end
    write_matrix(fid, models{i,size(models,2)});
end

write_matrix(fid,models{4,1});
write_matrix(fid, models{5,1}.M);
write_matrix(fid,  models{5,1}.V);
write_matrix_w(fid, models{5,1}.W);
write_matrix(fid, models{6,1}.M);
write_matrix(fid,  models{6,1}.V);
write_matrix_w(fid, models{6,1}.W);
write_matrix(fid, models{7,1}.M);
write_matrix(fid,  models{7,1}.V);
write_matrix_w(fid, models{7,1}.W);
fclose(fid);

end

function write_matrix(fid, mat)
[rows, cols] = size(mat);

fwrite(fid, rows, 'uint32');
fwrite(fid, cols, 'uint32');
minv = min(min(mat));
maxv = max(max(mat));
step = (maxv - minv) / 65535; % (2^16) - 1
mat = uint16((mat - minv) / step);

fwrite(fid, minv, 'single');
fwrite(fid, step, 'single');
fwrite(fid, mat, 'uint16');
end

function write_matrix_w(fid, weights)
[h, w] = size(weights);

fwrite(fid, h, 'uint32');
fwrite(fid, w, 'uint32');

for y = 1:h
    mmax = max(weights(y, :));
    mmin = min(weights(y, :));
    
    step = (mmax - mmin) / 255;
    
    A = [mmin, step];
    fwrite(fid, A, 'single');
    
    A = zeros(1, w);
    
    for x = 1:w
        A(x) = (weights(y, x) - mmin) / step;
    end
    
    A = uint8(A);
    
    fwrite(fid, A, 'uint8');
end
end

function write_matrix_32(fid, mat)
[rows, cols] = size(mat);

fwrite(fid, rows, 'uint32');
fwrite(fid, cols, 'uint32');
minv = min(min(mat));
maxv = max(max(mat));
step = (maxv - minv) / 4294967295; % (2^32) - 1
mat = uint32((mat - minv) / step);

fwrite(fid, minv, 'single');
fwrite(fid, step, 'single');
fwrite(fid, mat, 'uint32');
end

