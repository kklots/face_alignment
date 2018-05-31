function T = CalcAffineCo(sp,dp)

Y = dp(:);   
xr = sp(:,1)';
yr = sp(:,2)';

spSize = size(sp, 1);

X = zeros(spSize * 2, 4);
for i = 1:spSize
    X(i, :) = [xr(i), -yr(i), 1, 0];
end

for i=1:spSize
    X(i + spSize, :) = [yr(i), xr(i), 0, 1];
end

tT = pinv(X) * Y;
T = [tT(1) tT(2) 0;
    -tT(2) tT(1) 0;
     tT(3) tT(4) 1;];
