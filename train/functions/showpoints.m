function img = showpoints(img, points)
ptsSize = size(points, 1);
imshow(img);

for i = 1:ptsSize
    rectangle('Position',[points(i,1)-1,points(i,2)-1,3,3],'FaceColor','g','EdgeColor','g');
end

end