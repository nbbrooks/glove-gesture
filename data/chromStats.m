function chromStats(file, color)
small=double(imread(file));
i=find(small(:,:,1) ~= 255 & small(:,:,2) ~=255 & small(:,:,3) ~= 255);
[r,c,d]=ind2sub(size(small),i);
rgb=[];
rgbCh=[];
for i=1:size(r,1)
    rgb(i,:)=small(r(i),c(i),:);
    denom = small(r(i),c(i),1) + small(r(i),c(i),2) + small(r(i),c(i),3) + 1;
    rgbCh(i,1)=small(r(i),c(i),1) ./ denom;
    rgbCh(i,2)=small(r(i),c(i),2) ./ denom;
    rgbCh(i,3)=small(r(i),c(i),3) ./ denom;
end
avg=mean(rgbCh);
sdi=1./cov(rgbCh);

fprintf('    static const double R_CH_MEAN_%s = %f;\n', color, avg(1));
fprintf('    static const double G_CH_MEAN_%s = %f;\n', color, avg(2));
fprintf('    static const double B_CH_MEAN_%s = %f;\n', color, avg(3));
fprintf('    static const double R_CH_VAR_INV_%s = %f;\n', color, sdi(1,1));
fprintf('    static const double G_CH_VAR_INV_%s = %f;\n', color, sdi(2,2));
fprintf('    static const double B_CH_VAR_INV_%s = %f;\n', color, sdi(3,3));
end

