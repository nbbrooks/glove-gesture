function hsvStats(file, color)
small=double(imread(file));
i=find(small(:,:,1) ~= 255 & small(:,:,2) ~=255 & small(:,:,3) ~= 255);
% small=rgb2hsv(small);
small = rgb2hsv(small ./ 255.0);
[r,c,d]=ind2sub(size(small),i);
hsv=[];
for i=1:size(r,1)
    hsv(i,1)=small(r(i),c(i),1);
    hsv(i,2)=small(r(i),c(i),2);
    hsv(i,3)=small(r(i),c(i),3);
end
avg=mean(hsv);
sdi=1./cov(hsv);

fprintf('    static const double H_MIN_%s = %f;\n', color, min(hsv(:,1))*180);
fprintf('    static const double H_MAX_%s = %f;\n', color, max(hsv(:,1))*180);
fprintf('    static const double S_MIN_%s = %f;\n', color, min(hsv(:,2))*255);
fprintf('    static const double S_MAX_%s = %f;\n', color, max(hsv(:,2))*255);
fprintf('    static const double V_MIN_%s = %f;\n', color, min(hsv(:,3))*255);
fprintf('    static const double V_MAX_%s = %f;\n', color, max(hsv(:,3))*255);
end