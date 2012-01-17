function hsvStats(file, color1, color2)
pixels=double(imread(file));
i=find(pixels(:,:,1) ~= 255 & pixels(:,:,2) ~=255 & pixels(:,:,3) ~= 255);
% pixels=rgb2hsv(pixels);
pixels = rgb2hsv(pixels ./ 255.0);
[r,c,d]=ind2sub(size(pixels),i);
hsv1=[];
hsv2=[];

if(color1 == color2)
    for i=1:size(r,1)
        hsv1(i,1)=pixels(r(i),c(i),1);
        hsv1(i,2)=pixels(r(i),c(i),2);
        hsv1(i,3)=pixels(r(i),c(i),3);
    end
    avg1=mean(hsv1);
    sdi1=1./cov(hsv1);
    fprintf('    static const double H_MIN_%s = %f; // (%f)\n', color1, min(hsv1(:,1))*180, min(hsv1(:,1))*360);
    fprintf('    static const double H_MAX_%s = %f; // (%f)\n', color1, max(hsv1(:,1))*180, max(hsv1(:,1))*360);
    fprintf('    static const double S_MIN_%s = %f; // (%f)\n', color1, min(hsv1(:,2))*255, min(hsv1(:,2))*100);
    fprintf('    static const double S_MAX_%s = %f; // (%f)\n', color1, max(hsv1(:,2))*255, max(hsv1(:,2))*100);
    fprintf('    static const double V_MIN_%s = %f; // (%f)\n', color1, min(hsv1(:,3))*255, min(hsv1(:,3))*100);
    fprintf('    static const double V_MAX_%s = %f; // (%f)\n', color1, max(hsv1(:,3))*255, max(hsv1(:,3))*100);
else
    i1=1;
    i2=1;
    for i=1:size(r,1)
        if(pixels(r(i),c(i),1) < 0.5)
            hsv1(i1,1)=pixels(r(i),c(i),1);
            hsv1(i1,2)=pixels(r(i),c(i),2);
            hsv1(i1,3)=pixels(r(i),c(i),3);
            i1=i1+1;
        else
            hsv2(i2,1)=pixels(r(i),c(i),1);
            hsv2(i2,2)=pixels(r(i),c(i),2);
            hsv2(i2,3)=pixels(r(i),c(i),3);
            i2=i2+1;
        end
    end
    avg1=mean(hsv1);
    sdi1=1./cov(hsv1);
    avg2=mean(hsv2);
    sdi2=1./cov(hsv2);
    fprintf('    static const double H_MIN_%s = %f; // (%f)\n', color1, min(hsv1(:,1))*180, min(hsv1(:,1))*360);
    fprintf('    static const double H_MAX_%s = %f; // (%f)\n', color1, max(hsv1(:,1))*180, max(hsv1(:,1))*360);
    fprintf('    static const double S_MIN_%s = %f; // (%f)\n', color1, min(hsv1(:,2))*255, min(hsv1(:,2))*100);
    fprintf('    static const double S_MAX_%s = %f; // (%f)\n', color1, max(hsv1(:,2))*255, max(hsv1(:,2))*100);
    fprintf('    static const double V_MIN_%s = %f; // (%f)\n', color1, min(hsv1(:,3))*255, min(hsv1(:,3))*100);
    fprintf('    static const double V_MAX_%s = %f; // (%f)\n', color1, max(hsv1(:,3))*255, max(hsv1(:,3))*100);
    
    fprintf('    static const double H_MIN_%s = %f; // (%f)\n', color2, min(hsv2(:,1))*180, min(hsv2(:,1))*360);
    fprintf('    static const double H_MAX_%s = %f; // (%f)\n', color2, max(hsv2(:,1))*180, max(hsv2(:,1))*360);
    fprintf('    static const double S_MIN_%s = %f; // (%f)\n', color2, min(hsv2(:,2))*255, min(hsv2(:,2))*100);
    fprintf('    static const double S_MAX_%s = %f; // (%f)\n', color2, max(hsv2(:,2))*255, max(hsv2(:,2))*100);
    fprintf('    static const double V_MIN_%s = %f; // (%f)\n', color2, min(hsv2(:,3))*255, min(hsv2(:,3))*100);
    fprintf('    static const double V_MAX_%s = %f; // (%f)\n', color2, max(hsv2(:,3))*255, max(hsv2(:,3))*100);
end
end