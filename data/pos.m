clear;
small=double(imread('green6.ppm'));
i=find(small(:,:,2) < 255);
[r,c,d]=ind2sub(size(small),i);
t=[];
for i=1:size(r,1)
    rgb(i,:)=small(r(i),c(i),:);
    denom = small(r(i),c(i),1) + small(r(i),c(i),2) + small(r(i),c(i),3) + 1;
    rgbCh(i,1)=small(r(i),c(i),1) ./ denom;
    rgbCh(i,2)=small(r(i),c(i),2) ./ denom;
    rgbCh(i,3)=small(r(i),c(i),3) ./ denom;
end
mean(rgb)
cov(rgb)
mean(rgbCh)
cov(rgbCh)
1./cov(rgbCh)
