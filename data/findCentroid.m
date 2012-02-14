function findCentroid(file, figNum)
colorImage = imread(file);
bwImage = im2bw(colorImage, .9);
[r,c,v] = find(bwImage == 0);

m00 = 0;
m10 = 0;
m20 = 0;
m11 = 0;
m01 = 0;
m02 = 0;

for(i=1:size(v))
    m00 = m00 + v(i);
    m10 = m10 + (c(i) * v(i));
    m20 = m20 + (c(i) * c(i) * v(i));
    m11 = m11 + (c(i) * r(i) * v(i));
    m01 = m01 + (r(i) * v(i));
    m02 = m02 + (r(i) * r(i) * v(i));
end
%cx = x pixel of centroid center
%cy = y pixel of centroid center
%l1 = width of centroid
%l2 = height of centroid
%theta = angle from x axis
cx = m10 / m00;
cy = m01 / m00;
a = m20 / m00 - cx * cx;
b = 2 * (m11 / m00 - cx * cy);
c = m02 / m00 - cy * cy;
theta = atan2(b, (a - c)) / 2;
l1 = sqrt(6 * (a + c + sqrt(b ^ 2 + (a - c) ^ 2)));
l2 = sqrt(6 * (a + c - sqrt(b ^ 2 + (a - c) ^ 2)));

fprintf('cx    = %f\n',cx);
fprintf('cy    = %f\n',cy);
fprintf('a     = %f\n',a);
fprintf('b     = %f\n',b);
fprintf('c     = %f\n',c);
fprintf('theta = %f\n',theta);
fprintf('l1    = %f\n',l1);
fprintf('l2    = %f\n',l2);

% Pixel coordinates for [upper-left upper-right lower-right lower-left]
c = [...
    cx - cos(theta) * l1 / 2 + sin(theta) * l2 / 2 ...
    cx + cos(theta) * l1 / 2 + sin(theta) * l2 / 2 ...
    cx + cos(theta) * l1 / 2 - sin(theta) * l2 / 2 ...
    cx - cos(theta) * l1 / 2 - sin(theta) * l2 / 2 ...
    cx - cos(theta) * l1 / 2 + sin(theta) * l2 / 2 ...
    ];
r = [...
    cy - cos(theta) * l2 / 2 - sin(theta) * l1 / 2 ...
    cy - cos(theta) * l2 / 2 + sin(theta) * l1 / 2 ...
    cy + cos(theta) * l2 / 2 + sin(theta) * l1 / 2 ...
    cy + cos(theta) * l2 / 2 - sin(theta) * l1 / 2 ...
    cy - cos(theta) * l2 / 2 - sin(theta) * l1 / 2 ...
    ];
fprintf('U-L = [%d,%d]\n', round(c(1)), round(r(1)));
fprintf('U-R = [%d,%d]\n', round(c(2)), round(r(2)));
fprintf('L-R = [%d,%d]\n', round(c(3)), round(r(3)));
fprintf('L-L = [%d,%d]\n', round(c(4)), round(r(4)));

figure(figNum);

imshow(colorImage);
hold on
for k = 1:numel(b)
    plot(c(:), r(:), 'r', 'Linewidth', 3)
end
hold off

%gImage = rgb2gray(colorImage);
%gImage = im2uint8(bwImage);
%colorImageC = roifill(gImage,c,r);
%imshow(colorImageC);
end
