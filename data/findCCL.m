function [segImage labelImage labelCorres labelCount labelCountCorres] = findCCL(file, figNum)
colorImage = imread(file);
bwImage = im2bw(colorImage, .9);

figure(figNum);
imshow(bwImage);

height = size(bwImage, 1);
width = size(bwImage, 2);
labelImage = zeros(size(bwImage)); % Label for each pixel in image
labelCount = []; % Count of occurrences of each label for IDing largest CCL
labelCorres = [];
newestLabel = 0; % The latest new label that was created
for(r=2:height)
    for(c=2:width-1)
        neighLabels = [];
        hasNeigh = 0;
        if(bwImage(r, c) == 0)
            if(labelImage(r, c-1) ~= 0) % W
                neighLabels = [neighLabels labelCorres(labelImage(r, c-1))];
                hasNeigh = 1;
            end
            if(labelImage(r-1, c-1) ~= 0) % NW
                neighLabels = [neighLabels labelCorres(labelImage(r-1, c-1))];
                hasNeigh = 1;
            end
            if(labelImage(r-1, c) ~= 0) % N
                neighLabels = [neighLabels labelCorres(labelImage(r-1, c))];
                hasNeigh = 1;
            end
            if(labelImage(r-1, c+1) ~= 0) % NE
                neighLabels = [neighLabels labelCorres(labelImage(r-1, c+1))];
                hasNeigh = 1;
            end
            
            if(hasNeigh == 0)
                % Use new label
                newestLabel = newestLabel + 1;
                labelImage(r, c) = newestLabel;
                % Record correspondence
                labelCorres(newestLabel) = newestLabel;
            else
                % Use smallest label
                minLabel = min(neighLabels);
                labelImage(r, c) = minLabel;
                % Update correspondences
                if(labelImage(r, c-1) ~= 0)
                    labelCorres(labelImage(r, c-1)) = minLabel;
                end
                if(labelImage(r-1, c-1) ~= 0)
                    labelCorres(labelImage(r-1, c-1)) = minLabel;
                end
                if(labelImage(r-1, c) ~= 0)
                    labelCorres(labelImage(r-1, c)) = minLabel;
                end
                if(labelImage(r-1, c+1) ~= 0)
                    labelCorres(labelImage(r-1, c+1)) = minLabel;
                end
            end
            % Update label count
            if(labelImage(r, c) > size(labelCount))
                labelCount(labelImage(r, c)) = 0;
            end
            labelCount(labelImage(r, c)) = labelCount(labelImage(r, c)) + 1;
        end
    end
end
% Finish remapping correspondences
maxDegrees = 0;
for(r=1:size(colorImage,1))
    for(c=1:size(colorImage,2))
        if(labelImage(r,c) ~= 0)
            % While correspondence is more than one degree
            if(labelCorres(labelImage(r,c)) ~= labelCorres(labelCorres(labelImage(r,c))))
                curLabel = labelCorres(labelCorres(labelImage(r,c)));
                degrees = 1;
                while(curLabel ~= labelCorres(curLabel))
                    curLabel = labelCorres(curLabel);
                    degrees = degrees + 1;
                end
                if(degrees > maxDegrees)
                    maxDegrees = degrees;
                end
                labelCorres(labelImage(r,c)) = curLabel;
            end
        end
    end
end
maxDegrees
% Final count of labels
labelCountCorres = zeros(size(labelCount));
for(i=1:size(labelCorres))
    labelCountCorres(labelCorres(i)) = labelCountCorres(labelCorres(i)) + labelCount(i);
end
labelColor = [[1 1 0]; [1 0 1]; [0 1 1]; [1 0 0]; [0 1 0]; [0 0 1]; [0 0 0]];
segImage = ones(size(colorImage));
for(r=1:size(colorImage,1))
    for(c=1:size(colorImage,2))
        if(labelImage(r,c) ~= 0)
            labelImage(r,c);
            labelCorres(labelImage(r,c));
            mod(labelCorres(labelImage(r,c)), size(labelColor, 1)-1) + 1;
            labelColor(mod(labelCorres(labelImage(r,c)), size(labelColor, 1)-1) + 1, :);
            segImage(r, c, :) = labelColor(mod(labelCorres(labelImage(r,c)), size(labelColor, 1)-1) + 1, :);
        end
    end
end
imshow(segImage);


