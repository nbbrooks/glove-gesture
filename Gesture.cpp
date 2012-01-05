//#include "cv.h"
//#include "highgui.h"
//#include <stdio.h>
//#include <ctype.h>

#include <sstream>
#include "cv.h"
#include "Gesture.h"
#include "highgui.h"
using namespace cv;

int main(int argc, char** argv) {
    Gesture gesture(argc, argv);
    return 0;
}

Gesture::Gesture(int argc, char** argv) {
    firstPass = true;
    save = 0;
    CvCapture* capture = 0;

    if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        capture = cvCaptureFromCAM(argc == 2 ? argv[1][0] - '0' : 0);
    else if (argc == 2)
        capture = cvCaptureFromAVI(argv[1]);

    if (!capture) {
        fprintf(stderr, "Could not initialize capturing...\n");
        return;
    }

    printf("Hot keys: \n\tESC - quit the program\n");

    cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);

    for (;;) {
        frameImage = 0;
        int c;

        frameImage = cvQueryFrame(capture);
        if (!frameImage)
            break;
        height = frameImage->height;
        width = frameImage->width;
        step = frameImage->widthStep;
        depth = frameImage->depth;
        channels = frameImage->nChannels;
        data = (uchar *) frameImage->imageData;
        frameMatrix = cvarrToMat(frameImage);

        display = true;
        nothing(frameMatrix, outputMatrix);
        applyFlip(frameMatrix, frameMatrix);
        applyMedian(frameMatrix, frameMatrix);
//        applyInverse(frameMatrix, frameMatrix);
//        applyHistory(frameMatrix, prevFrame, outputMatrix);
        applyChRG(frameMatrix, yellowMatrix, R_CH_MEAN_Y, G_CH_MEAN_Y, R_CH_VAR_INV_Y, G_CH_VAR_INV_Y, THRESH_Y);
        applyChRG(frameMatrix, greenMatrix, R_CH_MEAN_G, G_CH_MEAN_G, R_CH_VAR_INV_G, G_CH_VAR_INV_G, THRESH_G);
//        applyChRB(frameMatrix, yellowMatrix, R_CH_MEAN_Y, B_CH_MEAN_Y, R_CH_VAR_INV_Y, B_CH_VAR_INV_Y, THRESH_Y);
//        applyChRGB(frameMatrix, yellowMatrix, R_CH_MEAN_Y, G_CH_MEAN_Y, B_CH_MEAN_Y, R_CH_VAR_INV_Y, G_CH_VAR_INV_Y, B_CH_VAR_INV_Y, THRESH_Y);

        Mat tempMatrix(greenMatrix.rows, greenMatrix.cols, CV_32FC1, Scalar(0));
        vector<Mat> rgb;
        // BGR ordering!!!
        rgb.push_back(tempMatrix);
        rgb.push_back(greenMatrix);
        rgb.push_back(yellowMatrix);
        merge(rgb, outputMatrix);
        
        outImage = outputMatrix;
//        printf("output: %d %d %d %d\n", outImage.height, outImage.width, outImage.nChannels, outImage.depth);
        
        if (display) {
            cvShowImage("Input", frameImage);
            cvShowImage("Output", &outImage);
        }

        c = cvWaitKey(10);
        if ((char) c == 27) {
            break;
        } else if ((char) c == 's') {
            std::stringstream ss;
            ss << "image-" << save << ".ppm";
            cvSaveImage(ss.str().data(), frameImage);
            save++;
        }

        if (firstPass) {
            firstPass = false;
        }
//        sleep(1);
    }

    cvReleaseCapture(&capture);
    cvDestroyWindow("Input");
    cvDestroyWindow("Output");

    return;
}

void Gesture::nothing(const Mat& src, Mat& dst) {
//    src.copyTo(dst);
//    dst = src.clone();
    dst = src;
    return;
}

void Gesture::applyFlip(const Mat& src, Mat& dst) {
    flip(src, dst, 1);
    return;
}

void Gesture::applyMedian(const Mat& src, Mat& dst) {
    inputMatrix = cvarrToMat(frameImage);
    medianBlur(inputMatrix, inputMatrix, 5);
    return;
}

void Gesture::applyInverse(const Mat& src, Mat& dst) {
    if (firstPass) {
        postFrame = cvCreateImage(cvSize(width, height), depth, channels);
        postData = (uchar *) postFrame->imageData;
    }
    for (i = 0; i < height; i++) for (j = 0; j < width; j++) for (k = 0; k < channels; k++) {
                postData[i * step + j * channels + k] = 255 - data[i * step + j * channels + k];
            }
    return;
}

void Gesture::applyHistory(const Mat& src, Mat& prev, Mat& dst) {
    if (firstPass) {
        procFrame = cvCreateImage(cvSize(width, height), depth, 1);
        postFrame = cvCreateImage(cvSize(width, height), depth, 1);
        //tr=128;
        //bitImage=cvCreateImage(cvSize(frame->width,frame->height),frame->depth,1);
        cvCvtColor(frameImage, procFrame, CV_BGR2GRAY);
        prevFrame = cvCloneImage(procFrame);
        return;
    }
    cvCvtColor(frameImage, procFrame, CV_BGR2GRAY);
    cvAbsDiff(prevFrame, procFrame, postFrame);
    //cvThreshold(postFrame,bitImage,tr,255,CV_THRESH_BINARY);
    //cvShowImage("CamSub 1",bitImage);   

    prevFrame = cvCloneImage(procFrame);

    return;
}

void Gesture::applyChRG(const Mat& src, Mat& dst, double rMean, double gMean, double rSDI, double gSDI, double thresh) {
    // Separate channels into single channel float matrices
    vector<Mat> rgb;
    split(src, rgb);
    Mat rFloat, gFloat, bFloat;
    // BGR ordering
    rgb[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
    rgb[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    rgb[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    divide(bFloat, denom, bChrom);
//    printf("%f %f %f\n", 
//            rChrom.at<float>(240, 320), gChrom.at<float>(240, 320), bChrom.at<float>(240, 320));

    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // r
    subtract(rChromV, Scalar(rMean), temp1);
    multiply(temp1, Scalar(rSDI), temp2);
    rGauss = temp2.mul(temp1);
    // g
    subtract(gChromV, Scalar(gMean), temp1);
    multiply(temp1, Scalar(gSDI), temp2);
    gGauss = temp2.mul(temp1);
    add(rGauss, gGauss, d);
    
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, dst);
    dst = dst.reshape(0, 480);
    
//    CvMat in_matrix = frameMatrix;
//    CvMat old_matrix = outputMatrix;
//    printf("%d %d %d: %f\n", 
//            frameMatrix.at<Vec3b>(240, 320)[2], frameMatrix.at<Vec3b>(240, 320)[1], frameMatrix.at<Vec3b>(240, 320)[0], 
//            CV_MAT_ELEM(old_matrix, float, 240, 320));
//    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
//        if (CV_MAT_ELEM(old_matrix, float, i, j) > THRESH) {
//            CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
//        } else {
//            CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
//        }
//    }
//    frameMatrix.at<Vec3b>(240, 320)[2] = 255;
//    frameMatrix.at<Vec3b>(240, 320)[1] = 0;
//    frameMatrix.at<Vec3b>(240, 320)[0] = 0;
//    CV_MAT_ELEM(old_matrix, float, 240, 320) = 128.f;
    threshold(dst, dst, thresh, 255, THRESH_BINARY);

    return;
}

void Gesture::applyChRB(const Mat& src, Mat& dst, double rMean, double bMean, double rSDI, double bSDI, double thresh) {
    // Separate channels into single channel float matrices
    vector<Mat> rgb;
    split(src, rgb);
    Mat rFloat, gFloat, bFloat;
    // BGR ordering
    rgb[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
    rgb[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    rgb[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    divide(bFloat, denom, bChrom);
//    printf("%f %f %f\n", 
//            rChrom.at<float>(240, 320), gChrom.at<float>(240, 320), bChrom.at<float>(240, 320));

    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // r
    subtract(rChromV, Scalar(rMean), temp1);
    multiply(temp1, Scalar(rSDI), temp2);
    rGauss = temp2.mul(temp1);
    // b
    subtract(bChromV, Scalar(bMean), temp1);
    multiply(temp1, Scalar(bSDI), temp2);
    bGauss = temp2.mul(temp1);
    add(rGauss, bGauss, d);
    
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, dst);
    dst = dst.reshape(0, 480);
    
//    CvMat in_matrix = frameMatrix;
//    CvMat old_matrix = outputMatrix;
//    printf("%d %d %d: %f\n", 
//            frameMatrix.at<Vec3b>(240, 320)[2], frameMatrix.at<Vec3b>(240, 320)[1], frameMatrix.at<Vec3b>(240, 320)[0], 
//            CV_MAT_ELEM(old_matrix, float, 240, 320));
//    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
//        if (CV_MAT_ELEM(old_matrix, float, i, j) > THRESH_Y) {
//            CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
//        } else {
//            CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
//        }
//    }
//    frameMatrix.at<Vec3b>(240, 320)[2] = 255;
//    frameMatrix.at<Vec3b>(240, 320)[1] = 0;
//    frameMatrix.at<Vec3b>(240, 320)[0] = 0;
//    CV_MAT_ELEM(old_matrix, float, 240, 320) = 128.f;
    threshold(dst, dst, thresh, 255, THRESH_BINARY);

    return;
}

void Gesture::applyChRGB(const Mat& src, Mat& dst, double rMean, double gMean, double bMean, double rSDI, double gSDI, double bSDI, double thresh) {
    Mat frameMatrix = cvarrToMat(frameImage);
    // Separate channels into single channel float matrices
    vector<Mat> rgb;
    split(frameMatrix, rgb);
    Mat rFloat, gFloat, bFloat;
    // BGR ordering
    rgb[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
    rgb[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    rgb[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    divide(bFloat, denom, bChrom);
//    printf("%f %f %f\n", 
//            rChrom.at<float>(240, 320), gChrom.at<float>(240, 320), bChrom.at<float>(240, 320));

    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // r
    subtract(rChromV, Scalar(rMean), temp1);
    multiply(temp1, Scalar(rSDI), temp2);
    rGauss = temp2.mul(temp1);
    // g
    subtract(gChromV, Scalar(gMean), temp1);
    multiply(temp1, Scalar(gSDI), temp2);
    gGauss = temp2.mul(temp1);
    add(rGauss, gGauss, d);
    // b
    subtract(bChromV, Scalar(bMean), temp1);
    multiply(temp1, Scalar(bSDI), temp2);
    bGauss = temp2.mul(temp1);
    add(d, bGauss, d);
    
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, dst);
    dst = dst.reshape(0, 480);
    
//    CvMat in_matrix = frameMatrix;
//    CvMat old_matrix = outputMatrix;
//    printf("%d %d %d: %f\n", 
//            frameMatrix.at<Vec3b>(240, 320)[2], frameMatrix.at<Vec3b>(240, 320)[1], frameMatrix.at<Vec3b>(240, 320)[0], 
//            CV_MAT_ELEM(old_matrix, float, 240, 320));
//    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
//        if (CV_MAT_ELEM(old_matrix, float, i, j) > THRESH_Y) {
//            CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
//        } else {
//            CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
//        }
//    }
//    frameMatrix.at<Vec3b>(240, 320)[2] = 255;
//    frameMatrix.at<Vec3b>(240, 320)[1] = 0;
//    frameMatrix.at<Vec3b>(240, 320)[0] = 0;
//    CV_MAT_ELEM(old_matrix, float, 240, 320) = 128.f;
    threshold(dst, dst, thresh, 255, THRESH_BINARY);

    return;
}