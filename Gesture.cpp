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
    //    arr = {{1,8,12,20,25}, {5,9,13,24,26}};

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
        frame = 0;
        int c;

        frame = cvQueryFrame(capture);
        if (!frame)
            break;
        height = frame->height;
        width = frame->width;
        step = frame->widthStep;
        depth = frame->depth;
        channels = frame->nChannels;
        data = (uchar *) frame->imageData;

        //printf("Processing a %ix%i image with step %i and %i channels and size %ld\n",height,width,step,channels,sizeof(frame)); 

        display = true;
//                nothing();
        applyFlip();
        applyMedian();
        //applyInverse();
        //applyHistory();
        //applyBackground();
//        applyChG();
        applyChRG();
//        applyChRB();
//        applyChRGB();

        if (display) {
            cvShowImage("Input", frame);
            cvShowImage("Output", &outFrame);
        }

        c = cvWaitKey(10);
        if ((char) c == 27) {
            break;
        } else if ((char) c == 's') {
            std::stringstream ss;
            ss << "image-" << save << ".ppm";
            cvSaveImage(ss.str().data(), frame);
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

void Gesture::nothing(void) {
    if (firstPass) {
        //        postFrame = cvCreateImage(cvSize(width, height), depth, channels);
    }
    //    postFrame = cvCloneImage(frame);
    Mat frameMatrix = cvarrToMat(frame);
    outFrame = frameMatrix;
    return;
}

void Gesture::applyFlip(void) {
    inputMatrix = cvarrToMat(frame);
    flip(inputMatrix, inputMatrix, 1);
    return;
}

void Gesture::applyMedian(void) {
//    outputMatrix = cvarrToMat(frame).clone();
//    medianBlur(outputMatrix, outputMatrix, 5);
//    outFrame = outputMatrix;
    inputMatrix = cvarrToMat(frame);
    medianBlur(inputMatrix, inputMatrix, 5);
//    frame = inputMatrix;
    return;
}

void Gesture::applyInverse(void) {
    if (firstPass) {
        postFrame = cvCreateImage(cvSize(width, height), depth, channels);
        postData = (uchar *) postFrame->imageData;
    }
    for (i = 0; i < height; i++) for (j = 0; j < width; j++) for (k = 0; k < channels; k++) {
                postData[i * step + j * channels + k] = 255 - data[i * step + j * channels + k];
            }
    return;
}

void Gesture::applyHistory(void) {
    if (firstPass) {
        procFrame = cvCreateImage(cvSize(width, height), depth, 1);
        postFrame = cvCreateImage(cvSize(width, height), depth, 1);
        //tr=128;
        //bitImage=cvCreateImage(cvSize(frame->width,frame->height),frame->depth,1);
        cvCvtColor(frame, procFrame, CV_BGR2GRAY);
        prevFrame = cvCloneImage(procFrame);
        return;
    }
    cvCvtColor(frame, procFrame, CV_BGR2GRAY);
    cvAbsDiff(prevFrame, procFrame, postFrame);
    //cvThreshold(postFrame,bitImage,tr,255,CV_THRESH_BINARY);
    //cvShowImage("CamSub 1",bitImage);   

    prevFrame = cvCloneImage(procFrame);

    return;
}

void Gesture::applyBackground(void) {

    if (firstPass) {
        //procFrame=cvCreateImage(cvSize(width,height),depth,1);
        postFrame = cvCreateImage(cvSize(width, height), depth, 3);
        //tr=128;
        //bitImage=cvCreateImage(cvSize(frame->width,frame->height),frame->depth,1);
        //cvCvtColor(frame, procFrame, CV_BGR2GRAY);
        prevFrame = cvCloneImage(frame);
        return;
    }
    //cvCvtColor(frame, procFrame,CV_BGR2GRAY);
    cvAbsDiff(frame, prevFrame, postFrame);
    //cvThreshold(postFrame,bitImage,tr,255,CV_THRESH_BINARY);
    //cvShowImage("CamSub 1",bitImage);  

    return;
}

void Gesture::applyChG(void) {
    Mat frameMatrix = cvarrToMat(frame);
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
    printf("%f %f %f\n", 
            rChrom.at<float>(240, 320), gChrom.at<float>(240, 320), bChrom.at<float>(240, 320));

    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // r
    subtract(rChromV, Scalar(R_CH_MEAN), temp1);
    multiply(temp1, Scalar(R_CH_VAR_INV), temp2);
//    rGauss = temp2.mul(temp1);
    // g
    subtract(gChromV, Scalar(G_CH_MEAN), temp1);
    multiply(temp1, Scalar(G_CH_VAR_INV), temp2);
    gGauss = temp2.mul(temp1);
//    add(rGauss, gGauss, d);
    // b
    subtract(bChromV, Scalar(B_CH_MEAN), temp1);
    multiply(temp1, Scalar(B_CH_VAR_INV), temp2);
    bGauss = temp2.mul(temp1);
//    add(d, bGauss, d);
    
    multiply(gGauss, Scalar(-0.5), expTerm);
    exp(expTerm, outputMatrix);
    outputMatrix = outputMatrix.reshape(0, 480);
    
    CvMat in_matrix = frameMatrix;
    CvMat old_matrix = outputMatrix;
    printf("%d %d %d: %f\n", 
            frameMatrix.at<Vec3b>(240, 320)[2], frameMatrix.at<Vec3b>(240, 320)[1], frameMatrix.at<Vec3b>(240, 320)[0], 
            CV_MAT_ELEM(old_matrix, float, 240, 320));
    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
        if (CV_MAT_ELEM(old_matrix, float, i, j) > THRESH) {
            CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
        } else {
            CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
        }
    }
    frameMatrix.at<Vec3b>(240, 320)[2] = 255;
    frameMatrix.at<Vec3b>(240, 320)[1] = 0;
    frameMatrix.at<Vec3b>(240, 320)[0] = 0;
    CV_MAT_ELEM(old_matrix, float, 240, 320) = 128.f;

    outFrame = outputMatrix;
    return;
}

void Gesture::applyChRG(void) {
    Mat frameMatrix = cvarrToMat(frame);
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
    printf("%f %f %f\n", 
            rChrom.at<float>(240, 320), gChrom.at<float>(240, 320), bChrom.at<float>(240, 320));

    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // r
    subtract(rChromV, Scalar(R_CH_MEAN), temp1);
    multiply(temp1, Scalar(R_CH_VAR_INV), temp2);
    rGauss = temp2.mul(temp1);
    // g
    subtract(gChromV, Scalar(G_CH_MEAN), temp1);
    multiply(temp1, Scalar(G_CH_VAR_INV), temp2);
    gGauss = temp2.mul(temp1);
    add(rGauss, gGauss, d);
    // b
    subtract(bChromV, Scalar(B_CH_MEAN), temp1);
    multiply(temp1, Scalar(B_CH_VAR_INV), temp2);
    bGauss = temp2.mul(temp1);
//    add(d, bGauss, d);
    
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, outputMatrix);
    outputMatrix = outputMatrix.reshape(0, 480);
    
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
    threshold(outputMatrix, outputMatrix, THRESH, 255, THRESH_BINARY);

    outFrame = outputMatrix;
    return;
}

void Gesture::applyChRB(void) {
    Mat frameMatrix = cvarrToMat(frame);
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
    printf("%f %f %f\n", 
            rChrom.at<float>(240, 320), gChrom.at<float>(240, 320), bChrom.at<float>(240, 320));

    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // r
    subtract(rChromV, Scalar(R_CH_MEAN), temp1);
    multiply(temp1, Scalar(R_CH_VAR_INV), temp2);
    rGauss = temp2.mul(temp1);
    // g
    subtract(gChromV, Scalar(G_CH_MEAN), temp1);
    multiply(temp1, Scalar(G_CH_VAR_INV), temp2);
    gGauss = temp2.mul(temp1);
//    add(rGauss, gGauss, d);
    // b
    subtract(bChromV, Scalar(B_CH_MEAN), temp1);
    multiply(temp1, Scalar(B_CH_VAR_INV), temp2);
    bGauss = temp2.mul(temp1);
    add(rGauss, bGauss, d);
    
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, outputMatrix);
    outputMatrix = outputMatrix.reshape(0, 480);
    
    CvMat in_matrix = frameMatrix;
    CvMat old_matrix = outputMatrix;
    printf("%d %d %d: %f\n", 
            frameMatrix.at<Vec3b>(240, 320)[2], frameMatrix.at<Vec3b>(240, 320)[1], frameMatrix.at<Vec3b>(240, 320)[0], 
            CV_MAT_ELEM(old_matrix, float, 240, 320));
    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
        if (CV_MAT_ELEM(old_matrix, float, i, j) > THRESH) {
            CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
        } else {
            CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
        }
    }
    frameMatrix.at<Vec3b>(240, 320)[2] = 255;
    frameMatrix.at<Vec3b>(240, 320)[1] = 0;
    frameMatrix.at<Vec3b>(240, 320)[0] = 0;
    CV_MAT_ELEM(old_matrix, float, 240, 320) = 128.f;

    outFrame = outputMatrix;
    return;
}

void Gesture::applyChRGB(void) {
    Mat frameMatrix = cvarrToMat(frame);
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
    printf("%f %f %f\n", 
            rChrom.at<float>(240, 320), gChrom.at<float>(240, 320), bChrom.at<float>(240, 320));

    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // r
    subtract(rChromV, Scalar(R_CH_MEAN), temp1);
    multiply(temp1, Scalar(R_CH_VAR_INV), temp2);
    rGauss = temp2.mul(temp1);
    // g
    subtract(gChromV, Scalar(G_CH_MEAN), temp1);
    multiply(temp1, Scalar(G_CH_VAR_INV), temp2);
    gGauss = temp2.mul(temp1);
    add(rGauss, gGauss, d);
    // b
    subtract(bChromV, Scalar(B_CH_MEAN), temp1);
    multiply(temp1, Scalar(B_CH_VAR_INV), temp2);
    bGauss = temp2.mul(temp1);
    add(d, bGauss, d);
    
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, outputMatrix);
    outputMatrix = outputMatrix.reshape(0, 480);
    
    CvMat in_matrix = frameMatrix;
    CvMat old_matrix = outputMatrix;
    printf("%d %d %d: %f\n", 
            frameMatrix.at<Vec3b>(240, 320)[2], frameMatrix.at<Vec3b>(240, 320)[1], frameMatrix.at<Vec3b>(240, 320)[0], 
            CV_MAT_ELEM(old_matrix, float, 240, 320));
    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
        if (CV_MAT_ELEM(old_matrix, float, i, j) > THRESH) {
            CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
        } else {
            CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
        }
    }
    frameMatrix.at<Vec3b>(240, 320)[2] = 255;
    frameMatrix.at<Vec3b>(240, 320)[1] = 0;
    frameMatrix.at<Vec3b>(240, 320)[0] = 0;
    CV_MAT_ELEM(old_matrix, float, 240, 320) = 128.f;

    outFrame = outputMatrix;
    return;
}

void Gesture::applyTemplate(void) {
}

void Gesture::applyColorSegmentation(void) {
}

void Gesture::calculateCentroid(void) {
}

void Gesture::calculate(void) {
}


// g++ Gesture.cpp -o Gesture -I /usr/local/include/opencv -L /usr/local/lib -lm -lcv -lhighgui -lcvaux

