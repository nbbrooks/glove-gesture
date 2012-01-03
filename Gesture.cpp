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
        //        nothing();
        //applyInverse();
        //applyHistory();
        //applyBackground();
//        applySkin();
        applyChRB();

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

void Gesture::applySkin(void) {
    Mat frameMatrix = cvarrToMat(frame);
    // Separate channels into single channel float matrices
    vector<Mat> rgb;
    split(frameMatrix, rgb);
    Mat rFloat, gFloat, bFloat;
    rgb[0].convertTo(rFloat, CV_32FC1, 1.0, 0);
    rgb[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    rgb[2].convertTo(bFloat, CV_32FC1, 1.0, 0);
//    printf("size frameMatrix %d %d %d\n", frameMatrix.size().height, frameMatrix.size().width, frameMatrix.channels());
//    printf("size rFloat %d %d %d\n", rFloat.size().height, rFloat.size().width, rFloat.channels());
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, rGauss, gGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    // (X-U)T
    subtract(rChromV, Scalar(R_CH_MEAN), temp1);
    // * S-1
    multiply(temp1, Scalar(R_CH_VAR_INV), temp2);
    rGauss = temp2.mul(rChromV);
    // * (X-U)
    subtract(gChromV, Scalar(G_CH_MEAN), temp1);
    multiply(temp1, Scalar(G_CH_VAR_INV), temp2);
    gGauss = temp2.mul(gChromV);
    add(rGauss, gGauss, d);
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, outputMatrix);
    outputMatrix = outputMatrix.reshape(0, 480);
//    outheight = outFrame.height;
//    outwidth = outFrame.width;
//    outstep = outFrame.widthStep;
//    outdepth = outFrame.depth;
//        outchannels = outFrame.nChannels;
//    outdata = (uchar *) outFrame.imageData;
//    printf("size outputMatrix %d %d %d\n", outputMatrix.size().height, outputMatrix.size().width, outputMatrix.channels());
//    printf("outFrame height %d width %d channels %d depth %d size %ld\n", outheight, outwidth, outchannels, outdepth, sizeof(outdata[0]));

    CvMat old_matrix = outputMatrix;
    printf("%f\n", CV_MAT_ELEM(old_matrix, float, 240, 320));
//    printf("%d %d\n", height, width);
    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
                if (CV_MAT_ELEM(old_matrix, float, i, j) > 100000.f) {
//                    printf("%f\n", CV_MAT_ELEM(old_matrix, float, i, j));
                    CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
                } else {
                    CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
                }
            }

//    double min, max;
//    minMaxLoc(outputMatrix, &min, &max);
//    minMaxLoc(const SparseMat& src, double* minVal, double* maxVal);
//    printf("min: %f\tmax: %f\n", min, max);

    outFrame = outputMatrix;
    return;
}

void Gesture::applySkin3(void) {
    Mat frameMatrix = cvarrToMat(frame);
    // Separate channels into single channel float matrices
    vector<Mat> rgb;
    split(frameMatrix, rgb);
    Mat rFloat, gFloat, bFloat;
    rgb[0].convertTo(rFloat, CV_32FC1, 1.0, 0);
    rgb[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    rgb[2].convertTo(bFloat, CV_32FC1, 1.0, 0);
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    divide(bFloat, denom, bChrom);
    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // (X-U)T
    subtract(rChromV, Scalar(R_CH_MEAN), temp1);
    // * S-1
    multiply(temp1, Scalar(R_CH_VAR_INV), temp2);
    // * (X-U)
    rGauss = temp2.mul(rChromV);
    // g
    subtract(gChromV, Scalar(G_CH_MEAN), temp1);
    multiply(temp1, Scalar(G_CH_VAR_INV), temp2);
    gGauss = temp2.mul(gChromV);
    add(rGauss, gGauss, d);
    // b
    subtract(bChromV, Scalar(B_CH_MEAN), temp1);
    multiply(temp1, Scalar(B_CH_VAR_INV), temp2);
    bGauss = temp2.mul(bChromV);
    add(d, bGauss, d);
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, outputMatrix);
    outputMatrix = outputMatrix.reshape(0, 480);
    
    CvMat old_matrix = outputMatrix;
    printf("%f\n", CV_MAT_ELEM(old_matrix, float, 240, 320));
    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
        if (CV_MAT_ELEM(old_matrix, float, i, j) > 100000.f) {
            CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
        } else {
            CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
        }
    }
    
    outFrame = outputMatrix;
    return;
}

void Gesture::applyChRB(void) {
    Mat frameMatrix = cvarrToMat(frame);
    // Separate channels into single channel float matrices
    vector<Mat> rgb;
    split(frameMatrix, rgb);
    Mat rFloat, gFloat, bFloat;
    rgb[0].convertTo(rFloat, CV_32FC1, 1.0, 0);
    rgb[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    rgb[2].convertTo(bFloat, CV_32FC1, 1.0, 0);
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    divide(bFloat, denom, bChrom);
    // Compute gaussian probability pixel is on hand
    rChromV = rChrom.reshape(0, 1);
    gChromV = gChrom.reshape(0, 1);
    bChromV = bChrom.reshape(0, 1);
    // r
    subtract(rChromV, Scalar(R_CH_MEAN), temp1);
    multiply(temp1, Scalar(R_CH_VAR_INV), temp2);
    rGauss = temp2.mul(temp1);
    // b
    subtract(bChromV, Scalar(B_CH_MEAN), temp1);
    multiply(temp1, Scalar(B_CH_VAR_INV), temp2);
    bGauss = temp2.mul(temp1);
    
    add(rGauss, bGauss, d);
    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, outputMatrix);
    outputMatrix = outputMatrix.reshape(0, 480);
    
    CvMat old_matrix = outputMatrix;
    printf("%f\n", CV_MAT_ELEM(old_matrix, float, 240, 320));
    for (i = 0; i < height; i++) for (j = 0; j < width; j++) {
        if (CV_MAT_ELEM(old_matrix, float, i, j) > 100000.f) {
            CV_MAT_ELEM(old_matrix, float, i, j) = 255.f;
        } else {
            CV_MAT_ELEM(old_matrix, float, i, j) = 0.f;
        }
    }
    
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

