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

    cvNamedWindow("Camera", CV_WINDOW_AUTOSIZE);

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

        nothing();
        //applyInverse();
        //applyHistory();
        //applyBackground();
        applySkin();

        cvShowImage("Camera", postFrame);

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
    cvDestroyWindow("Camera");

    return;
}

void Gesture::nothing(void) {
    if (firstPass) {
        postFrame = cvCreateImage(cvSize(width, height), depth, channels);
    }
    postFrame = cvCloneImage(frame);
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

    if (firstPass) {
        prevFrame = cvCloneImage(frame);
        postFrame = cvCreateImage(cvSize(width, height), depth, 3);
        // Statistics variables
        // Mean
        vector<Mat> meansT;
        Mat rMeanT = Mat(1, width * height, CV_32F);
        Mat gMeanT = Mat(1, width * height, CV_32F);
        rMeanT = Scalar(R_MEAN);
        gMeanT = Scalar(R_MEAN);
        meansT.push_back(rMeanT);
        meansT.push_back(gMeanT);
        merge(meansT, skinMeanT);
        // Variance
        vector<Mat> varsT;
        static float rVar[2] = {254.0, 0};
        static float gVar[2] = {0, 171.0};
        Mat rVarT = Mat(1, 1, CV_32F, rVar);
        Mat gVarT = Mat(1, 1, CV_32F, gVar);
        varsT.push_back(rVarT);
        varsT.push_back(gVarT);
        merge(varsT, skinVarInv);
//        static float var[2][2] = {{254.0, 0}, {0, 171.0}};
//        skinVarInv = Mat(2, 2, CV_32F, var).clone();
        std::cout << "skinMeanT is size " << skinMeanT.rows << "x" << skinMeanT.cols << "\n";
        std::cout << "skinVarInv is size " << skinVarInv.rows << "x" << skinVarInv.cols << "\n";
        return;
    }
    Mat frameMatrix = cvarrToMat(frame);
    vector<Mat> rgb;
    split(frameMatrix, rgb);
    Mat rFloat, gFloat, bFloat;
    rgb[0].convertTo(rFloat, CV_32F, 1.0, 0);
    rgb[1].convertTo(gFloat, CV_32F, 1.0, 0);
    rgb[2].convertTo(bFloat, CV_32F, 1.0, 0);
    Mat temp1, temp2, temp3, rChrom, gChrom;
    add(rgb[0], rgb[1], temp1);
    add(temp1, rgb[2], temp2);
    add(temp2, Scalar(1.0), temp3);
    divide(rgb[0], temp3, rChrom);
    divide(rgb[1], temp3, gChrom);
    std::cout << "1. gChrom is size " << gChrom.rows << "x" << gChrom.cols <<"x" << gChrom.dims << "\n";
    rChrom = rChrom.reshape(0,1);
    gChrom = gChrom.reshape(0,1);
    std::cout << "2. gChrom is size " << gChrom.rows << "x" << gChrom.cols <<"x" << gChrom.dims << "\n";
    vector<Mat> rgChrom;
    Mat rgT;
    rgChrom.push_back(rChrom);
    rgChrom.push_back(gChrom);
    merge(rgChrom, rgT);
    std::cout << "3. rgChrom is size " << rgChrom.size() << "\n";
    std::cout << "4. rgT is size " << rgT.rows << "x" << rgT.cols <<"x" << rgT.dims << "\n";
    
//    CvMat frameMatrix = cvarrToMat(frame);
//    CvMat *temp1, *temp2, *temp3, *temp4, *temp5;
//    subtract(frameMatrix, skinMean, temp1);
//    transpose(temp1, temp2);
//    multiply(temp2, skinVarInv, temp3);
//    multiply(temp3, temp1, temp4);
//    exp(temp4, postFrame);
    
//    Mat rgT = cvarrToMat(frame).t();
//    Mat convTest;
//    rgT.convertTo(convTest, CV_32F, 1.0 / 255, 0);
//    std::cout << "0. convTest is size " << convTest.rows << "x" << convTest.cols <<"x" << convTest.dims << "\n";
//    std::cout << "1. frame is size " << rgT.rows << "x" << rgT.cols <<"x" << rgT.dims << "\n";
//    vector<Mat> rgb;
//    split(rgT, rgb);
//    rgb.erase(rgb.begin() + 2);
//    merge(rgb, rgT);
//    std::cout << "2. frame is size " << rgT.rows << "x" << rgT.cols <<"x" << rgT.dims << "\n";
//    rgT = rgT.reshape(0,1);
//    std::cout << "3. frame is size " << rgT.rows << "x" << rgT.cols <<"x" << rgT.dims << "\n";
//    
//    
//    ///need to normalize rg
//    // (X-U)'*S^-1*(X-U)
//    Mat temp1, temp2, temp3, temp4, temp5;
//    subtract(rgT, skinMeanT, temp3);
//    transpose(temp3, temp1);
//    std::cout << "4. temp3 is size " << temp3.rows << "x" << temp3.cols <<"x" << temp3.dims << "\n";
//    multiply(temp1, skinVarInv, temp2);
//    std::cout << "5. temp2 is size " << temp2.rows << "x" << temp2.cols <<"x" << temp2.dims << "\n";
//    multiply(temp2, temp3, temp4);
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

