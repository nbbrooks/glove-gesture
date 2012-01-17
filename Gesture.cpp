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
    int c;
    std::cout << "argc is " << argc << std::endl;

    if (argc == 2) {
        // Process image
        cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);
        cvNamedWindow("Process", CV_WINDOW_AUTOSIZE);
        cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);

        std::cout << "File is " << argv[1] << std::endl;
        frameImage = cvLoadImage(argv[1]);
        frameMatrix = cvarrToMat(frameImage);
        cvShowImage("Input", frameImage);

        // Load binary template
        templateMatrix = imread("template-32.ppm", 0);

        GaussianBlur(frameMatrix, frameMatrix, Size(9, 9), 1, 1);
        cvtColor(frameMatrix, hsvMatrix, CV_BGR2HSV);
        applyTableHSV(frameMatrix, greenMatrix, H_MIN_G, H_MAX_G, S_MIN_G, S_MAX_G, V_MIN_G, V_MAX_G);
        applyTableHSV(frameMatrix, tempMatrix, H_MIN_R1, H_MAX_R1, S_MIN_R1, S_MAX_R1, V_MIN_R1, V_MAX_R1);
        applyTableHSV(frameMatrix, redMatrix, H_MIN_R2, H_MAX_R2, S_MIN_R2, S_MAX_R2, V_MIN_R2, V_MAX_R2);
        add(tempMatrix, redMatrix, redMatrix);

        Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, greenMatrix.type(), Scalar(0));
        vector<Mat> bgr;
        bgr.push_back(blueMatrix);
        bgr.push_back(greenMatrix);
        bgr.push_back(redMatrix);
        merge(bgr, tempMatrix);
        printInfo(tempMatrix);
        printInfo(redMatrix);
        printInfo(templateMatrix);
        tempImage = greenMatrix;
//        templateCircles(greenMatrix, outputMatrix, frameMatrix, greenMatrix, templateMatrix, THRESH_TEMPLATE_48);
        templateCircles(redMatrix, outputMatrix, frameMatrix, redMatrix, templateMatrix, THRESH_TEMPLATE_32);
        //        houghCircles(redMatrix, outputMatrix, frameMatrix, templateMatrix);
        outImage = outputMatrix;

        cvShowImage("Input", frameImage);
        cvShowImage("Process", &tempImage);
        cvShowImage("Output", &outImage);
        c = cvWaitKey();
        while ((char) c != 27) {
            c = cvWaitKey();
        }
        cvDestroyWindow("Input");
        cvDestroyWindow("Process");
        cvDestroyWindow("Output");
        return;
    }

    // Process video
    firstPass = true;
    save = 0;
    CvCapture* capture = 0;

    if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        capture = cvCaptureFromCAM(argc == 2 ? argv[1][0] - '0' : 1);
    else if (argc == 2)
        capture = cvCaptureFromAVI(argv[1]);

    if (!capture) {
        fprintf(stderr, "Could not initialize capturing...\n");
        return;
    }

    fprintf(stderr, "CV_8UC1 is %d\n", CV_8UC1);
    fprintf(stderr, "CV_8UC2 is %d\n", CV_8UC2);
    fprintf(stderr, "CV_8UC3 is %d\n", CV_8UC3);
    fprintf(stderr, "CV_8UC4 is %d\n", CV_8UC4);
    fprintf(stderr, "\n");
    fprintf(stderr, "CV_8SC1 is %d\n", CV_8SC1);
    fprintf(stderr, "CV_8SC2 is %d\n", CV_8SC2);
    fprintf(stderr, "CV_8SC3 is %d\n", CV_8SC3);
    fprintf(stderr, "CV_8SC4 is %d\n", CV_8SC4);
    fprintf(stderr, "\n");
    fprintf(stderr, "CV_16UC1 is %d\n", CV_16UC1);
    fprintf(stderr, "CV_16UC2 is %d\n", CV_16UC2);
    fprintf(stderr, "CV_16UC3 is %d\n", CV_16UC3);
    fprintf(stderr, "CV_16UC4 is %d\n", CV_16UC4);
    fprintf(stderr, "\n");
    fprintf(stderr, "CV_16SC1 is %d\n", CV_16SC1);
    fprintf(stderr, "CV_16SC2 is %d\n", CV_16SC2);
    fprintf(stderr, "CV_16SC3 is %d\n", CV_16SC3);
    fprintf(stderr, "CV_16SC4 is %d\n", CV_16SC4);
    fprintf(stderr, "\n");
    fprintf(stderr, "CV_32SC1 is %d\n", CV_32SC1);
    fprintf(stderr, "CV_32SC2 is %d\n", CV_32SC2);
    fprintf(stderr, "CV_32SC3 is %d\n", CV_32SC3);
    fprintf(stderr, "CV_32SC4 is %d\n", CV_32SC4);
    fprintf(stderr, "\n");
    fprintf(stderr, "CV_32FC1 is %d\n", CV_32FC1);
    fprintf(stderr, "CV_32FC2 is %d\n", CV_32FC2);
    fprintf(stderr, "CV_32FC3 is %d\n", CV_32FC3);
    fprintf(stderr, "CV_32FC4 is %d\n", CV_32FC4);
    fprintf(stderr, "\n");
    fprintf(stderr, "CV_64FC1 is %d\n", CV_64FC1);
    fprintf(stderr, "CV_64FC2 is %d\n", CV_64FC2);
    fprintf(stderr, "CV_64FC3 is %d\n", CV_64FC3);
    fprintf(stderr, "CV_64FC4 is %d\n", CV_64FC4);

    fprintf(stderr, "Hot keys: \n\tESC - quit the program\n");
    fprintf(stderr, "\ts   - save current input frame\n");

    cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Process", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);
    std::stringstream debug;
    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);

    // Load binary template
    templateMatrix = imread("template2.ppm", 0);

    for (;;) {
        frameImage = 0;

        frameImage = cvQueryFrame(capture);
        if (!frameImage)
            break;
        frameMatrix = cvarrToMat(frameImage);
        printInfo(frameMatrix);

        display = true;
        //        nothing(frameMatrix, outputMatrix);
        applyFlip(frameMatrix, frameMatrix);
        cvtColor(frameMatrix, hsvMatrix, CV_BGR2HSV);
        //        applyMedian(frameMatrix, frameMatrix);
        //        applyInverse(frameMatrix, frameMatrix);
        //        applyHistory(frameMatrix, prevFrame, outputMatrix);
        //        applyChRGB(frameMatrix, redMatrix, R_CH_MEAN_R, G_CH_MEAN_R, B_CH_MEAN_R, R_CH_VAR_INV_R, G_CH_VAR_INV_R, B_CH_VAR_INV_R, THRESH_R);
        //        applyChRGB(frameMatrix, greenMatrix, R_CH_MEAN_G, G_CH_MEAN_G, B_CH_MEAN_G, R_CH_VAR_INV_G, G_CH_VAR_INV_G, B_CH_VAR_INV_G, THRESH_G);
        //        applyChRGB(frameMatrix, blueMatrix, R_CH_MEAN_B, G_CH_MEAN_B, B_CH_MEAN_B, R_CH_VAR_INV_B, G_CH_VAR_INV_B, B_CH_VAR_INV_B, THRESH_B);
        //        applyChRB(frameMatrix, redMatrix, R_CH_MEAN_R, B_CH_MEAN_R, R_CH_VAR_INV_R, B_CH_VAR_INV_R, THRESH_R);
        //        applyChRB(frameMatrix, blueMatrix, R_CH_MEAN_B, B_CH_MEAN_B, R_CH_VAR_INV_B, B_CH_VAR_INV_B, THRESH_B);
        //        applyHSV(frameMatrix, redMatrix, H_MEAN_R, S_MEAN_R, V_MEAN_R, H_VAR_INV_R, S_VAR_INV_R, V_VAR_INV_R, THRESH_H);
        //        applyHSV(frameMatrix, greenMatrix, H_MEAN_G, S_MEAN_G, V_MEAN_G, H_VAR_INV_G, S_VAR_INV_G, V_VAR_INV_G, THRESH_S);
        //        applyHSV(frameMatrix, blueMatrix, H_MEAN_B, S_MEAN_B, V_MEAN_B, H_VAR_INV_B, S_VAR_INV_B, V_VAR_INV_B, THRESH_V);
        //        applyTableHSV(frameMatrix, blueMatrix, H_MIN_B, H_MAX_B, S_MIN_B, S_MAX_B, V_MIN_B, V_MAX_B);
        applyTableHSV(frameMatrix, greenMatrix, H_MIN_G, H_MAX_G, S_MIN_G, S_MAX_G, V_MIN_G, V_MAX_G);
        applyTableHSV(frameMatrix, tempMatrix, H_MIN_R1, H_MAX_R1, S_MIN_R1, S_MAX_R1, V_MIN_R1, V_MAX_R1);
        applyTableHSV(frameMatrix, redMatrix, H_MIN_R2, H_MAX_R2, S_MIN_R2, S_MAX_R2, V_MIN_R2, V_MAX_R2);
        add(tempMatrix, redMatrix, redMatrix);

        //        Mat redMatrix(blueMatrix.rows, blueMatrix.cols, blueMatrix.type(), Scalar(0));
        //        Mat greenMatrix(blueMatrix.rows, blueMatrix.cols, blueMatrix.type(), Scalar(0));
        Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, greenMatrix.type(), Scalar(0));
        vector<Mat> bgr;
        // BGR ordering!!!
        //        fprintf(stderr, "redMatrix: %d %d %d %d\n", redMatrix.rows, redMatrix.cols, redMatrix.depth(), redMatrix.type());
        //        fprintf(stderr, "greenMatrix: %d %d %d %d\n", greenMatrix.rows, greenMatrix.cols, greenMatrix.depth(), greenMatrix.type());
        //        fprintf(stderr, "blueMatrix: %d %d %d %d\n", blueMatrix.rows, blueMatrix.cols, blueMatrix.depth(), blueMatrix.type());
        bgr.push_back(blueMatrix);
        bgr.push_back(greenMatrix);
        bgr.push_back(redMatrix);
        merge(bgr, tempMatrix);
//        templateCircles(greenMatrix, outputMatrix, frameMatrix, greenMatrix, templateMatrix, THRESH_TEMPLATE_48);
        templateCircles(redMatrix, outputMatrix, frameMatrix, redMatrix, templateMatrix, THRESH_TEMPLATE_48);
        //        houghCircles(redMatrix, outputMatrix, frameMatrix, templateMatrix);

        debug.str("");
        debug << (int) (hsvMatrix.at<Vec3b > (240, 320)[0]) << "," <<
                (int) (hsvMatrix.at<Vec3b > (240, 320)[1]) << "," <<
                (int) (hsvMatrix.at<Vec3b > (240, 320)[2]);
        putText(frameMatrix, debug.str(), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
        putText(tempMatrix, debug.str(), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
        debug.str("");
        debug << (int) frameMatrix.at<Vec3b > (240, 320)[2] << "," <<
                (int) frameMatrix.at<Vec3b > (240, 320)[1] << "," <<
                (int) frameMatrix.at<Vec3b > (240, 320)[0];
        putText(frameMatrix, debug.str(), Point(0, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
        putText(tempMatrix, debug.str(), Point(0, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
        cross(frameMatrix, 240, 320, 1);
        cross(tempMatrix, 240, 320, 1);

        tempImage = tempMatrix;
        outImage = outputMatrix;

        //        fprintf(stderr, "output: %d %d %d %d\n", outImage.height, outImage.width, outImage.nChannels, outImage.depth);

        if (display) {
            cvShowImage("Input", frameImage);
            cvShowImage("Process", &tempImage);
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
    cvDestroyWindow("Process");
    cvDestroyWindow("Output");

    return;
}

void Gesture::nothing(const Mat& src, Mat& dst) {
    dst = src;
    return;
}

void Gesture::applyFlip(const Mat& src, Mat& dst) {
    flip(src, dst, 1);
    return;
}

void Gesture::applyMedian(const Mat& src, Mat& dst) {
    medianBlur(src, dst, 5);
    return;
}

void Gesture::applyInverse(const Mat& src, Mat& dst) {
    int width = src.cols;
    int height = src.rows;
    int channels = src.channels();
    int depth = src.depth();
    int step = src.step;
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
    int width = src.cols;
    int height = src.rows;
    int depth = src.depth();
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
    vector<Mat> bgr;
    split(src, bgr);
    Mat rFloat, gFloat, bFloat;
    // BGR ordering
    bgr[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
    bgr[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    bgr[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    divide(bFloat, denom, bChrom);
    //    fprintf(stderr, "%f %f %f\n", 
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
    //    fprintf(stderr, "%d %d %d: %f\n", 
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
    vector<Mat> bgr;
    split(src, bgr);
    Mat rFloat, gFloat, bFloat;
    // BGR ordering
    bgr[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
    bgr[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    bgr[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    divide(bFloat, denom, bChrom);
    //    fprintf(stderr, "%f %f %f\n", 
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
    //    fprintf(stderr, "%d %d %d: %f\n", 
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
    // Separate channels into single channel float matrices
    vector<Mat> bgr;
    split(src, bgr);
    Mat rFloat, gFloat, bFloat;
    // BGR ordering
    bgr[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
    bgr[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
    bgr[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
    // Compute chromacity for r and g
    Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
    add(rFloat, gFloat, temp1);
    add(temp1, bFloat, temp2);
    add(temp2, Scalar(1.0), denom);
    divide(rFloat, denom, rChrom);
    divide(gFloat, denom, gChrom);
    divide(bFloat, denom, bChrom);
    //    fprintf(stderr, "%f %f %f\n", 
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
    //    fprintf(stderr, "%d %d %d: %f\n", 
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

void Gesture::applyGaussHSV(const Mat& src, Mat& dst, double hMean, double sMean, double vMean, double hSDI, double sSDI, double vSDI, double thresh) {
    // Separate channels into single channel float matrices
    cvtColor(src, dst, CV_BGR2HSV);
    vector<Mat> hsv;
    split(dst, hsv);
    Mat hFloat, sFloat, vFloat;
    hsv[0].convertTo(hFloat, CV_32FC1, 1.0, 0);
    hsv[1].convertTo(sFloat, CV_32FC1, 1.0, 0);
    hsv[2].convertTo(vFloat, CV_32FC1, 1.0, 0);

    // Compute gaussian probability pixel is on hand
    Mat hV, sV, vV, temp1, temp2, hGauss, sGauss, vGauss, d, expTerm;
    hV = hFloat.reshape(0, 1);
    sV = sFloat.reshape(0, 1);
    vV = vFloat.reshape(0, 1);
    // r
    subtract(hV, Scalar(hMean), temp1);
    multiply(temp1, Scalar(hSDI), temp2);
    hGauss = temp2.mul(temp1);
    // g
    subtract(sV, Scalar(sMean), temp1);
    multiply(temp1, Scalar(sSDI), temp2);
    sGauss = temp2.mul(temp1);
    add(hGauss, sGauss, d);
    // b
    subtract(vV, Scalar(vMean), temp1);
    multiply(temp1, Scalar(vSDI), temp2);
    vGauss = temp2.mul(temp1);
    add(d, vGauss, d);

    multiply(d, Scalar(-0.5), expTerm);
    exp(expTerm, dst);
    dst = dst.reshape(0, 480);
    threshold(dst, dst, thresh, 255, THRESH_BINARY);

    return;
}

void Gesture::applyTableHSV(const Mat& src, Mat& dst, double hMin, double hMax, double sMin, double sMax, double vMin, double vMax) {
    // Separate channels into single channel float matrices
    cvtColor(src, dst, CV_BGR2HSV);
    vector<Mat> hsv;
    split(dst, hsv);
    //    Mat hFloat, sFloat, vFloat;
    //    hsv[0].convertTo(hFloat, CV_32FC1, 1.0, 0);
    //    hsv[1].convertTo(sFloat, CV_32FC1, 1.0, 0);
    //    hsv[2].convertTo(vFloat, CV_32FC1, 1.0, 0);

    Mat hMinT, hMaxT, sMinT, sMaxT, vMinT, vMaxT;
    // Apply min
    threshold(hsv[0], hMinT, hMin, 255, THRESH_BINARY);
    threshold(hsv[1], sMinT, sMin, 255, THRESH_BINARY);
    threshold(hsv[2], vMinT, vMin, 255, THRESH_BINARY);

    // Apply max
    threshold(hsv[0], hMaxT, hMax, 255, THRESH_BINARY_INV);
    threshold(hsv[1], sMaxT, sMax, 255, THRESH_BINARY_INV);
    threshold(hsv[2], vMaxT, vMax, 255, THRESH_BINARY_INV);

    Mat hT, sT, vT;
    // OR the min and max
    //    bitwise_and(vMinT, vMaxT, dst);
    bitwise_and(hMinT, hMaxT, hT);
    bitwise_and(sMinT, sMaxT, sT);
    bitwise_and(vMinT, vMaxT, vT);
    // AND the OR results
    bitwise_and(hT, sT, dst);
    //    bitwise_and(dst, vT, dst);

    return;
}

void Gesture::templateCircles(const Mat& src, Mat& dst, Mat& drawMatrix, Mat& procMatrix, Mat& templ, double thresh) {
    int count = 0;
//    GaussianBlur(src, dst, Size(9, 9), 1, 1);
    //    medianBlur(src, dst, 5);

    /// Do the Matching and Normalize
    Mat srcNorm, templateNorm;
    src.convertTo(srcNorm, CV_32FC1, 1.0 / 255.0, 0);
    templ.convertTo(templateNorm, CV_32FC1, 1.0 / 255.0, 0);
//    normalize(src, srcNorm, 0, 1, NORM_MINMAX, -1, Mat());
//    normalize(templ, templateNorm, 0, 1, NORM_MINMAX, -1, Mat());
    matchTemplate(srcNorm, templateNorm, dst, CV_TM_SQDIFF);
    fprintf(stderr, "Template\n");
    printInfo(src);
    printInfo(srcNorm);
    printInfo(templ);
    printInfo(templateNorm);
    printInfo(dst);
//    normalize(dst, dst, 0, 1, NORM_MINMAX, -1, Mat());

    /// Localizing the best match with minMaxLoc
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;

    Mat minMaxMat = dst.clone();
    normalize(dst, dst, 0, 1, NORM_MINMAX, -1, Mat());
    minMaxLoc(minMaxMat, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    printf("Min v thresh: %lf v %lf.\n", minVal, thresh);
    while(minVal < thresh) {
        count++;
        printf("Min (%lf < %lf) at [%d,%d].\n", minVal, thresh, minLoc.x, minLoc.y);
        // Draw rectangles in original frame at max location
        rectangle(drawMatrix, minLoc, Point(minLoc.x + templ.cols, minLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
        // Set area around this point to 0s so it isn't considered again
        rectangle(minMaxMat, 
                Point(minLoc.x - templ.cols / 2, minLoc.y - templ.rows / 2), 
                Point(minLoc.x + templ.cols / 2, minLoc.y + templ.rows / 2), Scalar(thresh), CV_FILLED, 8, 0);
//        cvShowImage("Input", frameImage);
//        outImage = dst;
//        cvShowImage("Output", &outImage);
//        minVal = 5;
//        int c = cvWaitKey();
//        while ((char) c != 27) {
//            c = cvWaitKey();
//        }
        // Find next max
        minMaxLoc(minMaxMat, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    }
    fprintf(stderr, "%d circles detected.\n", count);
}

void Gesture::houghCircles(const Mat& src, Mat& dst, Mat& drawMatrix, Mat& templ) {
    // Smooth it, otherwise a lot of false circles may be detected
    //    GaussianBlur(src, dst, Size(9, 9), 4, 4);
    medianBlur(src, dst, 5);

    vector<Vec3f> circles;
    HoughCircles(src, circles, CV_HOUGH_GRADIENT, 2, 40, 200, 100);
    for (size_t i = 0; i < circles.size(); i++) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // draw the circle center
        circle(drawMatrix, center, 3, Scalar(0, 255, 0), -1, 8, 0);
        // draw the circle outline
        circle(drawMatrix, center, radius, Scalar(0, 0, 255), 3, 8, 0);
    }
    fprintf(stderr, "Found %ld circles.\n", circles.size());
}

void Gesture::printInfo(const Mat& mat) {
    fprintf(stderr, "mat: %d %d %d %d %d\n", mat.rows, mat.cols, mat.channels(), mat.depth(), mat.type());
}

void Gesture::cross(Mat &mat, int row, int col, int l) {
    for (int i = -l; i <= l; i++) {
        for (int j = -l; j <= l; j++) {
            mat.at<Vec3b > (row + i, col + j)[2] = 255;
            mat.at<Vec3b > (row + i, col + j)[1] = 0;
            mat.at<Vec3b > (row + i, col + j)[0] = 0;
        }
    }
}