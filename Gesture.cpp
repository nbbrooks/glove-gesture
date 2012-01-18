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
    int c, save = 0;
    std::stringstream debug;
    Vector<Point> greenCircles32, greenCircles48, greenCircles64, redCircles32, redCircles48, redCircles64;
    template32Matrix = imread("template-32.ppm", 0);
    template48Matrix = imread("template-48.ppm", 0);
    template64Matrix = imread("template-64.ppm", 0);

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
    
    if (argc == 2 && strlen(argv[1]) > 1 && !isdigit(argv[1][0])) {
        // Load image
        std::cout << "File is " << argv[1] << std::endl;
        frameImage = cvLoadImage(argv[1]);
        frameMatrix = cvarrToMat(frameImage);

        // Pre-processing
        medianBlur(frameMatrix, frameMatrix, 5);
        GaussianBlur(frameMatrix, frameMatrix, Size(9, 9), 1, 1);
        // Colorspace processing
        cvtColor(frameMatrix, hsvMatrix, CV_BGR2HSV);
        applyTableHSV(frameMatrix, greenMatrix, H_MIN_G, H_MAX_G, S_MIN_G, S_MAX_G, V_MIN_G, V_MAX_G);
        applyTableHSV(frameMatrix, tempMatrix, H_MIN_R1, H_MAX_R1, S_MIN_R1, S_MAX_R1, V_MIN_R1, V_MAX_R1);
        applyTableHSV(frameMatrix, redMatrix, H_MIN_R2, H_MAX_R2, S_MIN_R2, S_MAX_R2, V_MIN_R2, V_MAX_R2);
        add(tempMatrix, redMatrix, redMatrix);
        // Circle detection
        templateCircles(greenMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, greenCircles32);
        templateCircles(greenMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, greenCircles48);
        templateCircles(greenMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, greenCircles64);
        templateCircles(redMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, redCircles32);
        templateCircles(redMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, redCircles48);
        templateCircles(redMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, redCircles64);
//        fprintf(stderr, "%lu green circles, %lu red circles.\n",
//                (greenCircles32.size() + greenCircles32.size() + greenCircles32.size()),
//                (redCircles32.size() + redCircles48.size() + redCircles64.size()));
        // Output images
        // Segmentation image
        Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, greenMatrix.type(), Scalar(0));
        vector<Mat> bgr;
        bgr.push_back(blueMatrix);
        bgr.push_back(greenMatrix);
        bgr.push_back(redMatrix);
        merge(bgr, tempMatrix);
        // Box detected circles in images
        drawSquares(frameMatrix, greenCircles32, template32Matrix.cols, Scalar(0, 255, 0));
        drawSquares(frameMatrix, greenCircles48, template48Matrix.cols, Scalar(0, 255, 0));
        drawSquares(frameMatrix, greenCircles64, template64Matrix.cols, Scalar(0, 255, 0));
        drawSquares(frameMatrix, redCircles32, template32Matrix.cols, Scalar(0, 0, 255));
        drawSquares(frameMatrix, redCircles48, template48Matrix.cols, Scalar(0, 0, 255));
        drawSquares(frameMatrix, redCircles64, template64Matrix.cols, Scalar(0, 0, 255));
        drawSquares(tempMatrix, greenCircles32, template32Matrix.cols, Scalar(0, 255, 0));
        drawSquares(tempMatrix, greenCircles48, template48Matrix.cols, Scalar(0, 255, 0));
        drawSquares(tempMatrix, greenCircles64, template64Matrix.cols, Scalar(0, 255, 0));
        drawSquares(tempMatrix, redCircles32, template32Matrix.cols, Scalar(0, 0, 255));
        drawSquares(tempMatrix, redCircles48, template48Matrix.cols, Scalar(0, 0, 255));
        drawSquares(tempMatrix, redCircles64, template64Matrix.cols, Scalar(0, 0, 255));
        // Display results
        tempImage = tempMatrix;
        normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
        outImage = outputMatrix;
        cvShowImage("Input", frameImage);
        cvShowImage("Process", &tempImage);
        cvShowImage("Output", &outImage);
        // Pause before quitting
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
    CvCapture* capture = 0;

    if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0]))) {
        capture = cvCreateCameraCapture(argc == 2 ? argv[1][0] - '0' : 0);
    }
//    else if (argc == 2)
//        capture = cvCaptureFromAVI(argv[1]);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);

    if (!capture) {
        fprintf(stderr, "Could not initialize capturing.\n");
        return;
    }

    for (;;) {
        // Frame capture
        frameImage = cvQueryFrame(capture);
        if (!frameImage)
            break;
        frameMatrix = cvarrToMat(frameImage);
        display = true;
        // Pre-processing
        medianBlur(frameMatrix, frameMatrix, 5);
        GaussianBlur(frameMatrix, frameMatrix, Size(9, 9), 1, 1);
        // Colorspace processing
        cvtColor(frameMatrix, hsvMatrix, CV_BGR2HSV);
        applyTableHSV(hsvMatrix, greenMatrix, H_MIN_G, H_MAX_G, S_MIN_G, S_MAX_G, V_MIN_G, V_MAX_G);
        applyTableHSV(hsvMatrix, tempMatrix, H_MIN_R1, H_MAX_R1, S_MIN_R1, S_MAX_R1, V_MIN_R1, V_MAX_R1);
        applyTableHSV(hsvMatrix, redMatrix, H_MIN_R2, H_MAX_R2, S_MIN_R2, S_MAX_R2, V_MIN_R2, V_MAX_R2);
        add(tempMatrix, redMatrix, redMatrix);
        // Circle detection
        templateCircles(greenMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, greenCircles32);
        templateCircles(greenMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, greenCircles48);
        templateCircles(greenMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, greenCircles64);
        templateCircles(redMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, redCircles32);
        templateCircles(redMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, redCircles48);
        templateCircles(redMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, redCircles64);
        fprintf(stderr, "%lu green circles, %lu red circles.\n",
                (greenCircles32.size() + greenCircles32.size() + greenCircles32.size()),
                (redCircles32.size() + redCircles48.size() + redCircles64.size()));
        // Output images
        // Segmentation image
        Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, greenMatrix.type(), Scalar(0));
        vector<Mat> bgr;
        bgr.push_back(blueMatrix);
        bgr.push_back(greenMatrix);
        bgr.push_back(redMatrix);
        merge(bgr, tempMatrix);
        // Box detected circles in images
        drawSquares(frameMatrix, greenCircles32, template32Matrix.cols, Scalar(0, 255, 0));
        drawSquares(frameMatrix, greenCircles48, template48Matrix.cols, Scalar(0, 255, 0));
        drawSquares(frameMatrix, greenCircles64, template64Matrix.cols, Scalar(0, 255, 0));
        drawSquares(frameMatrix, redCircles32, template32Matrix.cols, Scalar(0, 0, 255));
        drawSquares(frameMatrix, redCircles48, template48Matrix.cols, Scalar(0, 0, 255));
        drawSquares(frameMatrix, redCircles64, template64Matrix.cols, Scalar(0, 0, 255));
        drawSquares(tempMatrix, greenCircles32, template32Matrix.cols, Scalar(0, 255, 0));
        drawSquares(tempMatrix, greenCircles48, template48Matrix.cols, Scalar(0, 255, 0));
        drawSquares(tempMatrix, greenCircles64, template64Matrix.cols, Scalar(0, 255, 0));
        drawSquares(tempMatrix, redCircles32, template32Matrix.cols, Scalar(0, 0, 255));
        drawSquares(tempMatrix, redCircles48, template48Matrix.cols, Scalar(0, 0, 255));
        drawSquares(tempMatrix, redCircles64, template64Matrix.cols, Scalar(0, 0, 255));
        // Display values for center pixel
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
        rectangle(frameMatrix, Point(239, 319), Point(241, 321), Scalar(0, 0, 255), CV_FILLED, 8, 0);
        rectangle(tempMatrix, Point(239, 319), Point(241, 321), Scalar(0, 0, 255), CV_FILLED, 8, 0);
        // Display results
        tempImage = tempMatrix;
        normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
        outImage = outputMatrix;
        if (display) {
            cvShowImage("Input", frameImage);
            cvShowImage("Process", &tempImage);
            cvShowImage("Output", &outImage);
        }
        // Poll for input before looping
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
        // Clear detected circle list each cycle
        greenCircles32.clear();
        greenCircles48.clear();
        greenCircles64.clear();
        redCircles32.clear();
        redCircles48.clear();
        redCircles64.clear();
//        sleep(1);
    }
    // Destroy and quit
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
    uchar *data = src.data;
    uchar *postData;
    if (firstPass) {
        postFrame = cvCreateImage(cvSize(width, height), depth, channels);
        postData = (uchar *) postFrame->imageData;
    }
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) for (int k = 0; k < channels; k++) {
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
    vector<Mat> hsv;
    split(src, hsv);
    // Apply min
    Mat hMinT, hMaxT, sMinT, sMaxT, vMinT, vMaxT;
    threshold(hsv[0], hMinT, hMin, 255, THRESH_BINARY);
    threshold(hsv[1], sMinT, sMin, 255, THRESH_BINARY);
    threshold(hsv[2], vMinT, vMin, 255, THRESH_BINARY);
    // Apply max
    threshold(hsv[0], hMaxT, hMax, 255, THRESH_BINARY_INV);
    threshold(hsv[1], sMaxT, sMax, 255, THRESH_BINARY_INV);
    threshold(hsv[2], vMaxT, vMax, 255, THRESH_BINARY_INV);
    // OR the min and max
    Mat hT, sT, vT;
    bitwise_and(hMinT, hMaxT, hT);
    bitwise_and(sMinT, sMaxT, sT);
    bitwise_and(vMinT, vMaxT, vT);
    // AND the OR results
    bitwise_and(hT, sT, dst);

    return;
}

void Gesture::templateCircles(const Mat& src, Mat& dst, Mat& templ, double thresh, Vector<Point>& circles) {
    // Normalize 0 to 1 so matching scores are in usable range
    Mat srcNorm, templateNorm;
    src.convertTo(srcNorm, CV_32FC1, 1.0 / 255.0, 0);
    templ.convertTo(templateNorm, CV_32FC1, 1.0 / 255.0, 0);
    // Template matching
    matchTemplate(srcNorm, templateNorm, dst, CV_TM_SQDIFF);
    // Find best matches
    double minVal;
    Point minLoc;
    minMaxLoc(dst, &minVal, NULL, &minLoc, NULL, Mat());
//    printf("Checking %lf < %lf at [%d,%d].\n", minVal, thresh, minLoc.x, minLoc.y);
    while (minVal < thresh) {
        circles.push_back(minLoc);
        // Set area around this point to 0s so it isn't considered again
        rectangle(dst,
                Point(minLoc.x - templ.cols / 2, minLoc.y - templ.rows / 2),
                Point(minLoc.x + templ.cols / 2, minLoc.y + templ.rows / 2), Scalar(thresh), CV_FILLED, 8, 0);
        // Find next max
        minMaxLoc(dst, &minVal, NULL, &minLoc, NULL, Mat());
//        printf("Checking %lf < %lf at [%d,%d].\n", minVal, thresh, minLoc.x, minLoc.y);
    }
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

void Gesture::drawSquares(Mat& src, Vector<Point>& circles, int length, Scalar color) {
    Point point;
    for (size_t i = 0; i < circles.size(); i++) {
        point = circles[i];
        rectangle(src, point, Point(point.x + length, point.y + length), color, 2, 8, 0);
        rectangle(src, point, Point(point.x + length, point.y + length), color, 2, 8, 0);
    }
}