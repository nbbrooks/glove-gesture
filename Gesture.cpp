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
    printf("\ts   - save current input frame\n");

    cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);
    templateImage = imread("template.ppm", 1 );

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
        //        nothing(frameMatrix, outputMatrix);
        applyFlip(frameMatrix, frameMatrix);
        applyMedian(frameMatrix, frameMatrix);
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
        applyTableHSV(frameMatrix, blueMatrix, H_MIN_B, H_MAX_B, S_MIN_B, S_MAX_B, V_MIN_B, V_MAX_B);
        applyTableHSV(frameMatrix, redMatrix, H_MIN_R, H_MAX_R, S_MIN_R, S_MAX_R, V_MIN_R, V_MAX_R);

        //        Mat redMatrix(blueMatrix.rows, blueMatrix.cols, CV_32FC1, Scalar(0));
        Mat greenMatrix(blueMatrix.rows, blueMatrix.cols, blueMatrix.type(), Scalar(0));
        //        Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, CV_32FC1, Scalar(0));
        vector<Mat> bgr;
        // BGR ordering!!!
        //        printf("redMatrix: %d %d %d %d\n", redMatrix.rows, redMatrix.cols, redMatrix.depth(), redMatrix.type());
        //        printf("greenMatrix: %d %d %d %d\n", greenMatrix.rows, greenMatrix.cols, greenMatrix.depth(), greenMatrix.type());
        //        printf("blueMatrix: %d %d %d %d\n", blueMatrix.rows, blueMatrix.cols, blueMatrix.depth(), blueMatrix.type());
        bgr.push_back(blueMatrix);
        bgr.push_back(greenMatrix);
        bgr.push_back(redMatrix);
        merge(bgr, outputMatrix);
//        circles(redMatrix, tempMatrix, frameMatrix, templateImage);

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

void Gesture::circles(const Mat& src, Mat& dst, Mat& frameMatrix, Mat& templ) {
    std::cout<<"cirlces\n";
//    int result_cols = src.cols - templ.cols + 1;
//    int result_rows = src.rows - templ.rows + 1;
//    dst.create(result_cols, result_rows, CV_32FC1);
    /// Do the Matching and Normalize
    int method = CV_TM_SQDIFF;
    src.convertTo(dst, CV_8U, 1.0, 0);
    templ.convertTo(templ, CV_8U, 1.0, 0);
    std::cout<<"matchTemplate\n";
    printf("dst: %d %d %d %d\n", dst.rows, dst.cols, dst.depth(), dst.type());
    printf("templ: %d %d %d %d\n", templ.rows, templ.cols, templ.depth(), templ.type());
    matchTemplate(dst, templ, dst, method);
//    normalize(dst, dst, 0, 1, NORM_MINMAX, -1, Mat());

    /// Localizing the best match with minMaxLoc
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;

    std::cout<<"minMaxLoc\n";
    minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
    if (method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED) {
        matchLoc = minLoc;
    } else {
        matchLoc = maxLoc;
    }

    std::cout<<"rectangle\n";
    /// Show me what you got
    rectangle(frameMatrix, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//    rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
}

//void MatchingMethod(int, void*) {
//    /// Source image to display
//    Mat img_display;
//    img.copyTo(img_display);
//
//    /// Create the result matrix
//    int result_cols = img.cols - templ.cols + 1;
//    int result_rows = img.rows - templ.rows + 1;
//
//    result.create(result_cols, result_rows, CV_32FC1);
//
//    /// Do the Matching and Normalize
//    matchTemplate(img, templ, result, match_method);
//    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
//
//    /// Localizing the best match with minMaxLoc
//    double minVal;
//    double maxVal;
//    Point minLoc;
//    Point maxLoc;
//    Point matchLoc;
//
//    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
//
//    /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
//    if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED) {
//        matchLoc = minLoc;
//    } else {
//        matchLoc = maxLoc;
//    }
//
//    /// Show me what you got
//    rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//    rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//
//    imshow(image_window, img_display);
//    imshow(result_window, result);
//
//    return;
//}