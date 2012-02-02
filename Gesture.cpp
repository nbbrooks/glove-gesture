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
    int bufferIndex = 0, bufferModeCount, bufferModeIndex, gesture;
    int gestureBuffer[BUFFER_SIZE], modeCounter[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        gestureBuffer[i] = 0;
        modeCounter[i] = 0;
    }
    Vector<Point> greenCircles32, greenCircles48, greenCircles64, redCircles32, redCircles48, redCircles64;
    template32Matrix = imread("template-32.ppm", 0);
    template48Matrix = imread("template-48.ppm", 0);
    template64Matrix = imread("template-64.ppm", 0);

    // Normalize 0 to 1 so matching scores are in usable range
    template32Matrix.convertTo(template32Matrix, CV_32FC1, 1.0 / 255.0, 0);
    template48Matrix.convertTo(template48Matrix, CV_32FC1, 1.0 / 255.0, 0);
    template64Matrix.convertTo(template64Matrix, CV_32FC1, 1.0 / 255.0, 0);

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
        if (DEBUG) {
            cvWaitKey(0);
        }

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
        // Normalize 0 to 1 so matching scores are in usable range
        greenMatrix.convertTo(greenMatrix, CV_32FC1, 1.0 / 255.0, 0);
        redMatrix.convertTo(redMatrix, CV_32FC1, 1.0 / 255.0, 0);
        Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, greenMatrix.type(), Scalar(0));
        vector<Mat> bgr;
        //Red 64
        templateCircles(redMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, redCircles64);
        if (DEBUG) {
            bgr.clear();
            bgr.push_back(blueMatrix);
            bgr.push_back(greenMatrix);
            bgr.push_back(redMatrix);
            merge(bgr, tempMatrix);
            normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
            showImages(frameMatrix, tempMatrix, outputMatrix);
        }
        // Red 48
        templateCircles(redMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, redCircles48);
        if (DEBUG) {
            bgr.clear();
            bgr.push_back(blueMatrix);
            bgr.push_back(greenMatrix);
            bgr.push_back(redMatrix);
            merge(bgr, tempMatrix);
            normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
            showImages(frameMatrix, tempMatrix, outputMatrix);
        }
        // Red 32
        templateCircles(redMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, redCircles32);
        if (DEBUG) {
            bgr.clear();
            bgr.push_back(blueMatrix);
            bgr.push_back(greenMatrix);
            bgr.push_back(redMatrix);
            merge(bgr, tempMatrix);
            normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
            showImages(frameMatrix, tempMatrix, outputMatrix);
        }
        // Green 64
        templateCircles(greenMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, greenCircles64);
        if (DEBUG) {
            bgr.clear();
            bgr.push_back(blueMatrix);
            bgr.push_back(greenMatrix);
            bgr.push_back(redMatrix);
            merge(bgr, tempMatrix);
            normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
            showImages(frameMatrix, tempMatrix, outputMatrix);
        }
        // Green 48
        templateCircles(greenMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, greenCircles48);
        if (DEBUG) {
            bgr.clear();
            bgr.push_back(blueMatrix);
            bgr.push_back(greenMatrix);
            bgr.push_back(redMatrix);
            merge(bgr, tempMatrix);
            normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
            showImages(frameMatrix, tempMatrix, outputMatrix);
        }
        // Green 32
        templateCircles(greenMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, greenCircles32);
        if (DEBUG) {
            bgr.clear();
            bgr.push_back(blueMatrix);
            bgr.push_back(greenMatrix);
            bgr.push_back(redMatrix);
            merge(bgr, tempMatrix);
            normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
            showImages(frameMatrix, tempMatrix, outputMatrix);
        }
        fprintf(stderr, "%lu green circles, %lu red circles.\n",
                (greenCircles32.size() + greenCircles48.size() + greenCircles64.size()),
                (redCircles32.size() + redCircles48.size() + redCircles64.size()));

        // Output images
        // Segmentation image
        bgr.clear();
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
        // Normalize 0 to 1 so matching scores are in usable range
        greenMatrix.convertTo(greenMatrix, CV_32FC1, 1.0 / 255.0, 0);
        redMatrix.convertTo(redMatrix, CV_32FC1, 1.0 / 255.0, 0);
        Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, greenMatrix.type(), Scalar(0));
        vector<Mat> bgr;
        templateCircles(redMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, redCircles64);
        templateCircles(redMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, redCircles48);
        templateCircles(redMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, redCircles32);
        templateCircles(greenMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, greenCircles64);
        templateCircles(greenMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, greenCircles48);
        templateCircles(greenMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, greenCircles32);
        fprintf(stderr, "%lu green circles, %lu red circles.\n",
                (greenCircles32.size() + greenCircles48.size() + greenCircles64.size()),
                (redCircles32.size() + redCircles48.size() + redCircles64.size()));

        /*
         * Here we use a rolling window of size BUFFER_SIZE so that several 
         * frames of the same gesture must be captured to result in a command.
         * We calculate the mode and make sure the mode count is above 
         * MODE_MINIMUM.
         */
        // First check that we have a possible number of detected circles
        if ((greenCircles32.size() + greenCircles48.size() + greenCircles64.size()) <= MAX_GREEN_CIRCLES &&
                (redCircles32.size() + redCircles48.size() + redCircles64.size()) <= MAX_RED_CIRCLES) {
            // Store number as <number green circles><number red circles>
            gestureBuffer[bufferIndex] = 10 * (greenCircles32.size() + greenCircles48.size() + greenCircles64.size()) +
                    redCircles32.size() + redCircles48.size() + redCircles64.size();
            bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
            // Calculate mode
            bufferModeCount = 0, bufferModeIndex = 0;
            for (int curNumIndex = 0; curNumIndex < BUFFER_SIZE; curNumIndex++) {
                modeCounter[curNumIndex] = 0;
                for (int i = 0; i < BUFFER_SIZE; i++) {
                    if (gestureBuffer[i] == gestureBuffer[curNumIndex]) {
                        modeCounter[curNumIndex]++;
                    }
                }
                if (modeCounter[curNumIndex] > bufferModeCount) {
                    bufferModeIndex = curNumIndex;
                    bufferModeCount = modeCounter[curNumIndex];
                }
            }
            // Check against minimum mode count
            if (bufferModeCount >= MODE_MINIMUM) {
                gesture = gestureBuffer[bufferModeIndex];
                // Print out mode number
                largePrint(gesture);
            }
        }

        // Output images
        // Segmentation image
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

void Gesture::templateCircles(Mat& src, Mat& dst, Mat& templ, double thresh, Vector<Point>& circles) {
    // Template matching
    matchTemplate(src, templ, dst, CV_TM_SQDIFF);
    // Find best matches
    double minVal;
    Point minLoc;
    minMaxLoc(dst, &minVal, NULL, &minLoc, NULL, Mat());
    //printf("Checking %lf < %lf at [%d,%d].\n", minVal, thresh, minLoc.x, minLoc.y);
    while (minVal < thresh) {
        // Set area around this point to 0s so it isn't considered again
        rectangle(dst,
                Point(minLoc.x - templ.cols / 2, minLoc.y - templ.rows / 2),
                Point(minLoc.x + templ.cols / 2, minLoc.y + templ.rows / 2), Scalar(thresh), CV_FILLED, 8, 0);
        //fprintf(stderr, "dst rectangle at [%d,%d] with length %d.\n",
        //        minLoc.x - templ.cols / 2, minLoc.y - templ.rows / 2, templ.cols);
        // Remember to compensate for the template size border taken out during template matching
        rectangle(src,
                Point(minLoc.x, minLoc.y),
                Point(minLoc.x + templ.cols, minLoc.y + templ.rows), Scalar(0), CV_FILLED, 8, 0);
        circles.push_back(Point(minLoc.x + templ.cols / 2, minLoc.y + templ.rows / 2));
        //fprintf(stderr, "src rectangle at [%d,%d] with length %d.\n",
        //        minLoc.x + templ.cols / 2, minLoc.y + templ.rows / 2, templ.cols);

        // Find next max
        minMaxLoc(dst, &minVal, NULL, &minLoc, NULL, Mat());
        //printf("Checking %lf < %lf at [%d,%d].\n", minVal, thresh, minLoc.x, minLoc.y);
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
    //fprintf(stderr, "Drawing %ld squares of length %d.\n", circles.size(), length);
    Point point;
    for (size_t i = 0; i < circles.size(); i++) {
        point = circles[i];
        rectangle(src,
                Point(point.x - length / 2, point.y - length / 2),
                Point(point.x + length / 2, point.y + length / 2), color, 2, 8, 0);
    }
}

void Gesture::showImages(const Mat& inputMatrix, const Mat& processMatrix, const Mat& outputMatrix) {
    IplImage inputImage, processImage, outputImage;
    if (!inputMatrix.empty()) {
        inputImage = inputMatrix;
        cvShowImage("Input", &inputImage);
        fprintf(stderr, "Input matrix is size [%d,%d]\n", inputMatrix.rows, inputMatrix.cols);
    }
    if (!processMatrix.empty()) {
        processImage = processMatrix;
        cvShowImage("Process", &processImage);
        fprintf(stderr, "Process matrix is size [%d,%d]\n", processMatrix.rows, processMatrix.cols);
    }
    if (!outputMatrix.empty()) {
        outputImage = outputMatrix;
        cvShowImage("Output", &outputImage);
        fprintf(stderr, "Output matrix is size [%d,%d]\n", outputMatrix.rows, outputMatrix.cols);
    }
    // Pause before quitting
    int c = cvWaitKey();
    while ((char) c != 27) {
        c = cvWaitKey();
    }
    fprintf(stderr, "----------------------------\n");
}

std::string largeNumbers[10][16] = {
    {"     000000000     ",
        "   00:::::::::00   ",
        " 00:::::::::::::00 ",
        "0:::::::000:::::::0",
        "0::::::0   0::::::0",
        "0:::::0     0:::::0",
        "0:::::0     0:::::0",
        "0:::::0 000 0:::::0",
        "0:::::0 000 0:::::0",
        "0:::::0     0:::::0",
        "0:::::0     0:::::0",
        "0::::::0   0::::::0",
        "0:::::::000:::::::0",
        " 00:::::::::::::00 ",
        "   00:::::::::00   ",
        "     000000000     "},
    {"  1111111   ",
        " 1::::::1   ",
        "1:::::::1   ",
        "111:::::1   ",
        "   1::::1   ",
        "   1::::1   ",
        "   1::::1   ",
        "   1::::l   ",
        "   1::::l   ",
        "   1::::l   ",
        "   1::::l   ",
        "   1::::l   ",
        "111::::::111",
        "1::::::::::1",
        "1::::::::::1",
        "111111111111"},
    {" 222222222222222    ",
        "2:::::::::::::::22  ",
        "2::::::222222:::::2 ",
        "2222222     2:::::2 ",
        "            2:::::2 ",
        "            2:::::2 ",
        "         2222::::2  ",
        "    22222::::::22   ",
        "  22::::::::222     ",
        " 2:::::22222        ",
        "2:::::2             ",
        "2:::::2             ",
        "2:::::2       222222",
        "2::::::2222222:::::2",
        "2::::::::::::::::::2",
        "22222222222222222222"},
    {" 333333333333333   ",
        "3:::::::::::::::33 ",
        "3::::::33333::::::3",
        "3333333     3:::::3",
        "            3:::::3",
        "            3:::::3",
        "    33333333:::::3 ",
        "    3:::::::::::3  ",
        "    33333333:::::3 ",
        "            3:::::3",
        "            3:::::3",
        "            3:::::3",
        "3333333     3:::::3",
        "3::::::33333::::::3",
        "3:::::::::::::::33 ",
        " 333333333333333   "},
    {"       444444444  ",
        "      4::::::::4  ",
        "     4:::::::::4  ",
        "    4::::44::::4  ",
        "   4::::4 4::::4  ",
        "  4::::4  4::::4  ",
        " 4::::4   4::::4  ",
        "4::::444444::::444",
        "4::::::::::::::::4",
        "4444444444:::::444",
        "          4::::4  ",
        "          4::::4  ",
        "          4::::4  ",
        "        44::::::44",
        "        4::::::::4",
        "        4444444444"},
    {"555555555555555555 ",
        "5::::::::::::::::5 ",
        "5::::::::::::::::5 ",
        "5:::::555555555555 ",
        "5:::::5            ",
        "5:::::5            ",
        "5:::::5555555555   ",
        "5:::::::::::::::5  ",
        "555555555555:::::5 ",
        "            5:::::5",
        "            5:::::5",
        "5555555     5:::::5",
        "5::::::55555::::::5",
        " 55:::::::::::::55 ",
        "   55:::::::::55   ",
        "     555555555     "},
    {"        66666666   ",
        "       6::::::6    ",
        "      6::::::6     ",
        "     6::::::6      ",
        "    6::::::6       ",
        "   6::::::6        ",
        "  6::::::6         ",
        " 6::::::::66666    ",
        "6::::::::::::::66  ",
        "6::::::66666:::::6 ",
        "6:::::6     6:::::6",
        "6:::::6     6:::::6",
        "6::::::66666::::::6",
        " 66:::::::::::::66 ",
        "   66:::::::::66   ",
        "     666666666     "},
    {"77777777777777777777",
        "7::::::::::::::::::7",
        "7::::::::::::::::::7",
        "777777777777:::::::7",
        "           7::::::7 ",
        "          7::::::7  ",
        "         7::::::7   ",
        "        7::::::7    ",
        "       7::::::7     ",
        "      7::::::7      ",
        "     7::::::7       ",
        "    7::::::7        ",
        "   7::::::7         ",
        "  7::::::7          ",
        " 7::::::7           ",
        "77777777            "},
    {"     888888888     ",
        "   88:::::::::88   ",
        " 88:::::::::::::88 ",
        "8::::::88888::::::8",
        "8:::::8     8:::::8",
        "8:::::8     8:::::8",
        " 8:::::88888:::::8 ",
        "  8:::::::::::::8  ",
        " 8:::::88888:::::8 ",
        "8:::::8     8:::::8",
        "8:::::8     8:::::8",
        "8:::::8     8:::::8",
        "8::::::88888::::::8",
        " 88:::::::::::::88 ",
        "   88:::::::::88   ",
        "     888888888     "},
    {"     999999999     ",
        "   99:::::::::99   ",
        " 99:::::::::::::99 ",
        "9::::::99999::::::9",
        "9:::::9     9:::::9",
        "9:::::9     9:::::9",
        " 9:::::99999::::::9",
        "  99::::::::::::::9",
        "    99999::::::::9 ",
        "         9::::::9  ",
        "        9::::::9   ",
        "       9::::::9    ",
        "      9::::::9     ",
        "     9::::::9      ",
        "    9::::::9       ",
        "   99999999        "}
};

void Gesture::largePrint(int circleCount) {
    int greenCount = circleCount / 10;
    int redCount = circleCount % 10;
    for (int i = 0; i < 16; i++) {
        fprintf(stderr, "%s\t:\t%s\n", largeNumbers[greenCount][i].c_str(), largeNumbers[redCount][i].c_str());
    }
}