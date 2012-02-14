#ifndef GESTURE_H
#define	GESTURE_H

#include <vector>

using namespace cv;

class Gesture {
public:
    Gesture(int, char **);
private:
    static const bool DEBUG = false;

    // Gesture statistical mode settings
    static const int BUFFER_SIZE = 10;
    static const int MODE_MINIMUM = 5;
    static const int MAX_GREEN_CIRCLES = 5;
    static const int MAX_RED_CIRCLES = 2;

    // Template matching settings
    static const double THRESH_R = 0.005;
    static const double THRESH_G = 0.1;
    static const double THRESH_B = 0.005;
    static const double THRESH_TEMPLATE_32 = 250.0;
    static const double THRESH_TEMPLATE_48 = 500.0;
    static const double THRESH_TEMPLATE_64 = 1280.0;

    // Color segmentation settings
    // OpenCV (uint) / OpenCV (float) / GIMP
    // H: [0,180] / [0,360] / [0,360]
    // S: [0,255] / [0,1] / [0,100]
    // V: [0,255] / [0,1] / [0,100]
    // Green
    static const double H_MIN_G = 36.190475; // (72.380951)
    static const double H_MAX_G = 67.500000; // (135.000000)
    static const double S_MIN_G = 83.191490; // (32.624114)
    static const double S_MAX_G = 189.572375; // (74.342108)
    // Red
    static const double H_MIN_R1 = 0.220588; // (0.441176)
    static const double H_MAX_R1 = 9.120000; // (18.240000)
    static const double S_MIN_R1 = 141.666673; // (55.555558)
    static const double S_MAX_R1 = 216.128055; // (84.756100)
    static const double H_MIN_R2 = 173.235291; // (346.470581)
    static const double H_MAX_R2 = 179.763779; // (359.527557)
    static const double S_MIN_R2 = 157.382812; // (61.718750)
    static const double S_MAX_R2 = 208.935482; // (81.935483)

    IplImage* frameImage;
    IplImage* prevFrame;
    IplImage* procFrame;
    IplImage* postFrame;
    IplImage tempImage, outImage;
    int outheight, outwidth, outstep, outdepth, outchannels;
    bool firstPass, display;
    Mat frameMatrix, outputMatrix, redMatrix, greenMatrix, tempMatrix,
    template32Matrix, template48Matrix, template64Matrix, hsvMatrix,
    cclMatrix;
    float centroidStats[5];

    void nothing(const Mat& src, Mat& dst);
    void applyFlip(const Mat& src, Mat& dst);
    void applyMedian(const Mat& src, Mat& dst);
    void applyInverse(const Mat& src, Mat& dst);
    void applyHistory(const Mat& src, Mat& prev, Mat& dst);
    void applyChRG(const Mat& src, Mat& dst, double rMean, double gMean, double rSDI, double gSDI, double thresh);
    void applyChRB(const Mat& src, Mat& dst, double rMean, double bMean, double rSDI, double bSDI, double thresh);
    void applyChRGB(const Mat& src, Mat& dst, double rMean, double gMean, double bMean, double rSDI, double gSDI, double bSDI, double thresh);
    void applyGaussHSV(const Mat& src, Mat& dst, double hMean, double sMean, double vMean, double hSDI, double sSDI, double vSDI, double thresh);
    void applyTableHSV(const Mat& src, Mat& dst, double hMin, double hMax, double sMin, double sMax, double vMin, double vMax);
    void templateCircles(Mat& src, Mat& dst, Mat& templ, double thresh, Vector<Point>& circles);
    void houghCircles(const Mat& src, Mat& dst, Mat& drawMatrix, Mat& templ);
    void printInfo(const Mat &mat);
    void drawSquares(Mat& src, Vector<Point>& circles, int length, Scalar color);
    void showImages(const Mat& inputMatrix, const Mat& processMatrix, const Mat& outputMatrix);
    void findCCL(const Mat& inputMatrix, const Mat& processMatrix, const Mat& outputMatrix);
    void findCentroid(const Mat& inputMatrix, float** stats);
};

#endif	/* GESTURE_H */