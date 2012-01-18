#include <vector>

//#pragma once
using namespace cv;

class Gesture {
public:
    Gesture(int, char **);
private:
    static const double THRESH_R = 0.005;
    static const double THRESH_G = 0.1;
    static const double THRESH_B = 0.005;
    static const double THRESH_TEMPLATE_32 = 250.0;
    static const double THRESH_TEMPLATE_48 = 500.0;
    static const double THRESH_TEMPLATE_64 = 1280.0;

    // Red
    static const double H_MIN_R1 = 0.000000; // (0.000000)
    static const double H_MAX_R1 = 5.760000; // (11.520000)
    static const double S_MIN_R1 = 97.142857; // (38.095238)
    static const double S_MAX_R1 = 151.594488; // (59.448819)
    static const double V_MIN_R1 = 207.000000; // (81.176471)
    static const double V_MAX_R1 = 254.000000; // (99.607843)
    static const double H_MIN_R2 = 168.857143; // (337.714286)
    static const double H_MAX_R2 = 179.791667; // (359.583333)
    static const double S_MIN_R2 = 85.351240; // (33.471074)
    static const double S_MAX_R2 = 151.980000; // (59.600000)
    static const double V_MIN_R2 = 198.000000; // (77.647059)
    static const double V_MAX_R2 = 254.000000; // (99.607843)

    // Green
    static const double H_MIN_G = 54.000000; // (108.000000)
    static const double H_MAX_G = 77.763158; // (155.526316)
    static const double S_MIN_G = 43.562500; // (17.083333)
    static const double S_MAX_G = 113.220000; // (44.400000)
    static const double V_MIN_G = 237.000000; // (92.941176)
    static const double V_MAX_G = 254.000000; // (99.607843)

    IplImage* frameImage;
    IplImage* prevFrame;
    IplImage* procFrame;
    IplImage* postFrame;
    IplImage tempImage, outImage;
    int outheight, outwidth, outstep, outdepth, outchannels;
    bool firstPass, display;
    Mat frameMatrix, outputMatrix, redMatrix,greenMatrix, tempMatrix,
    template32Matrix, template48Matrix, template64Matrix, hsvMatrix;

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
    void templateCircles(const Mat& src, Mat& dst, Mat& templ, double thresh, Vector<Point>& circles);
    void houghCircles(const Mat& src, Mat& dst, Mat& drawMatrix, Mat& templ);
    void printInfo(const Mat &mat);
    void drawSquares(Mat& src, Vector<Point>& circles, int length, Scalar color);
};
