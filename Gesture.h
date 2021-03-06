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
    // Orange
    static const double H_MIN_O1 = 0.294118; // (0.588235)
    static const double H_MAX_O1 = 0.891089; // (1.782178)
    static const double S_MIN_O1 = 194.656487; // (76.335877)
    static const double S_MAX_O1 = 199.651167; // (78.294575)
    static const double H_MIN_O2 = 172.781952; // (345.563904)
    static const double H_MAX_O2 = 179.428574; // (358.857147)
    static const double S_MIN_O2 = 188.214282; // (73.809522)
    static const double S_MAX_O2 = 231.818189; // (90.909094)    
//    static const double H_MIN_O1 = 3.913043; // (7.826087)
//    static const double H_MAX_O1 = 15.695364; // (31.390728)
//    static const double S_MIN_O1 = 12.047244; // (4.724409)
//    static const double S_MAX_O1 = 227.109375; // (89.062500)
//    static const double H_MIN_O2 = 3.913043; // (7.826087)
//    static const double H_MAX_O2 = 15.695364; // (31.390728)
//    static const double S_MIN_O2 = 12.047244; // (4.724409)
//    static const double S_MAX_O2 = 227.109375; // (89.062500)


    IplImage* frameImage;
    IplImage* prevFrame;
    IplImage* procFrame;
    IplImage* postFrame;
    IplImage tempImage, outImage;
    Mat frameMatrix, outputMatrix, redMatrix, greenMatrix, orangeMatrix, tempMatrix,
    template32Matrix, template48Matrix, template64Matrix, hsvMatrix,
    cclMatrix;
    // [cx, cy, theta, width, height]
    double centroidStats[5];
    
    // Runtime functions
    void processStream(int index);
    void processFile(const char *fileName);
    void analyzeFile(const char *fileName, const char *suffix1, const char *suffix2);
    // Image processing functions
    void nothing(const Mat& src, Mat& dst);
    void applyFlip(const Mat& src, Mat& dst);
    void applyMedian(const Mat& src, Mat& dst);
    void applyTableHSV(const Mat& src, Mat& dst, double hMin, double hMax, double sMin, double sMax, double vMin, double vMax);
    void findCCL(const Mat& inputMatrix, Mat& outputMatrix, bool considerDiagonals);
    void findCentroid(const Mat& inputMatrix, double* stats);
    void findCircles(Mat& src, Mat& dst, Mat& templ, double thresh, Vector<Point>& circles);
    // Helper functions
    uint minimum(const std::vector<uint> input);
    // Debugging functions
    void drawCentroid(Mat& src, double* stats, Scalar color);
    void drawSquares(Mat& src, Vector<Point>& circles, int length, Scalar color);
    void printCVTypes();
    void printInfo(const Mat &mat);
    void showImages(const Mat& inputMatrix, const Mat& processMatrix, const Mat& outputMatrix);
};

#endif	/* GESTURE_H */