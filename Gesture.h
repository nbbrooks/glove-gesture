//#pragma once
using namespace cv;
        
class Gesture {
public:
    Gesture(int, char **);
private:
    static const double THRESH_R = 0.005;
    static const double THRESH_G = 0.1;
    static const double THRESH_B = 0.005;
    
    static const double R_CH_MEAN_R = 0.433882;
    static const double G_CH_MEAN_R = 0.278347;
    static const double B_CH_MEAN_R = 0.285490;
    static const double R_CH_VAR_INV_R = 10433.009987;
    static const double G_CH_VAR_INV_R = 74382.599499;
    static const double B_CH_VAR_INV_R = 15499.491534;
    
    static const double R_CH_MEAN_G = 0.240018;
    static const double G_CH_MEAN_G = 0.398915;
    static const double B_CH_MEAN_G = 0.359266;
    static const double R_CH_VAR_INV_G = 2712.172340;
    static const double G_CH_VAR_INV_G = 5224.780428;
    static const double B_CH_VAR_INV_G = 19822.026776;
    
    static const double R_CH_MEAN_B = 0.257304;
    static const double G_CH_MEAN_B = 0.341129;
    static const double B_CH_MEAN_B = 0.399796;
    static const double R_CH_VAR_INV_B = 49091.921348;
    static const double G_CH_VAR_INV_B = 113525.317156;
    static const double B_CH_VAR_INV_B = 55267.706807;
    
    // Blue
    static const double H_MIN_B = 100.851064;
    static const double H_MAX_B = 105.851064;
//    static const double H_MAX_B = 110.851064;
//    static const double H_MIN_B = 110.851064;
//    static const double H_MAX_B = 115.500000;
    static const double S_MIN_B = 46.130653;
    static const double S_MAX_B = 139.329897;
    static const double V_MIN_B = 191.000000;
    static const double V_MAX_B = 218.000000;
    
    // Red
    static const double H_MIN_R = 168.0000;
    static const double H_MAX_R = 179.687500;
    static const double S_MIN_R = 62.943038;
    static const double S_MAX_R = 143.157895;
    static const double V_MIN_R = 150.000000;
    static const double V_MAX_R = 182.000000;
        
    IplImage* frameImage;
    IplImage* prevFrame;
    IplImage* procFrame;
    IplImage* postFrame;
    IplImage outImage;
    int height, width, step, depth, channels;
    int outheight, outwidth, outstep, outdepth, outchannels;
    int save;
    uchar *data;
    uchar *procData;
    uchar *postData;
    uchar *outdata;
    int i, j, k;
    bool firstPass, display;
    Mat frameMatrix, colorMeanT, colorVarInv, inputMatrix, outputMatrix, redMatrix, greenMatrix, blueMatrix, tempMatrix, templateImage;
    
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
    void templateCircles(const Mat& src, Mat& dst, Mat& frameMatrix, Mat& templ);
    void houghCircles(const Mat& src, Mat& dst, Mat& frameMatrix, Mat& templ);
    void printInfo(const Mat &mat);
};
