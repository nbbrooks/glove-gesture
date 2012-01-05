//#pragma once
using namespace cv;
        
class Gesture {
public:
    Gesture(int, char **);
private:
    static const double THRESH_Y = 0.1;
    static const double THRESH_G = 0.1;
    // Pink
    // Measured mean chromacity values for red and green
    static const double R_CH_MEAN_P = 0.413464326726511;
    static const double G_CH_MEAN_P = 0.233825410563608;
    static const double B_CH_MEAN_P = 0.350683233395741;
    // Measured inverse std dev chromacity values for red and green
    static const double R_CH_VAR_INV_P = 12856.5616746348;
    static const double G_CH_VAR_INV_P = 17661.0926174714;
    static const double B_CH_VAR_INV_P = 35748.0113745476;
    
    // Green
    // Measured mean chromacity values for red and green
    static const double R_CH_MEAN_G = 0.279745482824469;
    static const double G_CH_MEAN_G = 0.389173827230952;
    static const double B_CH_MEAN_G = 0.328717603989065;
    // Measured inverse std dev chromacity values for red and green
    static const double R_CH_VAR_INV_G = 27548.0876730553;
    static const double G_CH_VAR_INV_G = 14208.2089616912;
    static const double B_CH_VAR_INV_G = 23977.4250454378;
    
    // Yellow
    // Measured mean chromacity values for red and green
    static const double R_CH_MEAN_Y = 0.365420145384165;
    static const double G_CH_MEAN_Y = 0.368234741618961;
    static const double B_CH_MEAN_Y = 0.264019571080339;
    // Measured inverse std dev chromacity values for red and green
    static const double R_CH_VAR_INV_Y = 30183.45063686865;
    static const double G_CH_VAR_INV_Y = 27203.35932348270;
    static const double B_CH_VAR_INV_Y = 10423.22968747357;
        
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
    Mat frameMatrix, colorMeanT, colorVarInv, inputMatrix, outputMatrix, yellowMatrix, greenMatrix, tempMatrix;
    
    void nothing(const Mat& src, Mat& dst);
    void applyFlip(const Mat& src, Mat& dst);
    void applyMedian(const Mat& src, Mat& dst);
    void applyInverse(const Mat& src, Mat& dst);
    void applyHistory(const Mat& src, Mat& prev, Mat& dst);
    void applyChRG(const Mat& src, Mat& dst, double rMean, double gMean, double rSDI, double gSDI, double thresh);
    void applyChRB(const Mat& src, Mat& dst, double rMean, double bMean, double rSDI, double bSDI, double thresh);
    void applyChRGB(const Mat& src, Mat& dst, double rMean, double gMean, double bMean, double rSDI, double gSDI, double bSDI, double thresh);
};
