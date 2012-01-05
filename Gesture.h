//#pragma once
using namespace cv;
        
class Gesture {
public:
    Gesture(int, char **);
private:
    static const double THRESH_Y = 0.1;
    static const double THRESH_G = 0.1;
//    static const double THRESH_Y = 0.75;
//    static const double THRESH_G = 0.75;
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
    static const double R_CH_MEAN_G = 0.256180986876333; //0.279745482824469;
    static const double G_CH_MEAN_G = 0.426178577031684; //0.389173827230952;
    static const double B_CH_MEAN_G = 0.315288620706374; //0.328717603989065;
    static const double R_CH_VAR_INV_G = 9566.59744933743; //27548.0876730553;
    static const double G_CH_VAR_INV_G = 8209.53372748956; //14208.2089616912;
    static const double B_CH_VAR_INV_G = 9736.21154743341; //23977.4250454378;
//    static const double R_CH_MEAN_G = 0.287164;
//    static const double G_CH_MEAN_G = 0.399463;
//    static const double B_CH_MEAN_G = 0.311769;
//    static const double R_CH_VAR_INV_G = 1361.386893;
//    static const double G_CH_VAR_INV_G = 720.537663;
//    static const double B_CH_VAR_INV_G = 2997.252804;
    
    // Yellow
    static const double R_CH_MEAN_Y = 0.370424749674244; //0.365420145384165;
    static const double G_CH_MEAN_Y = 0.389760396043991; //0.368234741618961;
    static const double B_CH_MEAN_Y = 0.237515614743063; //0.264019571080339;
    static const double R_CH_VAR_INV_Y = 13177.31862573654; //30183.45063686865;
    static const double G_CH_VAR_INV_Y = 16948.79514704690; //27203.35932348270;
    static const double B_CH_VAR_INV_Y = 6433.35013857765; //10423.22968747357;
//    static const double R_CH_MEAN_Y = 0.345987;
//    static const double G_CH_MEAN_Y = 0.361636;
//    static const double B_CH_MEAN_Y = 0.290413;
//    static const double R_CH_VAR_INV_Y = 1628.311875;
//    static const double G_CH_VAR_INV_Y = 743.681944;
//    static const double B_CH_VAR_INV_Y = 283.452942;
        
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
