//#pragma once
using namespace cv;
        
class Gesture {
public:
    Gesture(int, char **);
private:
    static const float THRESH = 0.1f;
//    // Pink
//    // Measured mean chromacity values for red and green
//    static const float R_CH_MEAN = 0.413464326726511;
//    static const float G_CH_MEAN = 0.233825410563608;
//    static const float B_CH_MEAN = 0.350683233395741;
//    // Measured inverse std dev chromacity values for red and green
//    static const float R_CH_VAR_INV = 12856.5616746348;
//    static const float G_CH_VAR_INV = 17661.0926174714;
//    static const float B_CH_VAR_INV = 35748.0113745476;
    
//    // Green
//    // Measured mean chromacity values for red and green
//    static const float R_CH_MEAN = 0.279745482824469;
//    static const float G_CH_MEAN = 0.389173827230952;
//    static const float B_CH_MEAN = 0.328717603989065;
//    // Measured inverse std dev chromacity values for red and green
//    static const float R_CH_VAR_INV = 27548.0876730553;
//    static const float G_CH_VAR_INV = 14208.2089616912;
//    static const float B_CH_VAR_INV = 23977.4250454378;
    
    // Yellow
    // Measured mean chromacity values for red and green
    static const float R_CH_MEAN = 0.365420145384165;
    static const float G_CH_MEAN = 0.368234741618961;
    static const float B_CH_MEAN = 0.264019571080339;
    // Measured inverse std dev chromacity values for red and green
    static const float R_CH_VAR_INV = 30183.45063686865;
    static const float G_CH_VAR_INV = 27203.35932348270;
    static const float B_CH_VAR_INV = 10423.22968747357;
        
    IplImage* frame;
    IplImage* prevFrame;
    IplImage* procFrame;
    IplImage* postFrame;
    IplImage outFrame;
    int height, width, step, depth, channels;
    int outheight, outwidth, outstep, outdepth, outchannels;
    int save;
    uchar *data;
    uchar *procData;
    uchar *postData;
    uchar *outdata;
    int i, j, k;
    bool firstPass, display;
    Mat skinMeanT, skinVarInv, inputMatrix, outputMatrix;
//    Mat skinVarInv(2, 2, CV_8UC2, );
    
    void nothing(void);
    void applyFlip(void);
    void applyMedian(void);
    void applyInverse(void);
    void applyBackground(void);
    void applyHistory(void);
    void applyTemplate(void);
    void applyChG(void);
    void applyChRG(void);
    void applyChRB(void);
    void applyChRGB(void);
    void applyColorSegmentation(void);
    void calculateCentroid(void);
    void calculate(void);
};
