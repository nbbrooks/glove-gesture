//#pragma once
using namespace cv;
        
class Gesture {
public:
    Gesture(int, char **);
private:
    // Brown
    // Measured mean chromacity values for red and green
    static const float R_CH_MEAN = 0.320960070321339;
    static const float G_CH_MEAN = 0.325954943146125;
    static const float B_CH_MEAN = 0.349575151609824;
    // Measured inverse std dev chromacity values for red and green
    static const float R_CH_VAR_INV = 1895.68502175326;
    static const float G_CH_VAR_INV = 11778.21112853496;
    static const float B_CH_VAR_INV = 1546.59181280968;
    
//    // White
//    // Measured mean chromacity values for red and green
//    static const float R_CH_MEAN = 0.331195010538250;
//    static const float G_CH_MEAN = 0.332107007988018;
//    static const float B_CH_MEAN = 0.334478031541126;
//    // Measured inverse std dev chromacity values for red and green   
//    static const float R_CH_VAR_INV = 10910.955085804;
//    static const float G_CH_VAR_INV = 55060.213621839;
//    static const float B_CH_VAR_INV = 8791.707209565;
    
//    // Blue
//    // Measured mean chromacity values for red and green
//    static const float R_CH_MEAN = 0.272419719355213;
//    static const float G_CH_MEAN = 0.308339748116777;
//    static const float B_CH_MEAN = 0.414705350685427;
//    // Measured inverse std dev chromacity values for red and green   
//    static const float R_CH_VAR_INV = 215.115336202807;
//    static const float G_CH_VAR_INV = 1236.405671747800;
//    static const float B_CH_VAR_INV = 119.176858592746;
    
//    // Skin
//    // Measured mean chromacity values for red and green
//    static const float R_CH_MEAN = 0.338260695104704;
//    static const float G_CH_MEAN = 0.319846092373669;
//    static const float B_CH_MEAN = 0.338931199942906;
//    // Measured inverse std dev chromacity values for red and green   
//    static const float R_CH_VAR_INV = 3851.96600707512;
//    static const float G_CH_VAR_INV = 5117.49006335319;
//    static const float B_CH_VAR_INV = 4196.26456951586;
    
//    // Measured mean chromacity values for red and green
//    static const float R_CH_MEAN = 0.3067;
//    static const float G_CH_MEAN = 0.2911;
//    static const float B_CH_MEAN = 0.0;
//    // Measured inverse std dev chromacity values for red and green
//    static const float R_CH_VAR_INV = 680.2268;
//    static const float G_CH_VAR_INV = 135.4993;
//    static const float B_CH_VAR_INV = 0.0;
        
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
    Mat skinMeanT, skinVarInv, outputMatrix;
//    Mat skinVarInv(2, 2, CV_8UC2, );
    
    void nothing(void);
    void applyInverse(void);
    void applyBackground(void);
    void applyHistory(void);
    void applyTemplate(void);
    void applyChRG(void);
    void applyChRGB(void);
    void applyChRB(void);
    void applyColorSegmentation(void);
    void calculateCentroid(void);
    void calculate(void);
};
