//#pragma once
using namespace cv;
        
class Gesture {
public:
    Gesture(int, char **);
private:
    static const uint R_MEAN = 93;
    static const uint G_MEAN = 89;
    static const uint R_VAR = 254;
    static const uint G_VAR = 171;
    IplImage* frame;
    IplImage* prevFrame;
    IplImage* procFrame;
    IplImage* postFrame;
    int height, width, step, depth, channels;
    int save;
    uchar *data;
    uchar *procData;
    uchar *postData;
    int i, j, k;
    bool firstPass;
    Mat skinMeanT, skinVarInv;
//    Mat skinVarInv(2, 2, CV_8UC2, );
    
    void nothing(void);
    void applyInverse(void);
    void applyBackground(void);
    void applyHistory(void);
    void applyTemplate(void);
    void applySkin(void);
    void applyColorSegmentation(void);
    void calculateCentroid(void);
    void calculate(void);
};
