#ifndef PROCESSING_H
#define	PROCESSING_H

#include <vector>

using namespace cv;

class Processing {
public:
    Processing(int, char **);
private:
    static const bool DEBUG = false;

    void applyInverse(const Mat& src, Mat& dst);
    void applyHistory(const Mat& src, Mat& prev, Mat& dst);
    void applyChRG(const Mat& src, Mat& dst, double rMean, double gMean, double rSDI, double gSDI, double thresh);
    void applyChRB(const Mat& src, Mat& dst, double rMean, double bMean, double rSDI, double bSDI, double thresh);
    void applyChRGB(const Mat& src, Mat& dst, double rMean, double gMean, double bMean, double rSDI, double gSDI, double bSDI, double thresh);
    void applyGaussHSV(const Mat& src, Mat& dst, double hMean, double sMean, double vMean, double hSDI, double sSDI, double vSDI, double thresh);
    void houghCircles(const Mat& src, Mat& dst, Mat& drawMatrix, Mat& templ);
};

#endif	/* GESTURE_H */