#include <sstream>
#include <vector>
#include "cv.h"
#include "Gesture.h"
#include "LargePrint.h"
#include "highgui.h"

using namespace cv;

int main(int argc, char** argv) {
  Processing processing(argc, argv);
  return 0;
}

Processing::Processing(int argc, char** argv) {
  return;
}

void Processing::applyInverse(const Mat& src, Mat& dst) {
  int width = src.cols;
  int height = src.rows;
  int channels = src.channels();
  int depth = src.depth();
  int step = src.step;
  uchar *data = src.data;
  uchar *postData;
  if (firstPass) {
    postFrame = cvCreateImage(cvSize(width, height), depth, channels);
    postData = (uchar *) postFrame->imageData;
  }
  for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) for (int k = 0; k < channels; k++) {
        postData[i * step + j * channels + k] = 255 - data[i * step + j * channels + k];
      }
  return;
}

void Processing::applyHistory(const Mat& src, Mat& prev, Mat& dst) {
  int width = src.cols;
  int height = src.rows;
  int depth = src.depth();
  if (firstPass) {
    procFrame = cvCreateImage(cvSize(width, height), depth, 1);
    postFrame = cvCreateImage(cvSize(width, height), depth, 1);
    //tr=128;
    //bitImage=cvCreateImage(cvSize(frame->width,frame->height),frame->depth,1);
    cvCvtColor(frameImage, procFrame, CV_BGR2GRAY);
    prevFrame = cvCloneImage(procFrame);
    return;
  }
  cvCvtColor(frameImage, procFrame, CV_BGR2GRAY);
  cvAbsDiff(prevFrame, procFrame, postFrame);
  //cvThreshold(postFrame,bitImage,tr,255,CV_THRESH_BINARY);
  //cvShowImage("CamSub 1",bitImage);   

  prevFrame = cvCloneImage(procFrame);

  return;
}

void Processing::applyChRG(const Mat& src, Mat& dst, double rMean, double gMean, double rSDI, double gSDI, double thresh) {
  // Separate channels into single channel float matrices
  vector<Mat> bgr;
  split(src, bgr);
  Mat rFloat, gFloat, bFloat;
  // BGR ordering
  bgr[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
  bgr[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
  bgr[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
  // Compute chromacity for r and g
  Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
  add(rFloat, gFloat, temp1);
  add(temp1, bFloat, temp2);
  add(temp2, Scalar(1.0), denom);
  divide(rFloat, denom, rChrom);
  divide(gFloat, denom, gChrom);
  divide(bFloat, denom, bChrom);

  // Compute gaussian probability pixel is on hand
  rChromV = rChrom.reshape(0, 1);
  gChromV = gChrom.reshape(0, 1);
  bChromV = bChrom.reshape(0, 1);
  // r
  subtract(rChromV, Scalar(rMean), temp1);
  multiply(temp1, Scalar(rSDI), temp2);
  rGauss = temp2.mul(temp1);
  // g
  subtract(gChromV, Scalar(gMean), temp1);
  multiply(temp1, Scalar(gSDI), temp2);
  gGauss = temp2.mul(temp1);
  add(rGauss, gGauss, d);

  multiply(d, Scalar(-0.5), expTerm);
  exp(expTerm, dst);
  dst = dst.reshape(0, 480);

  threshold(dst, dst, thresh, 255, THRESH_BINARY);

  return;
}

void Processing::applyChRB(const Mat& src, Mat& dst, double rMean, double bMean, double rSDI, double bSDI, double thresh) {
  // Separate channels into single channel float matrices
  vector<Mat> bgr;
  split(src, bgr);
  Mat rFloat, gFloat, bFloat;
  // BGR ordering
  bgr[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
  bgr[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
  bgr[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
  // Compute chromacity for r and g
  Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
  add(rFloat, gFloat, temp1);
  add(temp1, bFloat, temp2);
  add(temp2, Scalar(1.0), denom);
  divide(rFloat, denom, rChrom);
  divide(gFloat, denom, gChrom);
  divide(bFloat, denom, bChrom);

  // Compute gaussian probability pixel is on hand
  rChromV = rChrom.reshape(0, 1);
  gChromV = gChrom.reshape(0, 1);
  bChromV = bChrom.reshape(0, 1);
  // r
  subtract(rChromV, Scalar(rMean), temp1);
  multiply(temp1, Scalar(rSDI), temp2);
  rGauss = temp2.mul(temp1);
  // b
  subtract(bChromV, Scalar(bMean), temp1);
  multiply(temp1, Scalar(bSDI), temp2);
  bGauss = temp2.mul(temp1);
  add(rGauss, bGauss, d);

  multiply(d, Scalar(-0.5), expTerm);
  exp(expTerm, dst);
  dst = dst.reshape(0, 480);

  threshold(dst, dst, thresh, 255, THRESH_BINARY);

  return;
}

void Processing::applyChRGB(const Mat& src, Mat& dst, double rMean, double gMean, double bMean, double rSDI, double gSDI, double bSDI, double thresh) {
  // Separate channels into single channel float matrices
  vector<Mat> bgr;
  split(src, bgr);
  Mat rFloat, gFloat, bFloat;
  // BGR ordering
  bgr[2].convertTo(rFloat, CV_32FC1, 1.0, 0);
  bgr[1].convertTo(gFloat, CV_32FC1, 1.0, 0);
  bgr[0].convertTo(bFloat, CV_32FC1, 1.0, 0);
  // Compute chromacity for r and g
  Mat temp1, temp2, denom, rChrom, rChromV, gChrom, gChromV, bChrom, bChromV, rGauss, gGauss, bGauss, d, expTerm, outMatrix;
  add(rFloat, gFloat, temp1);
  add(temp1, bFloat, temp2);
  add(temp2, Scalar(1.0), denom);
  divide(rFloat, denom, rChrom);
  divide(gFloat, denom, gChrom);
  divide(bFloat, denom, bChrom);

  // Compute gaussian probability pixel is on hand
  rChromV = rChrom.reshape(0, 1);
  gChromV = gChrom.reshape(0, 1);
  bChromV = bChrom.reshape(0, 1);
  // r
  subtract(rChromV, Scalar(rMean), temp1);
  multiply(temp1, Scalar(rSDI), temp2);
  rGauss = temp2.mul(temp1);
  // g
  subtract(gChromV, Scalar(gMean), temp1);
  multiply(temp1, Scalar(gSDI), temp2);
  gGauss = temp2.mul(temp1);
  add(rGauss, gGauss, d);
  // b
  subtract(bChromV, Scalar(bMean), temp1);
  multiply(temp1, Scalar(bSDI), temp2);
  bGauss = temp2.mul(temp1);
  add(d, bGauss, d);

  multiply(d, Scalar(-0.5), expTerm);
  exp(expTerm, dst);
  dst = dst.reshape(0, 480);

  threshold(dst, dst, thresh, 255, THRESH_BINARY);

  return;
}

void Processing::applyGaussHSV(const Mat& src, Mat& dst, double hMean, double sMean, double vMean, double hSDI, double sSDI, double vSDI, double thresh) {
  // Separate channels into single channel float matrices
  cvtColor(src, dst, CV_BGR2HSV);
  vector<Mat> hsv;
  split(dst, hsv);
  Mat hFloat, sFloat, vFloat;
  hsv[0].convertTo(hFloat, CV_32FC1, 1.0, 0);
  hsv[1].convertTo(sFloat, CV_32FC1, 1.0, 0);
  hsv[2].convertTo(vFloat, CV_32FC1, 1.0, 0);

  // Compute gaussian probability pixel is on hand
  Mat hV, sV, vV, temp1, temp2, hGauss, sGauss, vGauss, d, expTerm;
  hV = hFloat.reshape(0, 1);
  sV = sFloat.reshape(0, 1);
  vV = vFloat.reshape(0, 1);
  // r
  subtract(hV, Scalar(hMean), temp1);
  multiply(temp1, Scalar(hSDI), temp2);
  hGauss = temp2.mul(temp1);
  // g
  subtract(sV, Scalar(sMean), temp1);
  multiply(temp1, Scalar(sSDI), temp2);
  sGauss = temp2.mul(temp1);
  add(hGauss, sGauss, d);
  // b
  subtract(vV, Scalar(vMean), temp1);
  multiply(temp1, Scalar(vSDI), temp2);
  vGauss = temp2.mul(temp1);
  add(d, vGauss, d);

  multiply(d, Scalar(-0.5), expTerm);
  exp(expTerm, dst);
  dst = dst.reshape(0, 480);
  threshold(dst, dst, thresh, 255, THRESH_BINARY);

  return;
}

void Processing::houghCircles(const Mat& src, Mat& dst, Mat& drawMatrix, Mat& templ) {
  // Smooth it, otherwise a lot of false circles may be detected
  //    GaussianBlur(src, dst, Size(9, 9), 4, 4);
  medianBlur(src, dst, 5);

  vector<Vec3f> circles;
  HoughCircles(src, circles, CV_HOUGH_GRADIENT, 2, 40, 200, 100);
  for (size_t i = 0; i < circles.size(); i++) {
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    // draw the circle center
    circle(drawMatrix, center, 3, Scalar(0, 255, 0), -1, 8, 0);
    // draw the circle outline
    circle(drawMatrix, center, radius, Scalar(0, 0, 255), 3, 8, 0);
  }
  fprintf(stderr, "Found %ld circles.\n", circles.size());
}