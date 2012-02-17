#include <sstream>
#include <vector>
#include "cv.h"
#include "Gesture.h"
#include "LargePrint.h"
#include "highgui.h"

using namespace cv;

int main(int argc, char** argv) {
  Gesture gesture(argc, argv);
  return 0;
}

Gesture::Gesture(int argc, char** argv) {
  template32Matrix = imread("template-32.ppm", 0);
  template48Matrix = imread("template-48.ppm", 0);
  template64Matrix = imread("template-64.ppm", 0);

  // Normalize 0 to 1 so matching scores are in usable range
  template32Matrix.convertTo(template32Matrix, CV_32FC1, 1.0 / 255.0, 0);
  template48Matrix.convertTo(template48Matrix, CV_32FC1, 1.0 / 255.0, 0);
  template64Matrix.convertTo(template64Matrix, CV_32FC1, 1.0 / 255.0, 0);

  fprintf(stderr, "Hot keys: \n\tESC - quit the program\n");
  fprintf(stderr, "\ts   - save current input frame\n");

  cvNamedWindow("Input", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Process", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Output", CV_WINDOW_AUTOSIZE);

  if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0]))) {
    int index = argc == 2 ? argv[1][0] - '0' : 0;
    processStream(index);
  } else if (argc == 2 && strlen(argv[1]) > 0 && !isdigit(argv[1][0])) {
    processFile(argv[1]);
  } else if (argc == 4 && strlen(argv[1]) > 0 && !isdigit(argv[1][0])) {
    analyzeFile(argv[1], argv[2], argv[3]);
  }
  
  cvDestroyWindow("Input");
  cvDestroyWindow("Process");
  cvDestroyWindow("Output");
  return;
}

void Gesture::processStream(int index) {
  // Run gesture recognition on a video stream
  int c, save = 0;
  std::stringstream debug;
  int bufferIndex = 0, bufferModeCount, bufferModeIndex, gesture;
  int gestureBuffer[BUFFER_SIZE], modeCounter[BUFFER_SIZE];
  for (int i = 0; i < BUFFER_SIZE; i++) {
    gestureBuffer[i] = 0;
    modeCounter[i] = 0;
  }
  Vector<Point> greenCircles32, greenCircles48, greenCircles64, redCircles32, redCircles48, redCircles64;
  bool display, firstPass = true;
  CvCapture* capture = cvCreateCameraCapture(index);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
  cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);

  if (!capture) {
    fprintf(stderr, "Could not initialize capturing.\n");
    return;
  }

  for (;;) {
    // Frame capture
    frameImage = cvQueryFrame(capture);
    if (!frameImage)
      break;
    frameMatrix = cvarrToMat(frameImage);
    display = true;

    // Pre-processing
    medianBlur(frameMatrix, frameMatrix, 5);
    GaussianBlur(frameMatrix, frameMatrix, Size(9, 9), 1, 1);

    // Colorspace processing
    cvtColor(frameMatrix, hsvMatrix, CV_BGR2HSV);
    applyTableHSV(hsvMatrix, greenMatrix, H_MIN_G, H_MAX_G, S_MIN_G, S_MAX_G, 0, 0);
    if (H_MIN_R1 != H_MIN_R2) {
      applyTableHSV(hsvMatrix, tempMatrix, H_MIN_R1, H_MAX_R1, S_MIN_R1, S_MAX_R1, 0, 0);
      applyTableHSV(hsvMatrix, redMatrix, H_MIN_R2, H_MAX_R2, S_MIN_R2, S_MAX_R2, 0, 0);
      add(tempMatrix, redMatrix, redMatrix);
    } else {
      applyTableHSV(hsvMatrix, redMatrix, H_MIN_R1, H_MAX_R1, S_MIN_R1, S_MAX_R1, 0, 0);
    }

    // Circle detection
    // Normalize 0 to 1 so matching scores are in usable range
    greenMatrix.convertTo(greenMatrix, CV_32FC1, 1.0 / 255.0, 0);
    redMatrix.convertTo(redMatrix, CV_32FC1, 1.0 / 255.0, 0);
    Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, greenMatrix.type(), Scalar(0));
    vector<Mat> bgr;
    findCircles(redMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, redCircles64);
    findCircles(redMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, redCircles48);
    findCircles(redMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, redCircles32);
    findCircles(greenMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, greenCircles64);
    findCircles(greenMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, greenCircles48);
    findCircles(greenMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, greenCircles32);
    fprintf(stderr, "%lu green circles, %lu red circles.\n",
            (greenCircles32.size() + greenCircles48.size() + greenCircles64.size()),
            (redCircles32.size() + redCircles48.size() + redCircles64.size()));

    /*
     * Here we use a rolling window of size BUFFER_SIZE so that several 
     * frames of the same gesture must be captured to result in a command.
     * We calculate the mode and make sure the mode count is above 
     * MODE_MINIMUM.
     */
    // First check that we have a possible number of detected circles
    if ((greenCircles32.size() + greenCircles48.size() + greenCircles64.size()) <= MAX_GREEN_CIRCLES &&
            (redCircles32.size() + redCircles48.size() + redCircles64.size()) <= MAX_RED_CIRCLES) {
      // Store number as <number green circles><number red circles>
      gestureBuffer[bufferIndex] = 10 * (greenCircles32.size() + greenCircles48.size() + greenCircles64.size()) +
              redCircles32.size() + redCircles48.size() + redCircles64.size();
      bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
      // Calculate mode
      bufferModeCount = 0, bufferModeIndex = 0;
      for (int curNumIndex = 0; curNumIndex < BUFFER_SIZE; curNumIndex++) {
        modeCounter[curNumIndex] = 0;
        for (int i = 0; i < BUFFER_SIZE; i++) {
          if (gestureBuffer[i] == gestureBuffer[curNumIndex]) {
            modeCounter[curNumIndex]++;
          }
        }
        if (modeCounter[curNumIndex] > bufferModeCount) {
          bufferModeIndex = curNumIndex;
          bufferModeCount = modeCounter[curNumIndex];
        }
      }
      // Check against minimum mode count
      if (bufferModeCount >= MODE_MINIMUM) {
        gesture = gestureBuffer[bufferModeIndex];
        // Print out mode number
        LargePrint::largePrint(gesture);
      }
    }

    // Output images
    // Segmentation image
    bgr.push_back(blueMatrix);
    bgr.push_back(greenMatrix);
    bgr.push_back(redMatrix);
    merge(bgr, tempMatrix);
    // Box detected circles in images
    drawSquares(frameMatrix, greenCircles32, template32Matrix.cols, Scalar(0, 255, 0));
    drawSquares(frameMatrix, greenCircles48, template48Matrix.cols, Scalar(0, 255, 0));
    drawSquares(frameMatrix, greenCircles64, template64Matrix.cols, Scalar(0, 255, 0));
    drawSquares(frameMatrix, redCircles32, template32Matrix.cols, Scalar(0, 0, 255));
    drawSquares(frameMatrix, redCircles48, template48Matrix.cols, Scalar(0, 0, 255));
    drawSquares(frameMatrix, redCircles64, template64Matrix.cols, Scalar(0, 0, 255));
    drawSquares(tempMatrix, greenCircles32, template32Matrix.cols, Scalar(0, 255, 0));
    drawSquares(tempMatrix, greenCircles48, template48Matrix.cols, Scalar(0, 255, 0));
    drawSquares(tempMatrix, greenCircles64, template64Matrix.cols, Scalar(0, 255, 0));
    drawSquares(tempMatrix, redCircles32, template32Matrix.cols, Scalar(0, 0, 255));
    drawSquares(tempMatrix, redCircles48, template48Matrix.cols, Scalar(0, 0, 255));
    drawSquares(tempMatrix, redCircles64, template64Matrix.cols, Scalar(0, 0, 255));

    // Display values for center pixel
    debug.str("");
    debug << (int) (hsvMatrix.at<Vec3b > (240, 320)[0]) << "," <<
            (int) (hsvMatrix.at<Vec3b > (240, 320)[1]) << "," <<
            (int) (hsvMatrix.at<Vec3b > (240, 320)[2]);
    putText(frameMatrix, debug.str(), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
    putText(tempMatrix, debug.str(), Point(0, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
    debug.str("");
    debug << (int) frameMatrix.at<Vec3b > (240, 320)[2] << "," <<
            (int) frameMatrix.at<Vec3b > (240, 320)[1] << "," <<
            (int) frameMatrix.at<Vec3b > (240, 320)[0];
    putText(frameMatrix, debug.str(), Point(0, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255));
    putText(tempMatrix, debug.str(), Point(0, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
    rectangle(frameMatrix, Point(239, 319), Point(241, 321), Scalar(0, 0, 255), CV_FILLED, 8, 0);
    rectangle(tempMatrix, Point(239, 319), Point(241, 321), Scalar(0, 0, 255), CV_FILLED, 8, 0);

    // Display results
    tempImage = tempMatrix;
    normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
    outImage = outputMatrix;
    if (display) {
      cvShowImage("Input", frameImage);
      cvShowImage("Process", &tempImage);
      cvShowImage("Output", &outImage);
    }

    // Poll for input before looping
    c = cvWaitKey(10);
    if ((char) c == 27) {
      break;
    } else if ((char) c == 's') {
      std::stringstream ss;
      ss << "image-" << save << ".ppm";
      cvSaveImage(ss.str().data(), frameImage);
      save++;
    }
    if (firstPass) {
      firstPass = false;
    }
    // Clear detected circle list each cycle
    greenCircles32.clear();
    greenCircles48.clear();
    greenCircles64.clear();
    redCircles32.clear();
    redCircles48.clear();
    redCircles64.clear();
  }
  // Destroy and quit
  cvReleaseCapture(&capture);
}

void Gesture::processFile(const char* fileName) {
  // Run gesture recognition on a single file
  Vector<Point> greenCircles32, greenCircles48, greenCircles64, redCircles32, redCircles48, redCircles64;
  // Load image
//  std::cout << "File is " << argv[1] << std::endl;
  std::cout << "File is " << fileName << std::endl;
//  frameImage = cvLoadImage(argv[1]);
  frameImage = cvLoadImage(fileName);
  frameMatrix = cvarrToMat(frameImage);
  if (DEBUG) {
    cvWaitKey(0);
  }

  // Pre-processing
  medianBlur(frameMatrix, frameMatrix, 5);
  GaussianBlur(frameMatrix, frameMatrix, Size(9, 9), 1, 1);

  // Colorspace processing
  cvtColor(frameMatrix, hsvMatrix, CV_BGR2HSV);
  applyTableHSV(hsvMatrix, greenMatrix, H_MIN_G, H_MAX_G, S_MIN_G, S_MAX_G, 0, 0);
  if (H_MIN_R1 != H_MIN_R2) {
    applyTableHSV(hsvMatrix, tempMatrix, H_MIN_R1, H_MAX_R1, S_MIN_R1, S_MAX_R1, 0, 0);
    applyTableHSV(hsvMatrix, redMatrix, H_MIN_R2, H_MAX_R2, S_MIN_R2, S_MAX_R2, 0, 0);
    add(tempMatrix, redMatrix, redMatrix);
  } else {
    applyTableHSV(hsvMatrix, redMatrix, H_MIN_R1, H_MAX_R1, S_MIN_R1, S_MAX_R1, 0, 0);
  }
  if (H_MIN_O1 != H_MIN_O2) {
    applyTableHSV(hsvMatrix, tempMatrix, H_MIN_O1, H_MAX_O1, S_MIN_O1, S_MAX_O1, 0, 0);
    applyTableHSV(hsvMatrix, orangeMatrix, H_MIN_O2, H_MAX_O2, S_MIN_O2, S_MAX_O2, 0, 0);
    add(tempMatrix, orangeMatrix, orangeMatrix);
  } else {
    applyTableHSV(hsvMatrix, orangeMatrix, H_MIN_O1, H_MAX_O1, S_MIN_O1, S_MAX_O1, 0, 0);
  }
  std::cout << "greenMatrix is type " << greenMatrix.type() << std::endl;
  std::cout << "greenMatrix[1, 1] is " << greenMatrix.at<uchar > (1, 1) << std::endl;
  fprintf(stderr, "greenMatrix[1, 1] is %d\n", greenMatrix.at<uchar > (1, 1));

  findCCL(orangeMatrix, cclMatrix, true);

  // Circle detection
  // Normalize 0 to 1 so matching scores are in usable range
  greenMatrix.convertTo(greenMatrix, CV_32FC1, 1.0 / 255.0, 0);
  redMatrix.convertTo(redMatrix, CV_32FC1, 1.0 / 255.0, 0);
  Mat blueMatrix(greenMatrix.rows, greenMatrix.cols, greenMatrix.type(), Scalar(0));
  vector<Mat> bgr;
  findCircles(redMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, redCircles64);
  findCircles(redMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, redCircles48);
  findCircles(redMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, redCircles32);
  findCircles(greenMatrix, outputMatrix, template64Matrix, THRESH_TEMPLATE_64, greenCircles64);
  findCircles(greenMatrix, outputMatrix, template48Matrix, THRESH_TEMPLATE_48, greenCircles48);
  findCircles(greenMatrix, outputMatrix, template32Matrix, THRESH_TEMPLATE_32, greenCircles32);
  fprintf(stderr, "%lu green circles, %lu red circles.\n",
          (greenCircles32.size() + greenCircles48.size() + greenCircles64.size()),
          (redCircles32.size() + redCircles48.size() + redCircles64.size()));

  // Output images
  // Segmentation image
  bgr.clear();
  bgr.push_back(blueMatrix);
  bgr.push_back(greenMatrix);
  bgr.push_back(redMatrix);
  merge(bgr, tempMatrix);
  // Box detected circles in images
  drawSquares(frameMatrix, greenCircles32, template32Matrix.cols, Scalar(0, 255, 0));
  drawSquares(frameMatrix, greenCircles48, template48Matrix.cols, Scalar(0, 255, 0));
  drawSquares(frameMatrix, greenCircles64, template64Matrix.cols, Scalar(0, 255, 0));
  drawSquares(frameMatrix, redCircles32, template32Matrix.cols, Scalar(0, 0, 255));
  drawSquares(frameMatrix, redCircles48, template48Matrix.cols, Scalar(0, 0, 255));
  drawSquares(frameMatrix, redCircles64, template64Matrix.cols, Scalar(0, 0, 255));
  drawSquares(tempMatrix, greenCircles32, template32Matrix.cols, Scalar(0, 255, 0));
  drawSquares(tempMatrix, greenCircles48, template48Matrix.cols, Scalar(0, 255, 0));
  drawSquares(tempMatrix, greenCircles64, template64Matrix.cols, Scalar(0, 255, 0));
  drawSquares(tempMatrix, redCircles32, template32Matrix.cols, Scalar(0, 0, 255));
  drawSquares(tempMatrix, redCircles48, template48Matrix.cols, Scalar(0, 0, 255));
  drawSquares(tempMatrix, redCircles64, template64Matrix.cols, Scalar(0, 0, 255));

  // Display results
  //    tempImage = tempMatrix;
  tempImage = orangeMatrix;
  //    normalize(outputMatrix, outputMatrix, 0, 1, NORM_MINMAX, -1, Mat());
  //    outImage = outputMatrix;
  outImage = cclMatrix;
  cvShowImage("Input", frameImage);
  cvShowImage("Process", &tempImage);
  cvShowImage("Output", &outImage);

  // Pause before quitting
  char c = cvWaitKey();
  while ((char) c != 27) {
    c = cvWaitKey();
  }
}

void Gesture::analyzeFile(const char *fileName, const char *suffix1, const char *suffix2) {
  // OpenCV (uint) / OpenCV (float) / GIMP
  // H: [0,180] / [0,360] / [0,360]
  // S: [0,255] / [0,1] / [0,100]
  // V: [0,255] / [0,1] / [0,100]
  // We are calculating the thresholds in OpenCV *FLOAT* HSV format
  // We are analyzing images in OpenCV *UINT* HSV format
  // Manual thehold checks in GIMP are in *GIMP* HSV format
  // They are all different so don't forget the conversions!
  double splitPointUInt = 90.0;

  // Calculate image statistics on a single file
  double hMin1, hMax1, sMin1, sMax1, hMin2, hMax2, sMin2, sMax2;
  // Load image
  std::cout << "File is " << fileName << std::endl;
  frameImage = cvLoadImage(fileName);
  frameMatrix = cvarrToMat(frameImage);

  // Colorspace processing
  frameMatrix.convertTo(frameMatrix, CV_32FC3);
  cvtColor(frameMatrix, hsvMatrix, CV_BGR2HSV);

//  if (strcmp(argv[2], argv[3]) == 0) {
  if (strcmp(suffix1, suffix2) == 0) {
    // Use for things with valid entries at only one point in HSV spectrum
    // both sets of min/maxes are printed out, but the program will on
    hMin1 = 360, hMax1 = 0, sMin1 = 1, sMax1 = 0;
  } else {
    // Use for things with valid entries at two points in HSV spectrum
    //    (ex: Red has entries near H = 0 and H = 180))
    // ___1 is for lower half of H spectrum, ___2 is for upper half
    hMin1 = splitPointUInt * 2, hMax1 = 0, sMin1 = 1, sMax1 = 0, hMin2 = 360, hMax2 = splitPointUInt * 2, sMin2 = 1, sMax2 = 0;
  }
  int valid = 0, invalid = 0;
  for (int i = 0; i < hsvMatrix.rows; i++) for (int j = 0; j < hsvMatrix.cols; j++) {
      if (hsvMatrix.at<Vec3f > (i, j)[0] != 0 && hsvMatrix.at<Vec3f > (i, j)[1] != 0 &&
              hsvMatrix.at<Vec3f > (i, j)[2] != 255) {
        valid++;
//        if (strcmp(argv[2], argv[3]) == 0) {
        if (strcmp(suffix1, suffix2) == 0) {
          if (hsvMatrix.at<Vec3f > (i, j)[0] < hMin1) {
            hMin1 = hsvMatrix.at<Vec3f > (i, j)[0];
          }
          if (hsvMatrix.at<Vec3f > (i, j)[0] > hMax1) {
            hMax1 = hsvMatrix.at<Vec3f > (i, j)[0];
          }
          if (hsvMatrix.at<Vec3f > (i, j)[1] < sMin1) {
            sMin1 = hsvMatrix.at<Vec3f > (i, j)[1];
          }
          if (hsvMatrix.at<Vec3f > (i, j)[1] > sMax1) {
            sMax1 = hsvMatrix.at<Vec3f > (i, j)[1];
          }
        } else {
          if (hsvMatrix.at<Vec3f > (i, j)[0] <= splitPointUInt) {
            if (hsvMatrix.at<Vec3f > (i, j)[0] < hMin1) {
              hMin1 = hsvMatrix.at<Vec3f > (i, j)[0];
            }
            if (hsvMatrix.at<Vec3f > (i, j)[0] > hMax1) {
              hMax1 = hsvMatrix.at<Vec3f > (i, j)[0];
            }
            if (hsvMatrix.at<Vec3f > (i, j)[1] < sMin1) {
              sMin1 = hsvMatrix.at<Vec3f > (i, j)[1];
            }
            if (hsvMatrix.at<Vec3f > (i, j)[1] > sMax1) {
              sMax1 = hsvMatrix.at<Vec3f > (i, j)[1];
            }
          } else {
            if (hsvMatrix.at<Vec3f > (i, j)[0] < hMin2) {
              hMin2 = hsvMatrix.at<Vec3f > (i, j)[0];
            }
            if (hsvMatrix.at<Vec3f > (i, j)[0] > hMax2) {
              hMax2 = hsvMatrix.at<Vec3f > (i, j)[0];
            }
            if (hsvMatrix.at<Vec3f > (i, j)[1] < sMin2) {
              sMin2 = hsvMatrix.at<Vec3f > (i, j)[1];
            }
            if (hsvMatrix.at<Vec3f > (i, j)[1] > sMax2) {
              sMax2 = hsvMatrix.at<Vec3f > (i, j)[1];
            }
          }
        }
      } else {
        invalid++;
      }
    }
  fprintf(stderr, "    static const double H_MIN_%s = %f; // (%f)\n", suffix1, hMin1 / 2.0, hMin1);
  fprintf(stderr, "    static const double H_MAX_%s = %f; // (%f)\n", suffix1, hMax1 / 2.0, hMax1);
  fprintf(stderr, "    static const double S_MIN_%s = %f; // (%f)\n", suffix1, sMin1 * 255.0, sMin1 * 100.0);
  fprintf(stderr, "    static const double S_MAX_%s = %f; // (%f)\n", suffix1, sMax1 * 255.0, sMax1 * 100.0);
  if (strcmp(suffix1, suffix2) != 0) {
    fprintf(stderr, "    static const double H_MIN_%s = %f; // (%f)\n", suffix2, hMin2 / 2.0, hMin2);
    fprintf(stderr, "    static const double H_MAX_%s = %f; // (%f)\n", suffix2, hMax2 / 2.0, hMax2);
    fprintf(stderr, "    static const double S_MIN_%s = %f; // (%f)\n", suffix2, sMin2 * 255.0, sMin2 * 100.0);
    fprintf(stderr, "    static const double S_MAX_%s = %f; // (%f)\n", suffix2, sMax2 * 255.0, sMax2 * 100.0);
  }
  fprintf(stderr, "VALID=%d\tINVALID=%d\n", valid, invalid);
}

void Gesture::nothing(const Mat& src, Mat& dst) {
  dst = src;
  return;
}

void Gesture::applyFlip(const Mat& src, Mat& dst) {
  flip(src, dst, 1);
  return;
}

void Gesture::applyMedian(const Mat& src, Mat& dst) {
  medianBlur(src, dst, 5);
  return;
}

void Gesture::applyTableHSV(const Mat& src, Mat& dst, double hMin, double hMax, double sMin, double sMax, double vMin, double vMax) {
  // Separate channels into single channel float matrices
  vector<Mat> hsv;
  split(src, hsv);
  // Apply min
  Mat hMinT, hMaxT, sMinT, sMaxT, vMinT, vMaxT;
  threshold(hsv[0], hMinT, hMin, 255, THRESH_BINARY);
  threshold(hsv[1], sMinT, sMin, 255, THRESH_BINARY);
  //  threshold(hsv[2], vMinT, vMin, 255, THRESH_BINARY);
  // Apply max
  threshold(hsv[0], hMaxT, hMax, 255, THRESH_BINARY_INV);
  threshold(hsv[1], sMaxT, sMax, 255, THRESH_BINARY_INV);
  //  threshold(hsv[2], vMaxT, vMax, 255, THRESH_BINARY_INV);
  // OR the min and max
  Mat hT, sT, vT;
  bitwise_and(hMinT, hMaxT, hT);
  bitwise_and(sMinT, sMaxT, sT);
  //  bitwise_and(vMinT, vMaxT, vT);
  // AND the OR results
  bitwise_and(hT, sT, dst);

  return;
}

void Gesture::findCCL(const Mat& inputMatrix, Mat& processMatrix, bool considerDiagonals) {
  Mat labelMatrix = Mat::zeros(inputMatrix.rows, inputMatrix.cols, CV_32S);
  processMatrix = Mat::zeros(inputMatrix.rows, inputMatrix.cols, CV_8U);
  // Count of occurrences of each label 
  std::vector<uint > labelCount;
  // Keeps track of smallest label that each label is connected to - used in labeling merging step
  std::vector<uint > labelCorres;
  // Labels of neighbors
  std::vector<uint > neighborLabels;
  bool hasNeighbors;
  // The latest new label that was created
  uint minLabel, newestLabel = 0, largestLabelIndex = 0, largestLabelCount = 0;

  for (int r = 1; r < labelMatrix.rows - 1; r++) {
    for (int c = 1; c < labelMatrix.cols - 1; c++) {
      if (inputMatrix.at<uchar > (r, c) != 0) {
        hasNeighbors = false;
        neighborLabels.clear();
        // Check for neighbors and record their labels
        if (labelMatrix.at<uint > (r, c - 1) != 0) { // W
          neighborLabels.push_back(labelMatrix.at<uint > (r, c - 1));
          hasNeighbors = true;
        }
        if (labelMatrix.at<uint > (r - 1, c) != 0) { // N
          neighborLabels.push_back(labelMatrix.at<uint > (r - 1, c));
          hasNeighbors = true;
        }
        if (considerDiagonals) {
          if (labelMatrix.at<uint > (r - 1, c - 1) != 0) { // NW
            neighborLabels.push_back(labelMatrix.at<uint > (r - 1, c - 1));
            hasNeighbors = true;
          }
          if (labelMatrix.at<uint > (r - 1, c + 1) != 0) { // NE
            neighborLabels.push_back(labelMatrix.at<uint > (r - 1, c + 1));
            hasNeighbors = true;
          }
        }
        if (!hasNeighbors) {
          // Use new label
          newestLabel++;
          labelMatrix.at<uint > (r, c) = newestLabel;
          // Record correspondence
          if (labelCorres.size() < newestLabel) {
            labelCorres.resize(newestLabel + 1, 0);
            // Initialize correspondence to self
            labelCorres[newestLabel] = newestLabel;
          } else {
            // This will happen if reserve decides to put in more space than we request for a previous new label
            // Initialize correspondence to self
            labelCorres[newestLabel] = newestLabel;
          }
        } else {
          // Use smallest of neighbor's labels
          minLabel = minimum(neighborLabels);
          labelMatrix.at<uint > (r, c) = minLabel;
          // Update correspondences
          if (labelMatrix.at<uint > (r, c - 1) != 0) { // W
            labelCorres[labelMatrix.at<uint > (r, c - 1)] = minLabel;
          }
          if (labelMatrix.at<uint > (r - 1, c) != 0) { // N
            labelCorres[labelMatrix.at<uint > (r - 1, c)] = minLabel;
          }
          if (considerDiagonals) {
            if (labelMatrix.at<uint > (r - 1, c - 1) != 0) { // NW
              labelCorres[labelMatrix.at<uint > (r - 1, c - 1)] = minLabel;
            }
            if (labelMatrix.at<uint > (r - 1, c + 1) != 0) { // NE
              labelCorres[labelMatrix.at<uint > (r - 1, c + 1)] = minLabel;
            }
          }
        }
        // Update label count
        if (labelMatrix.at<uint > (r, c) > labelCount.size()) {
          labelCount.resize(labelMatrix.at<uint > (r, c) + 1);
          labelCount[labelMatrix.at<uint > (r, c)] = 0;
        }
        labelCount[labelMatrix.at<uint > (r, c)]++;
      }
    }
  }
  // Remap corresponding labels
  uint curCorres, degrees, maxDegrees = 0;
  for (int r = 1; r < labelMatrix.rows - 1; r++) {
    for (int c = 1; c < labelMatrix.cols - 1; c++) {
      if ((labelMatrix.at<uint > (r, c) != 0) &&
              (labelCorres[labelMatrix.at<uint > (r, c)] != labelCorres[labelCorres[labelMatrix.at<uint > (r, c)]])) {
        curCorres = labelCorres[labelCorres[labelMatrix.at<uint > (r, c)]];
        degrees = 1;
        while (curCorres != labelCorres[curCorres]) {
          curCorres = labelCorres[curCorres];
          degrees++;
        }
        if (degrees > maxDegrees) {
          maxDegrees = degrees;
        }
        labelCorres[labelMatrix.at<uint > (r, c)] = curCorres;
      }
    }
  }
  // Final count of labels - skip 0 (background) label
  for (int i = 1; i < labelCount.size(); i++) {
    if (i != labelCorres[i]) {
      // Don't double-count self-labels
      labelCount[labelCorres[i]] += labelCount[i];
    }
    if (labelCount[labelCorres[i]] > largestLabelCount) {
      largestLabelCount = labelCount[labelCorres[i]];
      largestLabelIndex = labelCorres[i];
    }
  }

  // Debug visualization
  for (int r = 1; r < labelMatrix.rows - 1; r++) {
    for (int c = 1; c < labelMatrix.cols - 1; c++) {
      if (labelMatrix.at<uint > (r, c) != 0) {
        // Keep background set to 0
        if (labelCorres[labelMatrix.at<uint > (r, c)] == largestLabelIndex) {
          // Set pixels in largest label to 255
          processMatrix.at<uchar > (r, c) = 255;
        } else {
          // Set pixels in non-largest labels to 128
          processMatrix.at<uchar > (r, c) = 128;
        }
      }
    }
  }
}

void Gesture::findCentroid(const Mat& inputMatrix, float** stats) {

}

void Gesture::findCircles(Mat& src, Mat& dst, Mat& templ, double thresh, Vector<Point>& circles) {
  // Template matching
  matchTemplate(src, templ, dst, CV_TM_SQDIFF);
  // Find best matches
  double minVal;
  Point minLoc;
  minMaxLoc(dst, &minVal, NULL, &minLoc, NULL, Mat());
  //printf("Checking %lf < %lf at [%d,%d].\n", minVal, thresh, minLoc.x, minLoc.y);
  while (minVal < thresh) {
    // Set area around this point to 0s so we don't get duplicate matches
    rectangle(dst,
            Point(minLoc.x - templ.cols / 2, minLoc.y - templ.rows / 2),
            Point(minLoc.x + templ.cols / 2, minLoc.y + templ.rows / 2), Scalar(thresh), CV_FILLED, 8, 0);
    //        minLoc.x - templ.cols / 2, minLoc.y - templ.rows / 2, templ.cols);
    // Remember to compensate for the template size border taken out during template matching
    rectangle(src,
            Point(minLoc.x, minLoc.y),
            Point(minLoc.x + templ.cols, minLoc.y + templ.rows), Scalar(0), CV_FILLED, 8, 0);
    circles.push_back(Point(minLoc.x + templ.cols / 2, minLoc.y + templ.rows / 2));
    //fprintf(stderr, "src rectangle at [%d,%d] with length %d.\n",
    //        minLoc.x + templ.cols / 2, minLoc.y + templ.rows / 2, templ.cols);

    // Find next max
    minMaxLoc(dst, &minVal, NULL, &minLoc, NULL, Mat());
    //printf("Checking %lf < %lf at [%d,%d].\n", minVal, thresh, minLoc.x, minLoc.y);
  }
}

uint Gesture::minimum(const std::vector<uint> input) {
  if (input.size() < 1) {
    return -1;
  }
  uint minimum = input[0];
  for (uint i = 1; i < input.size(); i++) {
    if (input[i] < minimum) {
      minimum = input[i];
    }
  }
  return minimum;
}

void Gesture::drawSquares(Mat& src, Vector<Point>& circles, int length, Scalar color) {
  //fprintf(stderr, "Drawing %ld squares of length %d.\n", circles.size(), length);
  Point point;
  for (size_t i = 0; i < circles.size(); i++) {
    point = circles[i];
    rectangle(src,
            Point(point.x - length / 2, point.y - length / 2),
            Point(point.x + length / 2, point.y + length / 2), color, 2, 8, 0);
  }
}

void Gesture::showImages(const Mat& inputMatrix, const Mat& processMatrix, const Mat& outputMatrix) {
  IplImage inputImage, processImage, outputImage;
  if (!inputMatrix.empty()) {
    inputImage = inputMatrix;
    cvShowImage("Input", &inputImage);
    fprintf(stderr, "Input matrix is size [%d,%d]\n", inputMatrix.rows, inputMatrix.cols);
  }
  if (!processMatrix.empty()) {
    processImage = processMatrix;
    cvShowImage("Process", &processImage);
    fprintf(stderr, "Process matrix is size [%d,%d]\n", processMatrix.rows, processMatrix.cols);
  }
  if (!outputMatrix.empty()) {
    outputImage = outputMatrix;
    cvShowImage("Output", &outputImage);
    fprintf(stderr, "Output matrix is size [%d,%d]\n", outputMatrix.rows, outputMatrix.cols);
  }
  // Pause before quitting
  int c = cvWaitKey();
  while ((char) c != 27) {
    c = cvWaitKey();
  }
  fprintf(stderr, "----------------------------\n");
}

void Gesture::printCVTypes() {
  fprintf(stderr, "CV_8UC1 is %d\n", CV_8UC1);
  fprintf(stderr, "CV_8UC2 is %d\n", CV_8UC2);
  fprintf(stderr, "CV_8UC3 is %d\n", CV_8UC3);
  fprintf(stderr, "CV_8UC4 is %d\n", CV_8UC4);
  fprintf(stderr, "\n");
  fprintf(stderr, "CV_8SC1 is %d\n", CV_8SC1);
  fprintf(stderr, "CV_8SC2 is %d\n", CV_8SC2);
  fprintf(stderr, "CV_8SC3 is %d\n", CV_8SC3);
  fprintf(stderr, "CV_8SC4 is %d\n", CV_8SC4);
  fprintf(stderr, "\n");
  fprintf(stderr, "CV_16UC1 is %d\n", CV_16UC1);
  fprintf(stderr, "CV_16UC2 is %d\n", CV_16UC2);
  fprintf(stderr, "CV_16UC3 is %d\n", CV_16UC3);
  fprintf(stderr, "CV_16UC4 is %d\n", CV_16UC4);
  fprintf(stderr, "\n");
  fprintf(stderr, "CV_16SC1 is %d\n", CV_16SC1);
  fprintf(stderr, "CV_16SC2 is %d\n", CV_16SC2);
  fprintf(stderr, "CV_16SC3 is %d\n", CV_16SC3);
  fprintf(stderr, "CV_16SC4 is %d\n", CV_16SC4);
  fprintf(stderr, "\n");
  fprintf(stderr, "CV_32SC1 is %d\n", CV_32SC1);
  fprintf(stderr, "CV_32SC2 is %d\n", CV_32SC2);
  fprintf(stderr, "CV_32SC3 is %d\n", CV_32SC3);
  fprintf(stderr, "CV_32SC4 is %d\n", CV_32SC4);
  fprintf(stderr, "\n");
  fprintf(stderr, "CV_32FC1 is %d\n", CV_32FC1);
  fprintf(stderr, "CV_32FC2 is %d\n", CV_32FC2);
  fprintf(stderr, "CV_32FC3 is %d\n", CV_32FC3);
  fprintf(stderr, "CV_32FC4 is %d\n", CV_32FC4);
  fprintf(stderr, "\n");
  fprintf(stderr, "CV_64FC1 is %d\n", CV_64FC1);
  fprintf(stderr, "CV_64FC2 is %d\n", CV_64FC2);
  fprintf(stderr, "CV_64FC3 is %d\n", CV_64FC3);
  fprintf(stderr, "CV_64FC4 is %d\n", CV_64FC4);
}

void Gesture::printInfo(const Mat& mat) {
  fprintf(stderr, "mat: %d %d %d %d %d\n", mat.rows, mat.cols, mat.channels(), mat.depth(), mat.type());
}