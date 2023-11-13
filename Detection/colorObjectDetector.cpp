#include "colorObjectDetector.H"

ColorObjectDetector::ColorObjectDetector() {}
ColorObjectDetector::~ColorObjectDetector() {}
bool ColorObjectDetector::configuration(ColorConfigurationParameters parameters, PartitionDetectionConfigurationParameter partitionParameter, int height, int width) {}
bool ColorObjectDetector::detect(cv::Mat &image, int &noOfObject, std::vector<cv::Rect> &boundingBox) {}
