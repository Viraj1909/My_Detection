#ifndef COLOROBJECTDETECTOR_H
#define COLOROBJECTDETECTOR_H

#include "detectionLibrary.H"

class ColorObjectDetector : public DetectionLibrary
{
public:
    ColorObjectDetector();
    ~ColorObjectDetector();

    bool configuration(ColorConfigurationParameters parameters, PartitionDetectionConfigurationParameter partitionParameter, int height, int width);
    bool detect(cv::Mat &image, int &noOfObject, std::vector<cv::Rect> &boundingBox);
};

#endif // COLOROBJECTDETECTOR_H
