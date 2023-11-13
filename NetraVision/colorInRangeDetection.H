#ifndef COLORINRANGEDETECTION_H
#define COLORINRANGEDETECTION_H

#include "colorObjectDetector.H"


class ColorInRangeDetection : public ColorObjectDetector
{
public:
    struct ColorDetectionDetails{
        std::vector<cv::Rect2f> rects;
        float areaFactor;
        cv::Mat maskedImage;
        double detectiontime =0;
        int detectedObjects=0;
    };
    ColorInRangeDetection();
    ~ColorInRangeDetection();

    bool configuration(ColorConfigurationParameters parameters,PartitionDetectionConfigurationParameter partitionParameter, int height, int width);
    bool detect(cv::Mat &image, int &noOfObject, std::vector<cv::Rect> &boundingBox);

private:
    ColorConfigurationParameters parameter;
    int heightPix=0, widthPix=0;
    ErrorDetails errorDetails;
};

#endif // COLORINRANGEDETECTION_H
