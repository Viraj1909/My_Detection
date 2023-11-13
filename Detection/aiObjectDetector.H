#ifndef AIOBJECTDETECTOR_H
#define AIOBJECTDETECTOR_H

#include "detectionLibrary.H"

class AIObjectDetector : public DetectionLibrary
{
public:
    AIObjectDetector();
    ~AIObjectDetector() = 0;

    bool configuration(DetectionConfigurationParameter parameters, PartitionDetectionConfigurationParameter partitionParameter);
    bool detect(cv::Mat &image, std::map<int,std::vector<std::pair<cv::Rect,float>>> &objectInfoList, int &objectCount);
};

#endif // AIOBJECTDETECTOR_H
