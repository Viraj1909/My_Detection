#ifndef YOLO_H
#define YOLO_H

#include "aiObjectDetector.H"

#include "darknet/darknet.h"
#include "darknet/parser.h"

#include <fstream>


class Yolo : public AIObjectDetector
{
public:

    Yolo();
    ~Yolo();

    bool configuration(DetectionConfigurationParameter parameters, PartitionDetectionConfigurationParameter partitionParameter);
    bool detect(cv::Mat &image, std::map<int,std::vector<std::pair<cv::Rect,float>>> &objectInfoList, int &objectCount);

private:
    bool fileExists(std::string& file);

    network *net =nullptr;
    std::vector<float> probability;

    ErrorDetails errorDetails;
    PartitionDetectionConfigurationParameter partitionParameter;
    int noOfClass = 0;
    int numObjects = 0;

    float nms = 0;
    float thresh=0;
    float threshHeir =0;
};

#endif // YOLO_H
