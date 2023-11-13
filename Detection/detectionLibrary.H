#ifndef DETECTIONLIBRARY_H
#define DETECTIONLIBRARY_H

#include <iostream>
#include <chrono>
#include <vector>
#include <sys/stat.h>
#include <filesystem>

//CV_Detection
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/dnn.hpp>

namespace fs = std::filesystem;

class DetectionLibrary
{
public:
    struct PartitionDetectionConfigurationParameter
    {
        int numberOfPartitions=0;
        std::vector<int> partitionToDetect;
        bool partitionFlag=false;
    };
    struct DetectionConfigurationParameter
    {
        std::string cfgFile = "";
        std::string nameFile = "";
        std::string weightFile="";
        float nms = 0;
        float thresh=0;
        float threshHeir =0;
    };
    struct ColorRange
    {
        int lowChannel1;
        int lowChannel2;
        int lowChannel3;
        int highChannel1;  
        int highChannel2;
        int highChannel3;
    };
    struct ColorConfigurationParameters{
        std::vector<ColorRange> colorRanges;
        int minContourSize;
        int maxContourSize;
    };
    enum ErrorCode
    {
        NoError = 2000,
        DetectionError = 2001,
        classifierNotSelected = 2002,
        FileNotFound = 2003,
        ConfigurationError = 2004
    };
    struct ErrorDetails{
        ErrorCode errorcode = NoError;
        std::string errormsg ="";
    };

    DetectionLibrary();
    virtual ~DetectionLibrary() = 0;

    virtual bool configuration(DetectionConfigurationParameter parameters, PartitionDetectionConfigurationParameter partitionParameter);
    virtual bool configuration(ColorConfigurationParameters parameters, PartitionDetectionConfigurationParameter partitionParameter, int height, int width);

    virtual bool detect(cv::Mat &image, std::map<int,std::vector<std::pair<cv::Rect,float>>> &objectInfoList, int &objectCount);
    virtual bool detect(cv::Mat &image, int &noOfObject, std::vector<cv::Rect> &boundingBox);


};

#endif // DETECTIONLIBRARY_H
