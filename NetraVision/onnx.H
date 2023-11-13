#ifndef ONNX_H
#define ONNX_H

#include "aiObjectDetector.H"
#include <fstream>
#include <string>

class Onnx: public AIObjectDetector
{
public:

    Onnx();
    ~Onnx();
    bool configuration(DetectionConfigurationParameter parameters, PartitionDetectionConfigurationParameter partitionParameter);
    bool detect(cv::Mat &image, std::map<int,std::vector<std::pair<cv::Rect,float>>> &objectInfoList, int &objectCount);
    void setError(ErrorCode code, const std::string &message);

private:
    bool fileExists(std::string& file);
    float sigmoid_x(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    ErrorDetails errorDetails;

    float nms = 0;
    float thresh=0;
    float threshHeir =0;

    cv::dnn::Net net_;
    std::vector<std::string> names;
    int size_=640;

    const float netAnchors[3][6] = { {12, 16, 19, 36, 40, 28},{36, 75, 76, 55, 72, 146},{142, 110, 192, 243, 459, 401} }; //yolov7-P5 anchors
    const int strideSize = 3;

    const float netStride[4] = { 8, 16.0,32,64 };

    std::vector<std::pair<cv::Rect, std::pair<float,int>>> detectionOutput_;

    float confidenceThreshold_;
    float nmsScoreThreshold = threshHeir * confidenceThreshold_;
};

#endif // ONNX_H
