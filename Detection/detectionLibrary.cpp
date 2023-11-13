#include "detectionLibrary.H"

DetectionLibrary::DetectionLibrary() {}
bool DetectionLibrary::configuration(DetectionConfigurationParameter parameters, PartitionDetectionConfigurationParameter partitionParameter){}
bool DetectionLibrary::configuration(ColorConfigurationParameters parameters, PartitionDetectionConfigurationParameter partitionParameter, int height, int width) {}
bool DetectionLibrary::detect(cv::Mat &image, std::map<int,std::vector<std::pair<cv::Rect,float>>> &objectInfoList, int &objectCount) {}
bool DetectionLibrary::detect(cv::Mat &image, int &noOfObject, std::vector<cv::Rect> &boundingBox) {}
DetectionLibrary::~DetectionLibrary() {}
