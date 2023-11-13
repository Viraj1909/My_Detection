#include "aiObjectDetector.H"

AIObjectDetector::AIObjectDetector() {}
AIObjectDetector::~AIObjectDetector() {}
bool AIObjectDetector::configuration(DetectionConfigurationParameter parameters, PartitionDetectionConfigurationParameter partitionParameter) {}
bool AIObjectDetector::detect(cv::Mat &image, std::map<int,std::vector<std::pair<cv::Rect,float>>> &objectInfoList, int &objectCount) {}
