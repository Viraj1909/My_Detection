#include <iostream>
#include "netravision.H"
using namespace std;
int main()
{
    NetraVision *object[2];

    std::vector<DetectionLibrary::DetectionConfigurationParameter> params(2);
    DetectionLibrary::PartitionDetectionConfigurationParameter partitionParameters;
    params[0].nms = 0.6;
    params[0].thresh = 0.6;
    params[0].threshHeir = 0.6;
    params[0].cfgFile = "/home/viraj/Document/ML/newcode/7Aug_All_Polymer/yolov4_PPP_Test.cfg";
    params[0].nameFile = "/home/viraj/Document/ML/newcode/7Aug_All_Polymer/PPP_Model.names";
    params[0].weightFile = "/home/viraj/Document/ML/newcode/7Aug_All_Polymer/yolov4_PPP_8000.weights";
    std::ifstream file(params[0].nameFile);
    std::string line;
    std::vector<int> noOfClass;
    noOfClass[0] = 0;
    std::vector<std::vector<std::string>> className;
    while (std::getline(file, line))
    {
        noOfClass[0] += 1;
        className[0].push_back(line);
    }

    params[1].nms = 0.6;
    params[1].thresh = 0.6;
    params[1].threshHeir = 0.6;
    params[1].cfgFile = "/home/viraj/Document/ML/newcode/Label_Unlabel/Label_Unlabel.cfg";
    params[1].nameFile = "/home/viraj/Document/ML/newcode/Label_Unlabel/label.names";
    params[1].weightFile = "/home/viraj/Document/ML/newcode/Label_Unlabel/Label_Unlabel_817000.weights";
    std::ifstream file2(params[1].nameFile);
    std::string line2;
    noOfClass[1] = 0;
    while (std::getline(file2, line2))
    {
        noOfClass[1] += 1;
        className[1].push_back(line);
    }
    params[1].nms = 0.6;
    params[1].thresh = 0.6;
    params[1].threshHeir = 0.6;
    params[1].cfgFile = "/home/viraj/Document/ML/newcode/Label_Unlabel/Label_Unlabel.cfg";
    params[1].nameFile = "/home/viraj/Document/ML/newcode/Label_Unlabel/label.names";
    params[1].weightFile = "/home/viraj/Document/ML/newcode/Label_Unlabel/Label_Unlabel_817000.weights";
    std::ifstream file2(params[1].nameFile);
    std::string line2;
    noOfClass[1] = 0;
    while (std::getline(file2, line2))
    {
        noOfClass[1] += 1;
    }

    partitionParameters.partitionFlag = false;
    partitionParameters.partitionToDetect = {0};
    partitionParameters.numberOfPartitions = 0;

    DetectionLibrary::ColorConfigurationParameters Nparams;
    // shriChakra-RGB
    Nparams.minContourSize = 5000;
    Nparams.maxContourSize = 100000;
    DetectionLibrary::ColorRange ranges;
    ranges.lowChannel1 = 0;
    ranges.lowChannel2 = 0;
    ranges.lowChannel3 = 95;
    ranges.highChannel1 = 255;
    ranges.highChannel2 = 255;
    ranges.highChannel3 = 255;
    Nparams.colorRanges.push_back(ranges);

    NetraVision::imageServiceParameter para;
    para.saveImageFilePath = "/home/viraj/Document/Viraj/test_folder/rawImg/";

    std::string Derror = "", Cerror = "";
    for (int i = 0; i < 2; i++)
    {
        object[i]->imageServiceConfiguration(para);
        object[i]->detectionConfiguration(NetraVision::ObjectDetection, params[i], partitionParameters, Derror);
        object[i]->colorConfiguration(NetraVision::ColorInRangeDetection, Nparams, partitionParameters, 40, 40, Cerror);
    }

    if (!Derror.empty() && !Cerror.empty())
    {
        std::cout << "this is error->" << Derror << "and" << Cerror << std::endl;
    }

    std::vector<cv::String> imagePaths;
    cv::glob("/home/viraj/Document/ML/newcode/data/*.jpg", imagePaths);

    std::cout << "this is main thread -> " << std::this_thread::get_id() << std::endl;
    int DmainCount = 0, CmainCount = 0;
    for (size_t image = 0; image < 50; image++)
    {
        for(int run=0;run<2;run++){
        object[run]->setSessionNumber(image); // will set the session number
        std::cout << "image passed -> " << image << std::endl;
        int darknetCount = -1, colorCount = -1;
        cv::Mat img = cv::imread(imagePaths[image]);
        std::map<int, std::vector<std::pair<cv::Rect, float>>> darknetResult;
        std::vector<cv::Rect> colorResult;
        std::string detErr = "";
        auto startTime = std::chrono::high_resolution_clock::now();
        object[run]->detectNetraVision(img, darknetResult, darknetCount, colorResult, colorCount, true, true, detErr);
        auto stopTime = std::chrono::high_resolution_clock::now();
        if (!detErr.empty())
        {
            std::cout << "this is detection error->" << detErr << std::endl;
        }
        for (int i = 0; i < noOfClass[run]; i++)
        {
            for (size_t j = 0; j < darknetResult[i].size(); j++)
            {
                cv::rectangle(img, darknetResult[i][j].first, cv::Scalar(0, 255, 0), 2);
                std::stringstream probStr;
                probStr << darknetResult[i][j].second;
                std::string text = className[i][run] + probStr.str();
                cv::putText(img, text, cv::Point(darknetResult[i][j].first.x, darknetResult[i][j].first.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 1);
            }
        }
        for (const cv::Rect &rect : colorResult)
        {
            cv::rectangle(img, rect, cv::Scalar(255, 0, 0), 2);
        }
        cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
        cv::imshow("display frame", img);
        cv::waitKey(250);
        DmainCount += darknetCount;
        CmainCount += colorCount;
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
        std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;
        }
    }
    for (int i = 0; i < 2; ++i) {
        delete object[i];
    }
    return 0;
}
