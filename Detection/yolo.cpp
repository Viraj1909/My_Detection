#include "yolo.H"

Yolo::Yolo()
{
    setlocale(LC_NUMERIC, "C");
}
Yolo::~Yolo()
{
    if (net != nullptr)
    {
        free_network(*net);
        net = nullptr;
    }
    probability.clear();
}

bool Yolo::configuration(DetectionConfigurationParameter parameters, PartitionDetectionConfigurationParameter partitionPara)
{
    try
    {
        nms = parameters.nms;
        thresh = parameters.thresh;
        threshHeir = parameters.threshHeir;

        if (!fileExists(parameters.cfgFile))
        {
            errorDetails.errorcode = FileNotFound;
            errorDetails.errormsg = ".cfg file not found";
            return false;
        }
        if (!fileExists(parameters.weightFile))
        {
            errorDetails.errorcode = FileNotFound;
            errorDetails.errormsg = ".weight file not found";
            return false;
        }
        if (!fileExists(parameters.nameFile))
        {
            errorDetails.errorcode = FileNotFound;
            errorDetails.errormsg = ".names file not found";
            return false;
        }
        std::ifstream file(parameters.nameFile);
        std::string line;
        while (std::getline(file, line))
        {
            noOfClass += 1;
        }

        net = (network *)xcalloc(1, sizeof(network));
        *net = parse_network_cfg_custom(const_cast<char *>(parameters.cfgFile.c_str()), 1, 1);
        load_weights(net, const_cast<char *>(parameters.weightFile.c_str()));

        partitionParameter = partitionPara;
    }
    catch (std::exception &e)
    {
        errorDetails.errorcode = ConfigurationError;
        errorDetails.errormsg = e.what();
        return false;
    }
    catch (...)
    {
        errorDetails.errorcode = ConfigurationError;
        errorDetails.errormsg = "Default Exception was catched";
        return false;
    }
    return true;
}
bool Yolo::detect(cv::Mat &matImage, std::map<int, std::vector<std::pair<cv::Rect, float>>> &objectInfoList, int &objectCount)
{
    try
    {
        detection *detections = nullptr;
        int nboxes = 0;

        if (net)
        {
            cv::Mat inputRgb;

            if (partitionParameter.partitionFlag == true)
            {
                int imageWidth = matImage.cols;
                int imageHeight = matImage.rows;
                int desiredWidth = imageWidth / partitionParameter.numberOfPartitions;

                for (int part : partitionParameter.partitionToDetect)
                {
                    // Calculate the ROI for the current partition
                    int startX = part * desiredWidth;
                    int endX = (part + 1) * desiredWidth;
                    if (endX > imageWidth)
                    {
                        endX = imageWidth;
                    }
                    cv::Rect roiRect(startX, 0, endX - startX, imageHeight);
                    cv::Mat portion = matImage(roiRect);

                    // Perform detection on the 'portion' of the image
                    cvtColor(portion, inputRgb, cv::COLOR_BGR2RGB);
                    image portionDarknetImage = make_image(portion.cols, portion.rows, 3);
                    copy_image_from_bytes(portionDarknetImage, (char *)inputRgb.data);
                    network_predict_image_letterbox(net, portionDarknetImage);

                    // Get detections for the 'portion' of the image
                    nboxes = 0;
                    detections = get_network_boxes(net, portion.cols, portion.rows, thresh, threshHeir, nullptr, 1, &nboxes, 1);

                    if (nms)
                    {
                        do_nms_sort(detections, nboxes, noOfClass, nms);
                    }

                    // Process the detections and store them in 'objectInfoList'
                    for (int i = 0; i < nboxes; i++)
                    {
                        for (int j = 0; j < noOfClass; ++j)
                        {
                            if (detections[i].prob[j] > thresh)
                            {
                                objectInfoList[j].push_back(std::make_pair(cv::Rect((startX + (detections[i].bbox.x - detections[i].bbox.w / 2) * portion.cols), ((detections[i].bbox.y - detections[i].bbox.h / 2) * portion.rows), (detections[i].bbox.w * portion.cols), (detections[i].bbox.h * portion.rows)), detections[i].prob[j]));
                                objectCount++;
                            }
                        }
                    }

                    free_detections(detections, nboxes);
                }
            }
            else
            {
                // Perform detection on the entire 'matImage'
                cvtColor(matImage, inputRgb, cv::COLOR_BGR2RGB);
                image darknetImage = make_image(matImage.cols, matImage.rows, 3);
                copy_image_from_bytes(darknetImage, (char *)inputRgb.data);
                network_predict_image_letterbox(net, darknetImage);

                nboxes = 0;
                detections = get_network_boxes(net, matImage.cols, matImage.rows, thresh, threshHeir, nullptr, 1, &nboxes, 1);

                if (nms)
                {
                    do_nms_sort(detections, nboxes, noOfClass, nms);
                }

                // Process the detections and store them in 'objectInfoList'
                for (int i = 0; i < nboxes; i++)
                {
                    for (int j = 0; j < noOfClass; ++j)
                    {
                        if (detections[i].prob[j] > thresh)
                        {
                            objectInfoList[j].push_back(std::make_pair(cv::Rect(((detections[i].bbox.x - detections[i].bbox.w / 2) * matImage.cols), ((detections[i].bbox.y - detections[i].bbox.h / 2) * matImage.rows), (detections[i].bbox.w * matImage.cols), (detections[i].bbox.h * matImage.rows)), detections[i].prob[j]));
                            objectCount++;
                        }
                    }
                }
                free_detections(detections, nboxes);
            }
            errorDetails.errorcode = NoError;
            errorDetails.errormsg = "";
            return true;
        }
        else
        {
            errorDetails.errorcode = DetectionError;
            errorDetails.errormsg = "Failed to initialize the neural network for object detection.";
            return false; // Return false to indicate failure.
        }
    }
    catch (std::exception &e)
    {
        errorDetails.errorcode = DetectionError;
        errorDetails.errormsg = e.what();
        return false;
    }
    catch (...)
    {
        errorDetails.errorcode = DetectionError;
        errorDetails.errormsg = "Default Exception was catched";
        return false;
    }
}
bool Yolo::fileExists(std::string &file)
{
    fs::path filePath(file);
    return fs::exists(filePath) && fs::is_regular_file(filePath);
}
