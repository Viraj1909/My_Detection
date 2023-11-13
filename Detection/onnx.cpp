#include "onnx.H"

Onnx::Onnx() {}

Onnx::~Onnx() {}

bool Onnx::configuration(DetectionConfigurationParameter parameters, PartitionDetectionConfigurationParameter partitionParameter)
{
    try {
        if(!fileExists(parameters.weightFile))
        {
            errorDetails.errorcode = FileNotFound;
            errorDetails.errormsg = ".weight file not found";
            return false;
        }
        net_ = cv::dnn::readNet(parameters.weightFile); //model file
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        std::ifstream nameFile(parameters.nameFile); //names file
        std::string name;

        while(std::getline(nameFile, name))
        {
            names.push_back(name);
        }

        confidenceThreshold_=thresh;
        threshHeir = parameters.thresh;
        nms = parameters.nms;

        errorDetails.errorcode = NoError;
        errorDetails.errormsg = "";

        return true;
    }
    catch (const std::exception& e) {
        errorDetails.errorcode = DetectionError;
        errorDetails.errormsg = e.what();
        return false;
    }
}

bool Onnx::detect(cv::Mat &image, std::map<int, std::vector<std::pair<cv::Rect, float>>> &objectInfoList, int &objectCount)
{
    try{
        cv::Mat blob;
        int col = image.cols;
        int row = image.rows;
        int maxLen = MAX(col, row);
        cv::Mat netInputImg = image.clone();
        if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
            cv::Mat resizeImg = cv::Mat::zeros(maxLen, maxLen, CV_8UC3);
            image.copyTo(resizeImg(cv::Rect(0, 0, col, row)));
            netInputImg = resizeImg;
        }
        cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(size_, size_), cv::Scalar(0, 0, 0), true, false);
        net_.setInput(blob);
        std::vector<cv::Mat> netOutputImg;
        net_.forward(netOutputImg, net_.getUnconnectedOutLayersNames());

        std::vector<int> classIds;//result id array
        std::vector<float> confidences;//As a result, each id corresponds to a confidence array
        std::vector<cv::Rect> boxes;//Each id rectangle
        float ratio_h = (float)netInputImg.rows / size_;
        float ratio_w = (float)netInputImg.cols / size_;
        int net_width = names.size() + 5;  //The output network width is the number of categories+5
        for (int stride = 0; stride < strideSize; stride++) {    //stride
            float* pdata = (float*)netOutputImg[stride].data;
            int grid_x = (int)(size_ / netStride[stride]);
            int grid_y = (int)(size_ / netStride[stride]);
            for (int anchor = 0; anchor < 3; anchor++) {	//anchors
                const float anchor_w = netAnchors[stride][anchor * 2];
                const float anchor_h = netAnchors[stride][anchor * 2 + 1];
                for (int i = 0; i < grid_y; i++) {
                    for (int j = 0; j < grid_x; j++) {
                        float box_score = sigmoid_x(pdata[4]); ;//Get the probability that an object is contained in the box of each row
                        if (box_score >= threshHeir) {
                            cv::Mat scores(1, names.size(), CV_32FC1, pdata + 5);
                            cv::Point classIdPoint;
                            double max_class_socre;
                            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
                            max_class_socre = sigmoid_x(max_class_socre);
                            if (max_class_socre >= confidenceThreshold_) {
                                float x = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
                                float y = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
                                float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
                                float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
                                int left = (int)(x - 0.5 * w) * ratio_w + 0.5;
                                int top = (int)(y - 0.5 * h) * ratio_h + 0.5;
                                classIds.push_back(classIdPoint.x);
                                confidences.push_back(max_class_socre * box_score);
                                boxes.push_back(cv::Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
                            }
                        }
                        pdata += net_width;//next line
                    }
                }
            }
        }

        //Perform non-maximum suppression to remove redundant overlapping boxes with lower confidence (NMS)
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, nmsScoreThreshold, nms, nms_result);
        for (size_t i = 0; i < nms_result.size(); i++) {
            int idx = nms_result[i];
            objectInfoList[classIds[idx]].push_back(std::make_pair(boxes[idx],confidences[idx]));
            objectCount++;
        }
        return true;
    }catch (std::exception &e) {
        errorDetails.errorcode = DetectionError;
        errorDetails.errormsg = e.what();
        return false;
    }
}
bool Onnx::fileExists(std::string& file)
{
    fs::path filePath(file);
    return fs::exists(filePath) && fs::is_regular_file(filePath);
}
