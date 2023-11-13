#include "colorInRangeDetection.H"

ColorInRangeDetection::ColorInRangeDetection() {}
ColorInRangeDetection::~ColorInRangeDetection() {}
bool ColorInRangeDetection::configuration(ColorConfigurationParameters parameters, PartitionDetectionConfigurationParameter partitionParameter, int height, int width)
{
    try{
        parameter = parameters;
        heightPix = height;
        widthPix = width;
        return true;
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
        errorDetails.errormsg = "Default Exception Catched";
        return false;
    }

}

bool ColorInRangeDetection::detect(cv::Mat &image, int &noOfObject, std::vector<cv::Rect> &boundingBox)
{
    try {
        ColorDetectionDetails ret;
        float image_hight = image.size().height;
        float image_width = image.size().width;
        unsigned int totalArea = 0;
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat HSV_image;
        cv::Rect bounding_rect;

        ret.rects.clear();
        ret.areaFactor = 0.0;
        ret.maskedImage = cv::Mat::zeros(image_hight, image_width, CV_8UC3);

        cvtColor(image, HSV_image, cv::COLOR_RGB2HSV_FULL);

        cv::Mat mask_combined(image_hight, image_width, CV_8UC1, cv::Scalar(0));
        for (size_t i = 0; i < parameter.colorRanges.size(); i++) {
            cv::Mat mask;
            cv::inRange(HSV_image, cv::Scalar(parameter.colorRanges[i].lowChannel1, parameter.colorRanges[i].lowChannel2, parameter.colorRanges[i].lowChannel3),
                        cv::Scalar(parameter.colorRanges[i].highChannel1, parameter.colorRanges[i].highChannel2, parameter.colorRanges[i].highChannel3), mask);
            mask_combined=mask_combined|mask;
        }
        cv::morphologyEx(mask_combined, mask_combined, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));

        findContours(mask_combined, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        for (size_t i = 0; i < contours.size(); i++)
        {
            cv::Scalar contour_color, font_color;
            contour_color = (i % 2 == 0) ? cv::Scalar(0, 0, 225) : cv::Scalar(0, 255, 0);
            font_color = (i % 2 == 0) ? cv::Scalar(255, 255, 255) : cv::Scalar(255, 255, 225);
            drawContours(ret.maskedImage, contours, i, contour_color, cv::FILLED, 8);

            double area = contourArea(contours[i], false);

            if (parameter.maxContourSize >= area && parameter.minContourSize <= area)
            {
                int ar = static_cast<int>(area);
                cv::putText(ret.maskedImage, std::to_string(ar), contours[i][contours[i].size() / 2], cv::FONT_HERSHEY_SIMPLEX, 1.2, font_color, 2);
                bounding_rect = boundingRect(contours[i]);
                if (bounding_rect.width >= widthPix && bounding_rect.height > heightPix)
                {
                    totalArea += area;
                    boundingBox.push_back(bounding_rect);
                    noOfObject++;
                }
            }
        }
        ret.areaFactor = (float(totalArea) / image.size().area());
        errorDetails.errorcode = NoError;
        errorDetails.errormsg = "";
        return true;
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
        errorDetails.errormsg = "Default Exception Catched";
        return false;
    }
}
