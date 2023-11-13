#ifndef NETRAVISION_H
#define NETRAVISION_H

#include "detectionSelector.H"
#include "spscbuffer.h"

#include <iostream>
#include <thread>
#include <atomic>
#include <condition_variable>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv_modules.hpp>
#include <opencv4/opencv2/dnn.hpp>

/**
 * @class NetraVision
 * @brief A class for computer vision and object detection using Darknet and color-based methods.
 */
class NetraVision
{
public:
    /**
     * @struct DetectionResults
     * @brief Structure to store the results of object detection.
     */
    struct DetectionResult
    {
        std::map<int, std::vector<std::pair<cv::Rect, float>>> result; ///< Object detection results.
        int objectCount;                                                     ///< Number of objects detected by Darknet.
    };
    struct ColorResult
    {
        std::vector<cv::Rect> results;                                   ///< Bounding boxes of objects detected by color-based methods.
        int colorCount;                                                       ///< Number of objects detected by color-based methods.
    };
    struct imageServiceParameter
    {
        std::string saveImageFilePath;
        //ADD MASK & BLUR needed variables.
    };

    /**
     * @enum DetectionObject
     * @brief Enumeration for Darknet detection method selection.
     */
    enum DetectionObject
    {
        ObjectDetection, ///< Darknet detection method.
        Onnx     ///< Onnx detection method.
    };

    /**
     * @enum DetectionColor
     * @brief Enumeration for color-based detection method selection.
     */
    enum DetectionColor
    {
        ColorInRangeDetection, ///< Color-in-range-based detection method.
        RegionGrow             ///< Region growing-based detection method.
    };

    /**
     * @brief Constructor for the NetraVision class.
     */
    NetraVision();

    /**
     * @brief Destructor for the NetraVision class.
     */
    ~NetraVision();

    /**
     * @brief Configure the detection method.
     * @param parameters Configuration parameters for detection.
     * @param partitionParameter Partition configuration for detection.
     * @param numberOfClass Number of object classes.
     * @param error Error message (if any) during configuration.
     * @return true if configuration is successful, false otherwise.
     */
    bool detectionConfiguration(DetectionObject method, DetectionLibrary::DetectionConfigurationParameter parameters, DetectionLibrary::PartitionDetectionConfigurationParameter partitionParameter, std::string &error);

    /**
     * @brief Configure the color-based detection method.
     * @param parameters Configuration parameters for color-based detection.
     * @param partitionParameter Partition configuration for color-based detection.
     * @param height Height of the image.
     * @param width Width of the image.
     * @param error Error message (if any) during configuration.
     * @return true if configuration is successful, false otherwise.
     */
    bool colorConfiguration(DetectionColor method, DetectionLibrary::ColorConfigurationParameters parameters, DetectionLibrary::PartitionDetectionConfigurationParameter partitionParameter, int height, int width, std::string &error);

    /**
     * @brief Perform object detection on an input image using selected methods.
     * @param image Input image for object detection.
     * @param objectInfoList Detected objects with associated information.
     * @param objectCount Total number of detected objects.
     * @param colorDetectionResults Bounding boxes of objects detected by color-based methods.
     * @param colorDetectionObjectCount Number of objects detected by color-based methods.
     * @param runDarknet Flag to run Darknet detection.
     * @param runColor Flag to run color-based detection.
     * @param error Error message (if any) during detection.
     */
    void detectNetraVision(const cv::Mat &image, std::map<int, std::vector<std::pair<cv::Rect, float>>> &objectInfoList, int &objectCount, std::vector<cv::Rect> &colorDetectionResults, int &colorDetectionObjectCount, bool runDarknet, bool runColor, std::string &error);

    void imageServiceConfiguration(imageServiceParameter);

    void setSessionNumber(int);

private:
    DetectionLibrary *objectDetector; ///< Pointer to the detection library.
    DetectionLibrary *colorDetector;   ///< Pointer to the color-based detection library.

    DetectionLibrary::ErrorDetails errors; ///< Error details.

    std::unique_ptr<std::thread> detectorThread; ///< Thread for detection.
    std::unique_ptr<std::thread> colorThread;   ///< Thread for color-based detection.
    std::unique_ptr<std::thread> saveImgThread;
    std::unique_ptr<std::thread> blurImgThread;
    std::unique_ptr<std::thread> maskImgThread;

    std::mutex mutex; ///< Mutex for synchronization.

    std::condition_variable cv, detectorCV, colorCV, saveImageCV, blurImgCV, maskerImgCV; ///< Condition variable for synchronization.

    std::atomic<bool> isDRunning; ///< Atomic flag for detection thread status.
    std::atomic<bool> isCRunning; ///< Atomic flag for color-based detection thread status.
    std::atomic<bool> isSaveImgRunning;
    std::atomic<bool> isBlurImgRunning;
    std::atomic<bool> isMaskImgRunning;

    std::atomic<bool> detectorRunning; ///< Atomic flag for detection status.
    std::atomic<bool> colorRunning;   ///< Atomic flag for color-based detection status.

    std::unique_ptr<SPSCBuffer<cv::Mat>> imageDetectionBuffer;
    std::unique_ptr<SPSCBuffer<cv::Mat>> imageColorBuffer;
    std::unique_ptr<SPSCBuffer<cv::Mat>> saveImageBuffer;
    std::unique_ptr<SPSCBuffer<cv::Mat>> blurImageBuffer;
    std::unique_ptr<SPSCBuffer<cv::Mat>> maskImageBuffer;
    std::unique_ptr<SPSCBuffer<DetectionResult>> detectionResultBuffer;
    std::unique_ptr<SPSCBuffer<ColorResult>> colorResultBuffer;


    DetectionResult detectResults;
    ColorResult colorResults;
    imageServiceParameter parameters;

    std::string error; ///< Error message.

    std::atomic<int> sessionNumber;

    /**
     * @brief Perform object detection on an input image.
     * @param image Input image for detection.
     * @param objectInfoList Detected objects with associated information.
     * @param objectCount Total number of detected objects.
     * @return true if detection is successful, false otherwise.
     */
    bool objectDetection(cv::Mat &image, std::map<int, std::vector<std::pair<cv::Rect, float>>> &objectInfoList, int &objectCount);

    /**
     * @brief Perform color-based object detection on an input image.
     * @param image Input image for color-based detection.
     * @param noOfObject Number of objects detected.
     * @param boundingBox Bounding boxes of detected objects.
     * @return true if color-based detection is successful, false otherwise.
     */
    bool colorDetection(cv::Mat &image, int &noOfObject, std::vector<cv::Rect> &boundingBox);

    /**
     * @brief Main loop for object detection thread.
     */
    void objectDetectLoop();

    /**
     * @brief Main loop for color-based object detection thread.
     */
    void colorDetectLoop();

    /**
     * @brief Stop the detection threads.
     */
    void stopDetectionThreads();

    void saveImageService(cv::Mat img, int imgNumber, std::string path);
    void saveImageLoop();
    void blurImageService(cv::Mat img);
    void blurImageLoop();
    void maskImageService(cv::Mat img);
    void maskImageLoop();
};

#endif // NETRAVISION_H
