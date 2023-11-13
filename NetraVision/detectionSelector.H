#ifndef DETECTIONSELECTOR_H
#define DETECTIONSELECTOR_H

#include "yolo.H"
#include "onnx.H"
#include "colorInRangeDetection.H"

class detectionSelector
{
public:
    detectionSelector();

    enum DetectionType {
        ObjectDetector,
        onnx,
        InRangeDetection,
        RegionGrow
    };

    static DetectionLibrary *generateDetection(DetectionType type);
};

#endif // DETECTIONSELECTOR_H
