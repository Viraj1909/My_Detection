#include "detectionSelector.H"

detectionSelector::detectionSelector() {}
DetectionLibrary *detectionSelector::generateDetection(DetectionType type)
{
    switch (type)
    {
    case ObjectDetector:
        return new Yolo();
    case onnx:
        return new Onnx();
    case InRangeDetection:
        return new ColorInRangeDetection();
    case RegionGrow:
        return nullptr;
    default:
        return nullptr;
    }
}
