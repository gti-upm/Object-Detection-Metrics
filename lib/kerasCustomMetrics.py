import keras.backend as K
import EvaluatorPoint, BoundingBox, BoundingBoxes

# TODO: still unfinished
def getBoundingBoxesYolo(y_true, y_pred):
    """
    y_true
    """
    """
    :param y_pred: 
    :type y_pred: 
    :return: 
    :rtype: 
    """
    """
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    folderGT = os.path.join(currentPath, 'groundtruths')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    """
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Process GT detections
    for:
        bb = BoundingBox(
            nameOfImage,
            idClass,
            x,
            y,
            w,
            h,
            CoordinatesType.Absolute, (200, 200),
            BBType.GroundTruth,
            format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)

        bb = BoundingBox(
            nameOfImage,
            idClass,
            x,
            y,
            w,
            h,
            CoordinatesType.Absolute, (200, 200),
            BBType.Detected,
            confidence,
            format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes

def error_dist(y_true, y_pred):

    metrics = EvaluatorPoint.GetPascalVOCMetrics(boundingboxes, distThreshold=30)
    return K.mean(metrics.dict['AP'])
