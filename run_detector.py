from Detector import *
import os

DATA_FOLDER = "dnn_model_data"

def run_detector():
    configPath = os.path.join(DATA_FOLDER, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join(DATA_FOLDER, "frozen_inference_graph.pb")
    # configPath = os.path.join(DATA_FOLDER, "yolov4-tiny.cfg")
    # modelPath = os.path.join(DATA_FOLDER, "yolov4-tiny.weights")
    classesPath = os.path.join(DATA_FOLDER, "coco.names")

    detector = Detector(configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__': 
    run_detector()