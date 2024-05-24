import cv2
from FastSAM.fastsam import FastSAM, FastSAMPrompt
import torch
import time
import os
from Detector import *

# Create the FastSAM model
model = FastSAM('FastSAM-x.pt')  # or FastSAM-x.pt

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DATA_FOLDER = "dnn_model_data"

# Object Detection Model
configPath = os.path.join(DATA_FOLDER, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelPath = os.path.join(DATA_FOLDER, "frozen_inference_graph.pb")
classesPath = os.path.join(DATA_FOLDER, "coco.names")

detector = Detector(configPath, modelPath, classesPath)


cap = cv2.VideoCapture(0)
if (cap.isOpened == False):
    print("Error opening file...")
                    
suc, frame = cap.read()
while suc:
    start = time.perf_counter()
    results = model(
                source=frame,
                device=DEVICE,
                retina_masks=True,
                imgsz=(448,256), # should match frame size of camera, must be multiple of 32.
                conf=0.4, # Confidence threshold
                iou=0.9, # Intersection over union threshold (Filter out duplicate detections)
            )

    (x, y, w, h), _ , _ = detector.getBbox(frame)

    # annotated_frame = show_seg(results[0].masks.data)
    prompt_process = FastSAMPrompt(frame, results, device=DEVICE)

    # Box Prompt --> Inpaint
    mask = np.squeeze(prompt_process.box_prompt(bbox=[x,y,x+w,y+h])).astype(np.uint8)
    inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    cv2.imshow("inpainted", inpainted)
    
    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time
            
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('img', frame)
    key = cv2.waitKey(1) & 0xFF
                
    if key == ord('q'): break
    suc, frame = cap.read()
cv2.destroyAllWindows()