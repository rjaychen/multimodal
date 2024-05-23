import cv2
import numpy as np
import time

np.random.seed(20)
class Detector: 
    def __init__(self, configPath, modelPath, classesPath, videoPath=None):
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.videoPath = videoPath

        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
        # self.classesList.insert(0, '__Background__') # add back for certain models 

    def getBbox(self, frame):
        classLabelIDs, confidences, bboxs = self.net.detect(frame, confThreshold=0.6)

        bboxs = list(bboxs)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))
        try: 
            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)[0]
            bbox = bboxs[np.squeeze(bboxIdx)]
            classConfidence = confidences[np.squeeze(bboxIdx)]
            classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx)])
            classLabel = self.classesList[classLabelID]
            classColor = [int(c) for c in self.colorList[classLabelID]] # optional
                
            displayText = "{}:{:.2f}".format(classLabel, classConfidence)
        except: 
            bbox = 0, 0, 1, 1
        
        return bbox, classColor, displayText

    def drawBbox(self, frame, useBlurInstead=False):
        # classLabelIDs, confidences, bboxs = self.net.detect(frame, confThreshold=0.6)

        # bboxs = list(bboxs)
        # confidences = list(np.array(confidences).reshape(1, -1)[0])
        # confidences = list(map(float, confidences))
        # try: 
        #     bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)[0]
        #     bbox = bboxs[np.squeeze(bboxIdx)]
        #     classConfidence = confidences[np.squeeze(bboxIdx)]
        #     classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx)])
        #     classLabel = self.classesList[classLabelID]
        #     classColor = [int(c) for c in self.colorList[classLabelID]] # optional
                
        #     displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                
        #     x,y,w,h = bbox
        #     cv2.rectangle(frame, (x,y), (x+w, y+h), color=(255,0,0), thickness=1)
        #     cv2.putText(frame, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
        # except: 
        #     x,y,w,h = 0, 0, 1, 1
        #     pass

        cv2.rectangle(frame, (x,y), (x+w, y+h), color=(255,0,0), thickness=1)
        cv2.putText(frame, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)
        if useBlurInstead:
            inpainted = frame 
            cropped = frame[y:y+h, x:x+w]
            blurred = cv2.medianBlur(cropped, 15)# cv2.GaussianBlur(cropped, kernelSize, 0)
            inpainted[y:y+h,x:x+w] = blurred
        else: inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        return inpainted
    
    def onVideo(self, useSelf=True):
        if not useSelf and self.videoPath is not None: cap = cv2.VideoCapture(self.videoPath)
        else: cap = cv2.VideoCapture(0)
        if (cap.isOpened == False):
            print("Error opening file...")
            return
        
        suc, image = cap.read()

        while suc:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.6)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))
            
            # # Multiple Bounding Boxes
            # bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)
            # if len(bboxIdx) != 0:
            #     for i in range(len(bboxIdx)):
            #         bbox = bboxs[np.squeeze(bboxIdx[i])]
            #         classConfidence = confidences[np.squeeze(bboxIdx[i])]
            #         classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
            #         classLabel = self.classesList[classLabelID]
            #         classColor = [int(c) for c in self.colorList[classLabelID]] # optional

            #         displayText = "{}:{:.2f}".format(classLabel, classConfidence)

            #         # print(f'{classLabel}, {classConfidence:.2f}, {bbox}') # --> display class Labels

            #         x,y,w,h = bbox
            #         cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,0,0), thickness=1)
            #         cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
            
            # Single Bounding Box
            # try: 
            #     bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)[0]
            #     bbox = bboxs[np.squeeze(bboxIdx)]
            #     classConfidence = confidences[np.squeeze(bboxIdx)]
            #     classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx)])
            #     classLabel = self.classesList[classLabelID]
            #     classColor = [int(c) for c in self.colorList[classLabelID]] # optional
                
            #     displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                
            #     x,y,w,h = bbox
            #     cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,0,0), thickness=1)
            #     cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
            # except: 
            #     x,y,w,h = 0, 0, 1, 1
            #     pass

            # # # Using blurring
            # # cropped = image[y:y+h, x:x+w]
            # # kernelSize = (15, 15)
            # # blurred = cv2.medianBlur(cropped, 15)# cv2.GaussianBlur(cropped, kernelSize, 0)
            # # image[y:y+h,x:x+w] = blurred
            # mask = np.zeros(image.shape[:2], dtype="uint8")
            # cv2.rectangle(mask, (x,y), (x+w, y+h), 255, -1)
            # inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            inpainted = self.drawBbox(image)
            cv2.imshow("Inpainting", inpainted)
            cv2.imshow("Original", image)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): break
            suc, image = cap.read()
        cv2.destroyAllWindows()
    
    