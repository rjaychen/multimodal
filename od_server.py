from server import *
from Detector import *
import os

DATA_FOLDER = "dnn_model_data"

class ODImageServer(ImageServer):
    def __init__(self, configPath, modelPath, classesPath):
        super().__init__()
        self.detector = Detector(configPath, modelPath, classesPath)

    def send_thread_func(self):
        """
        Sends (first) inpainted ROI then (second) bbox values to client.
        """
        cam = cv2.VideoCapture(0)
        self.sending = True
        while(cam.isOpened() and self.sending):
            time.sleep(0.5)
            suc, frame = cam.read()

            classLabelIDs, confidences, bboxs = self.detector.net.detect(frame, confThreshold=0.6)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))   
            
            # Get Bbox data
            (x, y, w, h) , _ , _ = self.detector.getBbox(frame)
            height, width, _ = frame.shape

            # Get Inpainted image data
            inpainted = self.detector.drawBbox(frame, onlyBbox=True)         
            encoded_image = cv2.imencode('.jpg', inpainted)[1].tobytes()
            
            with threading.Lock():
                clients = list(self.clients)

            for client in clients: 
                success = False 
                try:
                    success = client.send_image_with_bbox(encoded_image, bbox_values = (x,y,w,h), dims=(height, width))
                except:
                    success = False
                    client.client_socket.close()
                finally:
                    if not success:
                        with threading.Lock():
                            self.clients.remove(client)
        
if __name__ == "__main__":
    configPath = os.path.join(DATA_FOLDER, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join(DATA_FOLDER, "frozen_inference_graph.pb")
    classesPath = os.path.join(DATA_FOLDER, "coco.names")
    server = ODImageServer(configPath, modelPath, classesPath)
    server.run()