from pynput import keyboard
# import hl2ss
# import hl2ss_lnm
import os

from server import *
from Detector import *

DATA_FOLDER = "dnn_model_data"
IP_ADDRESS = "127.0.0.1"
IP_PORT = 8083

class ODImageServer(ImageServer):
    def __init__(self, host, port, configPath, modelPath, classesPath):
        super().__init__(host, port)
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

            # classLabelIDs, confidences, bboxs = self.detector.net.detect(frame, confThreshold=0.6)
            # bboxs = list(bboxs)
            # confidences = list(np.array(confidences).reshape(1, -1)[0])
            # confidences = list(map(float, confidences))   
            
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
                    print(f'sending {len(encoded_image)}')
                except:
                    success = False
                    client.client_socket.close()
                finally:
                    if not success:
                        with threading.Lock():
                            self.clients.remove(client)
    # def od_thread_func(self):
    #     """
    #     Sends (first) inpainted ROI then (second) bbox values to client.
    #     """
    #     # Settings ------------------------------------
    #     mode = hl2ss.StreamMode.MODE_1
    #     width     = 424
    #     height    = 240
    #     framerate = 30
    #     enable_mrc = False
    #     divisor = 1 
    #     profile = hl2ss.VideoProfile.H265_MAIN
    #     decoded_format = 'bgr24'
    #     # ----------------------------------------------
    #     hl2ss.start_subsystem_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc)
    #     def on_press(key):
    #             global enable
    #             enable = key != keyboard.Key.esc
    #             return enable

    #     listener = keyboard.Listener(on_press=on_press)
    #     listener.start()

    #     client = hl2ss_lnm.rx_pv(self.host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, divisor=divisor, profile=profile, decoded_format=decoded_format)
    #     client.open()
        
    #     while(self.server_running):
            
    #         time.sleep(0.5)
    #         data = client.get_next_packet()
    #         frame = data.payload.image
            
    #         # Get Bbox data
    #         (x, y, w, h) , _ , _ = self.detector.getBbox(frame)
    #         height, width, _ = frame.shape

    #         # Get Inpainted image data
    #         inpainted = self.detector.drawBbox(frame, onlyBbox=True)         
    #         encoded_image = cv2.imencode('.jpg', inpainted)[1].tobytes()
            
    #         with threading.Lock():
    #             clients = list(self.clients)

    #         for client in clients: 
    #             success = False 
    #             try:
    #                 success = client.send_image_with_bbox(encoded_image, bbox_values = (x,y,w,h), dims=(height, width))
    #             except:
    #                 success = False
    #                 client.client_socket.close()
    #             finally:
    #                 if not success:
    #                     with threading.Lock():
    #                         self.clients.remove(client)
        
if __name__ == "__main__":
    configPath = os.path.join(DATA_FOLDER, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join(DATA_FOLDER, "frozen_inference_graph.pb")
    classesPath = os.path.join(DATA_FOLDER, "coco.names")
    server = ODImageServer(IP_ADDRESS, IP_PORT, configPath, modelPath, classesPath)
    server.run()