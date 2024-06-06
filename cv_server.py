import os
import time
import queue
from pynput import keyboard
from server import *
from Detector import *
from FastSAM.fastsam import FastSAM, FastSAMPrompt

DEVICE = "cpu" # torch.device(
    # "cuda"
    # if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    # else "cpu"
# )
DATA_FOLDER = "dnn_model_data"
IP_ADDRESS = "192.168.1.27"
# IP_ADDRESS = "127.0.0.1"
IP_PORT = 8083

class ODImageServer(ImageServer):
    def __init__(self, host, port, configPath, modelPath, classesPath):
        super().__init__(host, port)
        self.detector = Detector(configPath, modelPath, classesPath)
        self.sam_model = FastSAM('FastSAM-x.pt') # TODO: change this to formal input
        self.recv_queue = queue.LifoQueue()

    def recv_thread_func(self):
        """
        Receive image from TCP and write it into a data queue.
        """
        while 1:
            # print('recv running')
            with threading.Lock():
                clients = list(self.clients) 
            for client in clients: 
                try:
                    data = client.recv_image_data()
                    if len(data) < 0: print('empty data recv')
                    # print(f'image_data {data}')
                    self.recv_queue.put(data)
                except Exception as e:
                    print(e)

    def send_thread_func(self):
        """
        Gets image data from queue, then sends (first) inpainted ROI then (second) bbox values to client.
        """
        # cam = cv2.VideoCapture(0)
        self.sending = True
        last_frame = None
        while(self.sending):
            if (self.recv_queue.not_empty):
                raw_frame = self.recv_queue.get()
                
                if len(raw_frame) > 0: 
                    frame = cv2.imdecode(np.asarray(bytearray(raw_frame), dtype=np.uint8), cv2.IMREAD_COLOR)
                    last_frame = frame
                elif last_frame is not None: frame = last_frame 
                else: frame = np.zeros((242, 420, 3), np.uint8)
                try: 
                    # Image Processing Step ----------------------------------------------------------------
                    # Get Bbox data
                    results = self.sam_model(
                        source=frame,
                        device="cpu", # TODO: adapt code for GPU usage
                        retina_masks=True,
                        imgsz=(448,256), # should match frame size of camera, must be multiple of 32.
                        conf=0.4, # Confidence threshold
                        iou=0.9, # Intersection over union threshold (Filter out duplicate detections)
                    )
                    (x, y, w, h) , _ , _ = self.detector.getBbox(frame)
                    height, width, _ = frame.shape
                    # Get object mask data
                    prompt_process = FastSAMPrompt(frame, results, device=DEVICE)
                    mask = np.squeeze(prompt_process.box_prompt(bbox=[x,y,x+w,y+h])).astype(np.uint8)
                    # Get Inpainted image data
                    inpainted = self.detector.drawBbox(frame, useBlurInstead=False, onlyBbox=True , input_mask=mask)
                    encoded_image = cv2.imencode('.jpg', inpainted)[1].tobytes()

                    with threading.Lock():
                        clients = list(self.clients)

                    for client in clients: 
                        success = False 
                        try:
                            success = client.send_image_with_bbox(encoded_image, bbox_values = (x,y,w,h), dims=(height, width))
                            # print(f'sending {len(encoded_image)}')
                        except:
                            success = False
                            client.client_socket.close()
                        finally:
                            if not success:
                                with threading.Lock():
                                    self.clients.remove(client)
                except: 
                    pass
        
    def run(self):
        self.listen_thread = threading.Thread(target=self.listen_thread_func)
        self.listen_thread.start()

        self.recv_thread = threading.Thread(target=self.recv_thread_func)
        self.recv_thread.start()

        self.sending_thread = threading.Thread(target=self.send_thread_func)
        self.sending_thread.start()

        while True: 
            line = input().strip().lower()
            if line == "stop":
                break
            elif line == "status":
                with threading.Lock():
                    print(f"# Clients: {len(self.clients)}")
            else:
                print(line)
        
        self.sending = False
        self.server_running = False
        self.server.close()
        self.listen_thread.join()
        self.recv_thread.join()
        self.sending_thread.join()
        print("\nPress <Enter> to quit")
        input()

if __name__ == "__main__":
    configPath = os.path.join(DATA_FOLDER, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join(DATA_FOLDER, "frozen_inference_graph.pb")
    classesPath = os.path.join(DATA_FOLDER, "coco.names")
    server = ODImageServer(IP_ADDRESS, IP_PORT, configPath, modelPath, classesPath)
    server.run()