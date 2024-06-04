from pynput import keyboard
# import hl2ss
# import hl2ss_lnm
import os

from server import *
from Detector import *
import queue

DATA_FOLDER = "dnn_model_data"
IP_ADDRESS = "192.168.1.27"
# IP_ADDRESS = "127.0.0.1"
IP_PORT = 8083

class ODImageServer(ImageServer):
    def __init__(self, host, port, configPath, modelPath, classesPath):
        super().__init__(host, port)
        self.detector = Detector(configPath, modelPath, classesPath)
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
            # print('sending')
            # time.sleep(0.5)
            if (self.recv_queue.not_empty):
                raw_frame = self.recv_queue.get()
                
                if len(raw_frame) > 0: 
                    frame = cv2.imdecode(np.asarray(bytearray(raw_frame), dtype=np.uint8), cv2.IMREAD_COLOR)
                    last_frame = frame
                elif last_frame: frame = last_frame 
                else: frame = np.zeros((242, 420, 3), np.uint8)
                # print(frame)
                try: 
                    # Get Bbox data
                    # print(frame.shape)
                    (x, y, w, h) , _ , _ = self.detector.getBbox(frame)
                    height, width, _ = frame.shape

                    # Get Inpainted image data
                    inpainted = self.detector.drawBbox(frame, useBlurInstead=True, onlyBbox=True)         
                    encoded_image = cv2.imencode('.jpg', inpainted)[1].tobytes()
                    
                    # # if you want to show full image 
                    # frame[x:x+w, y:y+h] = inpainted
                    # cv2.imshow("inpainted", frame)
                    # cv2.waitKey(0)

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