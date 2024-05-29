import cv2
# import numpy as np
import struct
import socket
import threading
import time

IP_ADDRESS = "127.0.0.1"
IP_PORT = 8083

class Client:
    def __init__(self, client_socket):
        self.client_socket = client_socket
        self.writer = client_socket.makefile('wb')

    def send_image_data(self, image_data):
        if not self.client_socket:
            return False
        try:
            self.writer.write(struct.pack('>I', len(image_data))) # send image payload size using Big-Endian data
            self.writer.write(image_data) # send actual payload
            self.writer.flush()
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            return False
        
    def send_image_with_bbox(self, image_data, bbox_values=(0,0,1,1), dims=(1,1)):
        """
        Sends (1) total length (2) bbox values [x,y,w,h] (3) dimensions [h,w] (4) image payload. 
        """
        if not self.client_socket:
            return False
        try:
            bbox_data = struct.pack('>6I', *bbox_values, *dims)
            total_length = len(image_data) + len(bbox_data)
            self.writer.write(struct.pack('>I', total_length)) # send total length
            self.writer.write(bbox_data) # send bbox data using Big-Endian
            self.writer.write(image_data) # send image payload
            self.writer.flush()
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            return False

class ImageServer:
    def __init__(self):
        self.server = None
        self.server_running = False
        self.listen_thread = None
        self.sending = False 
        self.sending_thread = None
        self.clients = []

    def listen_thread_func(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((IP_ADDRESS, IP_PORT))
        self.server.listen(5)
        self.server_running = True
        print("Server started, listening...")
        
        while self.server_running:
            try:
                client_socket, addr = self.server.accept()
                print(f"Accepted connection from: {addr}")
                with threading.Lock():
                    self.clients.append(Client(client_socket))

            except Exception as e:
                print(e)
                print("shutting down stream")

        with threading.Lock():
            for client in self.clients:
                try:
                    client.client_socket.close()
                except:
                    pass
            self.clients.clear()

    def send_thread_func(self):
        cam = cv2.VideoCapture(0)
        self.sending = True
        while(cam.isOpened() and self.sending):
            time.sleep(0.5)
            suc, frame = cam.read()
                        
            encoded_image = cv2.imencode('.jpg', frame)[1].tobytes()

            with threading.Lock():
                clients = list(self.clients)

            for client in clients: 
                success = False 
                try:
                    success = client.send_image_data(encoded_image)
                except:
                    success = False
                    client.client_socket.close()
                finally:
                    if not success:
                        with threading.Lock():
                            self.clients.remove(client)

    def run(self):
        self.listen_thread = threading.Thread(target=self.listen_thread_func)
        self.listen_thread.start()

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
        self.sending_thread.join()
        print("\nPress <Enter> to quit")
        input()

if __name__ == "__main__":
    server = ImageServer()
    server.run()
