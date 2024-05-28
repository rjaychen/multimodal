import cv2
import numpy as np
import socket
import threading
import time

IP_ADDRESS = "127.0.0.1"
IP_PORT = 8083

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
                client, addr = self.server.accept()
                print(f"Accepted connection from: {addr}")
                if client:
                    cam = cv2.VideoCapture(0)
                    while(cam.isOpened()):
                        ret, camImage = cam.read()
                        
                        byteString = bytes(cv2.imencode('.jpg', camImage)[1].tobytes())
                        fileSize = len(byteString)
                        totalSent = 0
                        client.sendall(str(fileSize).encode())

                        sizeConfirmation = client.recv(1024)

                        time.sleep(0.5)

                        # totalSent = 0
                        # while totalSent < fileSize:
                        #     totalSent += client.send(byteString[totalSent:])

                        # print(str(fileSize), str(totalSent),sizeConfirmation.decode('utf-8'))

            except Exception as e:
                print(e)
                print("shutting down video stream")
                _continue = False

        print("video stream exited.")

imageStreamer4()