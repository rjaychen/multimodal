import json
import cv2 as cv 
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
from PIL import Image
from io import BytesIO
import time
import gc
import concurrent.futures
import threading
import moviepy.editor as mp

cap = -1
session = requests.Session()
MRC_wAudio_Url = "rtsp://i3t:IotLab443@192.168.1.8/api/holographic/stream/live_med.mp4?holo=true&pv=true&mic=true&loopback=false"
MRC_woAudio_Url = "https://192.168.226.146/api/holographic/stream/live_med.mp4?holo=true&pv=true&mic=false&loopback=false"
Raw_wAudio_Url = "https://192.168.1.51/api/holographic/stream/live_med.mp4?holo=false&pv=true&mic=true&loopback=false"
Raw_woAudio_Url = "https://192.168.1.51/api/holographic/stream/live_med.mp4?holo=false&pv=true&mic=false&loopback=false"

def imageViewingThread():
    for chunk in res.iter_content(chunk_size=1024*30):
        print(chunk)
        img = np.asarray(bytearray(chunk), dtype="uint8")
        img = cv.imdecode(img, cv.IMREAD_COLOR)
        cv.imshow('image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    # cv.namedWindow("live transmission", cv.WINDOW_AUTOSIZE)
    # view_frame = cv.imread("VidFrame.jpg")
    # print("running imageViewingThread")

    # while True:
    #     view_frame = cv.imread("VidFrame.jpg")
    #     print ("reading the frame")

    #     if view_frame is not None:
    #         print("Viewing the frame")
    #         cv.imshow("live transmission", view_frame)

    #         key = cv.waitKey(5)
    #         if key == ord('q'):
    #            break

def audioReceivingThread():
    while True:
        video = mp.VideoFileClip(r"NewFile.mp4")
        video.audio.write_audiofile(r"AudioFile.mp3")
        print("Audio file is written")

def imageReceivingThread():
    # declare local variables
    frame = None
    ret = None
    lastUsedIndex = -1

    while 1:
        # open the received video file from HoloLens 2
        with open("NewFile.mp4", 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024*30):
                # write to a video
                if chunk:
                    print("Writing")
                    f.write(chunk)

                # read a video file
                vid = cv.VideoCapture("NewFile.mp4")
                lastFrameIndex = vid.get(cv.CAP_PROP_FRAME_COUNT)
                print('Last Frame: {}, Used Frame: {}'.format(lastFrameIndex, lastUsedIndex))

                # read an image frame from the video
                #if(lastFrameIndex > lastUsedIndex):
                vid.set(cv.CAP_PROP_POS_FRAMES, lastFrameIndex - 5)
                ret, frame = vid.read()

                # if reading a video is successful, write an image frame as a new file
                if ret:
                    print("Good")
                    #count = count + 1
                    cv.imwrite("VidFrame.jpg", frame)

                lastUsedIndex = lastFrameIndex

                # close the video after certain time
                #if(lastUsedIndex > 500):
                #    f.close()
                #    videoFileClosed = True

                num = gc.collect()

    # write an audio file from the video file
    if extractAudio:
        video = mp.VideoFileClip(r"NewFile.mp4")
        video.audio.write_audiofile(r"AudioFile.mp3")
        print("Audio file is written")

if __name__ == "__main__":
    global extractAudio

    # CHANGE THIS based on Application
    stream_url = Raw_wAudio_Url         # MRC_wAudio_Url, MRC_woAudio_Url, Raw_wAudio_Url, Raw_woAudio_Url

    if stream_url is Raw_wAudio_Url or MRC_wAudio_Url:
        extractAudio = True
    else:
        extractAudio = False

    # cap = cv.VideoCapture(0)
    # ret, frame = cap.read()
    # print(frame.shape)
    # while(True):
    #     cv.imshow("frame", frame)
    #     if cv.waitKey(1) & 0xFF == ord('q'): break
    
    # start the code for receiving video streams from HoloLens 2
    # res = session.get(url=MRC_wAudio_Url, verify=False, auth=HTTPBasicAuth('i3t', 'Iotlab443'), stream=True)
    cap = cv.VideoCapture()
    cap.open(MRC_wAudio_Url)
    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    # if(res.status_code == 200):
    #     bytes = bytes()
    #     for chunk in res.iter_content(chunk_size=1024*30):
    #         i = np.asarray(bytearray(chunk), dtype="uint8")
    #         print(i, len(i))
    #         i = cv.imdecode(i, cv.IMREAD_COLOR)
    #         cv.imshow('image', i)
    #         cv.waitKey(0)
    #         cv.destroyAllWindows()


    # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Run server
        #executor.submit(serverThread)

        # Run image receiving and viewing threads
        # executor.submit(imageViewingThread)
        # executor.submit(imageReceivingThread)
        # executor.submit(audioReceivingThread)