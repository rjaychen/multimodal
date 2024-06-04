# Modified from
# https://docs.python.org/3.8/library/socketserver.html#asynchronous-mixins
# https://stackoverflow.com/questions/46138771/python-multipleclient-server-with-queues

import socket
import threading
import socketserver
import queue
import json
import argparse
import time
from typing import Tuple
from SAMEdgeInference import SAMEdgeInference
from EdgeSAMEdgeInference import EdgeSAMEdgeInference
import os

# from MobileSAMEdgeInference import MobileSAMEdgeInference
import base64
from collections import defaultdict
import numpy as np
import struct
import io
from PIL import Image
import pickle
from utils import *
import warnings

warnings.filterwarnings("ignore")

MODEL_FILEPATHS = {
    "sam": "src/saved_models/sam-vit-base",
    "ar_gaze_sam": "src/saved_models/sam-vit-base",
    "mobile_sam": "src/saved_models/mobilesam/mobile_sam.pt",
    "edge_sam": "src/saved_models/edgesam/edge_sam_3x.pth",
}


class InferenceExecutionThread(threading.Thread):
    """
    Inference execution thread.
    """

    def __init__(
        self,
        model_filepath: str,
        device: str = "cuda",
        image_size: Tuple[int, int] = [1440, 1080],
        model_name="sam",
        trial_index=0,
    ):
        super(InferenceExecutionThread, self).__init__()
        # Python queue library is thread-safe.
        # https://docs.python.org/3.8/library/queue.html#module-Queue
        # We can put tasks into queue from multiple threads safely.
        self.model_filepath = model_filepath
        self.device = device
        self.image_size = image_size
        self.model_name = model_name
        self.trial_index = trial_index

        if model_name == "sam":
            self.inference_session = SAMEdgeInference(
                model_path=model_filepath,
                device=device,
                image_size=image_size,
                query_cloud=False,
                smart_image_switch=False,
            )
        elif model_name == "ar_gaze_sam":
            self.inference_session = SAMEdgeInference(
                model_path=model_filepath,
                device=device,
                image_size=image_size,
                query_cloud=True,
                smart_image_switch=True,
                trial_index=trial_index,
            )
        elif model_name == "mobile_sam":
            from MobileSAMEdgeInference import MobileSAMEdgeInference

            self.inference_session = MobileSAMEdgeInference(
                model_path=model_filepath, device=device, image_size=image_size
            )
        elif model_name == "edge_sam":
            self.inference_session = EdgeSAMEdgeInference(
                model_path=model_filepath, device=device, image_size=image_size
            )

        self.latency_history = np.array([])

    def convert_to_parsable_string(self, data: np.ndarray) -> str:
        return "|".join(
            [
                ",".join([str(x) for x in row[::-1]])
                for row in data
                if row[0] < self.image_size[1] and row[1] < self.image_size[0]
            ]
        )

    def run(self):
        """
        Run inference for the tasks in the queue.
        """
        while True:
            if not request_content_queue.empty():
                print(
                    "Current Thread: {}, Number of Active Threads: {}".format(
                        threading.current_thread().name, threading.active_count()
                    )
                )
                handler, data_dct = request_content_queue.get()

                start_time = time.time()

                print("running inference...")
                output = self.inference_session.run(
                    data_dct, debug=0, return_2d_mask=False
                )
                end_time = time.time()
                latency = end_time - start_time
                if latency <= 5:
                    self.latency_history = np.append(
                        self.latency_history, latency * 1000
                    )
                """
                
                output = np.array([[1, 2], [3, 4]])
                serialized_output = self.convert_to_parsable_string(test_output)
                print(serialized_output)
                """
                if output == "":
                    request_content_queue.task_done()
                    continue

                response = output.encode("utf-8")
                # add length of the message at the beginning
                response = struct.pack(">I", len(response)) + response
                # print('Sending answer "{}" ...'.format(output))
                handler.request.sendall(response)
                request_content_queue.task_done()
                print("Inference Done.")
                # print(self.latency_history)
                print(
                    f"#Queried: {len(self.latency_history[1:])}; Current Average Latency: {np.mean(self.latency_history[1:]):.4g} ms, std: {np.std(self.latency_history[1:]):4g} ms"
                )
                if len(self.latency_history) > 1:
                    with open(
                        f"logs/trial_{self.trial_index}/{self.model_name}_latency.txt",
                        "a",
                    ) as f:
                        # write the last latency record
                        f.write(f"{self.latency_history[-1]:.4g}\n")

                # wait 0.5s
                # time.sleep(2)
                # time.sleep(5)


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """

    TCP request handler.
    """

    def handle(self) -> None:
        """
        Handle method to override.
        """
        temp = []
        finished = True
        image_length = 0
        start = []
        while True:
            data = self.request.recv(500000)
            if not data:
                print("User disconnected.")
                break
            if start:
                data = start + data
                start = []
            if finished:
                image_length = int.from_bytes(data[:4], byteorder="big")
                point_w = int.from_bytes(data[4:8], byteorder="big")
                poiny_h = int.from_bytes(data[8:12], byteorder="big")
                image_data = data[12:]
            else:
                image_data = data
            # print(f"Received data length: {len(image_data)}")
            """
            # decode the data
            lst = []
            chunk_size = 1
            for i in range(0, len(image_data), chunk_size):
                lst.append(
                    int.from_bytes(image_data[i : i + chunk_size], byteorder="big")
                )
            if len(lst) + len(temp) == width * height:
                # print(
                #     f"adding up to total length/desired length: {len(temp) + len(lst)}/{width*height}"
                # )
                finished = True
                temp += lst
            elif len(lst) + len(temp) > width * height:
                # print("Bad data received. Discarding...")
                finished = True
                temp = []
                continue
            else:
                # print(
                #     f"adding up to total length/desired length: {len(temp) + len(lst)}/{width*height}"
                # )
                temp += lst
                finished = False
                continue

            
            """
            if len(image_data) + len(temp) >= image_length:
                start = image_data[image_length - len(temp) :]
                temp += image_data[: image_length - len(temp)]
                finished = True

            else:
                temp += image_data
                finished = False
                continue

            # convert temp back to bytes
            temp = bytes(temp)
            PIL_image = Image.open(io.BytesIO(temp))
            # print("Size: ", PIL_image.size, "Mode: ", PIL_image.mode)
            # change from RGBA to RGB
            if PIL_image.mode == "RGBA":
                PIL_image = PIL_image.convert("RGB")
            # print(np.array(PIL_image).shape)
            # PIL_image.show()

            data_dict = {
                "image": PIL_image,
                "point": [point_w, poiny_h],
            }
            request_content_queue.put((self, data_dict))
            temp = []
            # print("Task sent to queue.")


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """
    Mutlithread TCP server.
    """

    pass


def main() -> None:
    # host_default = "localhost"
    host_default = "192.168.1.31"
    # host_default = "0.0.0.0"
    port_default = 8091
    num_inference_sessions_default = 1
    model_name_default = "sam"

    parser = argparse.ArgumentParser(
        description="Digit Recognition server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, help="Default host IP.", default=host_default
    )
    parser.add_argument(
        "--port", type=int, help="Default port ID.", default=port_default
    )
    parser.add_argument(
        "--num_inference_sessions",
        type=int,
        help="Number of inference sessions.",
        default=num_inference_sessions_default,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Type of the segmentation model to use. Options: 'sam', 'mobile_sam', 'edge_sam'",
        default=model_name_default,
    )

    argv = parser.parse_args()

    host = argv.host
    port = argv.port
    num_inference_sessions = argv.num_inference_sessions
    model_name = argv.model_name

    # read index from "logs" folder, and increment by 1. Index indicated by directory "trial_{index}"
    index = 0
    while os.path.exists(f"logs/trial_{index}"):
        index += 1
    os.makedirs(f"logs/trial_{index}")

    model_filepath = MODEL_FILEPATHS[model_name]

    global request_content_queue
    # Do not use multiple queues.
    # It will slow down Python application significantly.
    # I have tested for each worker thread we have a queue.
    # The requests were put evenly into each of the queues.
    # But this slows down the latency significantly.
    request_content_queue = queue.LifoQueue()

    # Number of inference sessions.
    # Each inference session gets executed in an independent execution thread.
    global execution_threads

    print(f"Starting Digit Classification engine x {num_inference_sessions} ...")

    execution_threads = [
        InferenceExecutionThread(
            model_filepath=model_filepath,
            device="cuda",
            image_size=[640, 480],
            model_name=model_name,
            trial_index=index,
        )
        for _ in range(num_inference_sessions)
    ]
    for execution_thread in execution_threads:
        execution_thread.start()

    print("Starting Server ...")
    # Create the server, binding to localhost on port
    with ThreadedTCPServer((host, port), ThreadedTCPRequestHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        print("=" * 50)
        print("DC Server")
        print("=" * 50)
        server.serve_forever()

    for execution_thread in execution_threads:
        execution_thread.join()


if __name__ == "__main__":
    main()
