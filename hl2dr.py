#------------------------------------------------------------------------------
# This script uses depth frames received from the HoloLens to place textured
# quads in the Unity scene aligned with the surface the user is observing.
# A patch of 3D points at the center of the depth frame is used to fit a plane,
# which is then used to set the world transform of the quad such that the quad
# is aligned with observed surface (e.g., a wall).
# Press space to set a quad.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
import hl2ss_lnm
import hl2ss_3dcv
import hl2ss_rus
import hl2ss_mp
import hl2ss_3dcv
import multiprocessing as mp

# Additional Imports ----------------------------------------------------------
import sys
import os
sys.path.append(os.getcwd())
from Detector import * # Assume this is in cwd
import numpy as np
import time

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.168.1.31"

# Calibration folder (must exist but can be empty)
calibration_path = os.getcwd() + '/hl2ss/calibration'

# Quad scale in meters
scale = [0.2, 0.2, 1]
# Texture file (must be jpg or png)
texture_file = os.getcwd() + '/hl2ss/viewer/texture.jpg'

# Scaling factor for visibility
brightness = 8

# Detection Model -------------------------------------------------------------
DATA_FOLDER = "dnn_model_data"

configPath = os.path.join(DATA_FOLDER, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelPath = os.path.join(DATA_FOLDER, "frozen_inference_graph.pb")
classesPath = os.path.join(DATA_FOLDER, "coco.names")
detector = Detector(configPath, modelPath, classesPath)

# PV Settings -----------------------------------------------------------------

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Enable Mixed Reality Capture (Holograms)
enable_mrc = False

# Camera parameters
pv_width     = 424
pv_height    = 240
pv_framerate = 30

# Buffer length in seconds
buffer_length = 10

# Maximum depth in meters
max_depth = 3.0

# Framerate denominator (must be > 0)
# Effective FPS is framerate / divisor
divisor = 1 

# Decoded format
# Options include:
# 'bgr24'
# 'rgb24'
# 'bgra'
# 'rgba'
# 'gray8'
# decoded_format = 'bgr24'

#------------------------------------------------------------------------------

if __name__ == '__main__':
    enable = True
    trigger = False
    # Keyboard events ---------------------------------------------------------
    def clamp(v, min, max):
        return min if (v < min) else max if (v > max) else v

    def on_press(key):
        global enable
        global trigger
        if (key == keyboard.Key.esc):
            enable = False
        elif (key == keyboard.Key.space):
            trigger = True
        enable = key != keyboard.Key.esc
        return enable

    def on_release(key):
        global trigger
        if (key == keyboard.Key.space):
            trigger = False
        return True

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # with open(texture_file, mode='rb') as file:
    #     image = file.read()

    ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
    ipc.open()

    key = 0

    command_buffer = hl2ss_rus.command_buffer() # TODO: Figure out how TCP works with IPC
    command_buffer.remove_all()
    ipc.push(command_buffer)
    results = ipc.pull(command_buffer)

    previous  = False

    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)
    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    xy1, lt_scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

    u0 = hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH  // 2
    v0 = hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT // 2

    # uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    # xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)

    # Start PV and RM Depth Long Throw streams --------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate, decoded_format='rgb24'))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.configure(hl2ss.IPCPort.UNITY_MESSAGE_QUEUE, hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)) # TODO
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
    producer.initialize(hl2ss.IPCPort.UNITY_MESSAGE_QUEUE, pv_framerate * buffer_length) # TODO
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
    producer.start(hl2ss.IPCPort.UNITY_MESSAGE_QUEUE) # TODO
    
    # --------------------------------------------
    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)
    sink_ipc = consumer.create_sink(producer, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE, manager, None) # TODO: test

    sink_pv.get_attach_response()
    sink_depth.get_attach_response()
    sink_ipc.get_attach_response() # TODO: test

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)
    
    # Main Loop ---------------------------------------------------------------
    while (enable):
        # Wait for RM Depth Long Throw frame ----------------------------------
        sink_depth.acquire()

        # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
        _, data_lt = sink_depth.get_most_recent_frame()
        if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
            continue

        _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue
        
        _, 

        frame = data_pv.payload.image

        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length, data_pv.payload.principal_point)
        color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        # Image Analysis ------------------------------------------------------
        start = time.perf_counter()
        # Get Bbox data
        (x, y, w, h) , _ , _ = detector.getBbox(frame)
        height, width, _ = frame.shape

        # Get Inpainted image data
        inpainted = detector.drawBbox(frame, onlyBbox=True)         
        image = cv2.imencode('.jpg', inpainted)[1].tobytes()

        # Show depth image
        cv2.imshow('depth', data_lt.depth / np.max(data_lt.depth)) # Scaled for visibility
        cv2.waitKey(1)

        keydown = (not previous) and trigger
        previous = trigger

        if ((not hl2ss.is_valid_pose(data_lt.pose)) or (not keydown)):
            continue

        # Get the 3D points corresponding to the 7x7 patch in the center of the depth image
        depth = hl2ss_3dcv.rm_depth_normalize(data_lt.depth[(v0-3):(v0+4), (u0-3):(u0+4)], lt_scale[(v0-3):(v0+4), (u0-3):(u0+4)])
        xyz = hl2ss_3dcv.rm_depth_to_points(depth, xy1[(v0-3):(v0+4), (u0-3):(u0+4), :])
        xyz = hl2ss_3dcv.block_to_list(xyz)
        d = hl2ss_3dcv.block_to_list(depth).reshape((-1,))
        xyz = xyz[d > 0, :]

        # Need at least 3 points to fit a plane
        if (xyz.shape[0] < 3):
            print('Not enough points')
            continue

        # 4x4 matrix that converts 3D points in depth camera space to world space
        camera2world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
        points = hl2ss_3dcv.to_homogeneous(xyz) @ camera2world

        # Fit plane
        _, _, vh = np.linalg.svd(points)
        plane = vh[3, :]
        plane = plane / np.linalg.norm(plane[0:3])
        normal = plane[0:3]

        # Compute centroid
        centroid = np.median(points, axis=0)
        centroid = centroid[0:3]
            
        # Select the normal that points to the user
        camera_center = np.array([0, 0, 0, 1]).reshape((1, 4)) @ camera2world
        camera_center = camera_center[0, 0:4]
        direction = camera_center[0:3] - centroid
        if (np.dot(normal, direction) < np.dot(-normal, direction)):
            normal = -normal

        # Convert to left handed coordinates (Unity)
        normal[2] = -normal[2]
        centroid[2] = -centroid[2]

        # Find the axis and the angle of the rotation between the canonical normal and the plane normal
        canonical_normal = np.array([0, 0, -1]).reshape((1, 3)) # Normal that looks at the camera when the camera transform is the identity
        axis = np.cross(canonical_normal, normal)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(clamp(np.dot(canonical_normal, normal), -1, 1))

        # Convert axis-angle rotation to quaternion
        cos = np.cos(angle / 2)
        sin = np.sin(angle / 2)

        rotation = [axis[0,0] * sin, axis[0,1] * sin, axis[0,2] * sin, cos]

        # Add quad to Unity scene
        display_list = hl2ss_rus.command_buffer()
        display_list.begin_display_list() # Begin sequence
        display_list.create_primitive(hl2ss_rus.PrimitiveType.Quad) # Create quad, returns id which can be used to modify its properties
        display_list.set_target_mode(1) # Set server to use the last created object as target (this avoids waiting for the id)
        display_list.set_world_transform(key, centroid, rotation, scale) # Set the quad's world transform
        display_list.set_texture(key, image) # Set the quad's texture
        display_list.set_active(key, 1) # Make the quad visible
        display_list.set_target_mode(0) # Restore target mode
        display_list.end_display_list() # End sequence

        ipc.push(display_list) # Send commands to server
        results = ipc.pull(display_list) # Get results from server
        
        key = results[1]

        print(f'Created quad with id {key}')

    command_buffer = hl2ss_rus.command_buffer()
    command_buffer.remove_all()

    ipc.push(command_buffer)
    results = ipc.pull(command_buffer)

    ipc.close()

    # Stop PV and RM Depth Long Throw streams ---------------------------------
    sink_pv.detach()
    sink_depth.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Stop keyboard events ----------------------------------------------------
    listener.join()