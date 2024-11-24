#!/usr/bin/env python3
import time

import cv2
import argparse
import asyncio
import numpy as np
from face_landmarks_detection.utils.utils import init_cv_cap
from socketUtil import WebSocketClient
from drawUtil import draw_bounding_box
from inference import HailoInference
from processingUtil import preprocess_faces, postprocess_faces

WS_URL = "ws://192.168.0.233:5000"
RECONNECT_INTERVAL = 3

def init_cam_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def face_detection(hailoObject, frame, input_width, input_height):
    output_names = {
        'bbox_pred': 'scrfd_10g/conv54',
        'cls_pred': 'scrfd_10g/conv53'
    }
    processed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_faces(
        frame, input_size=(input_width, input_height)
    )
    processed_frame = processed_frame.astype(np.float32)
    processed_frame = np.expand_dims(processed_frame, axis=0)

    raw_detections = hailoObject.run(processed_frame)

    outputs = {
        'scrfd_10g/conv54': raw_detections[output_names['bbox_pred']],
        'scrfd_10g/conv53': raw_detections[output_names['cls_pred']]
    }

    return postprocess_faces(outputs, img_w, img_h, scale, pad_w, pad_h)

def parse_args():
    parser = argparse.ArgumentParser(description='Face detection using scrfd_10g.hef with OpenCV camera feed')
    parser.add_argument("-fd", "--face_det",
                        default='model/scrfd_10g.hef',
                        type=str,
                        help="Path for the .hef model for face detection.",
                        required=False
                        )
    parser.add_argument("-fl", "--face_landmark",
                        default="model/face-landmarks-detection.hef",
                        type=str,
                        help="Path for the .hef model for face landmark detection.",
                        required=False
                        )
    args = parser.parse_args()
    return args

def init_hailo_obj(model, in_type, out_type):
    # Load the HEF model for face detection
    hailo_face_detection_inference = HailoInference(model, input_type=in_type, output_type=out_type)
    # Get input shape for face detection
    input_height, input_width, _ = hailo_face_detection_inference.get_input_shape()

    return hailo_face_detection_inference, input_height, input_width


async def main():
    args = parse_args()

    # Create WebSocket client
    ws_client = WebSocketClient(WS_URL, RECONNECT_INTERVAL)

    # Run bash commands and capture their outputs (for initial connection only)
    command_outputs = await ws_client.run_bash_commands()

    # Try to connect to the WebSocket server initially
    await ws_client.connect()

    # Load the HEF model for face detection
    hailo_face_detection_inference, input_height, input_width = init_hailo_obj(args.face_det, 'FLOAT32', 'FLOAT32')

    # Capture video from webcam
    cap = init_cv_cap()
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    fps_start_time = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff
        fps_start_time = fps_end_time

        original_frame = frame.copy()

        faces = face_detection(hailo_face_detection_inference, frame, input_width, input_height)

        # Ensure all data types are compatible with JSON serialization
        tensors = [{'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'score': float(score)} for
                   (x1, y1, x2, y2, score) in faces]

        # Draw bounding boxes
        for (x1, y1, x2, y2, score) in faces:
            draw_bounding_box(original_frame, score, (x1, y1), (x2, y2))

        # Display FPS
        cv2.putText(original_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)

        # Attempt reconnection if WebSocket is not connected
        if ws_client.websocket is None:
            ws_client.first_connection = True
            print("Attempting to reconnect to WebSocket...")
            await ws_client.connect()
            if ws_client.websocket is None:
                print("Reconnection failed. Retrying in 5 seconds.")
                await asyncio.sleep(RECONNECT_INTERVAL)
                continue

        # Send frame, tensors, and command outputs (only on first connect)
        if ws_client.first_connection:
            await ws_client.send_data(original_frame, tensors, command_outputs)
            ws_client.first_connection = False  # Reset flag after first send
        else:
            await ws_client.send_data(original_frame, tensors)

        # Show the frame with landmarks
        cv2.imshow('Webcam Face Landmarks', original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hailo_face_detection_inference.release_device()

if __name__ == '__main__':
    asyncio.run(main())








    #
    # hef = HEF(args.net)
    # input_vstream_info = hef.get_input_vstream_infos()[0]
    # output_vstream_infos = hef.get_output_vstream_infos()
    # input_shape = input_vstream_info.shape
    # input_height, input_width, _ = input_shape
    #
    # output_names = {
    #     'bbox_pred': 'scrfd_10g/conv54',
    #     'cls_pred': 'scrfd_10g/conv53'
    # }
    # first_connection = True
    #
    # with VDevice(device_ids=devices) as target:
    #     configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    #     network_group = target.configure(hef, configure_params)[0]
    #     network_group_params = network_group.create_params()
    #
    #     input_vstreams_params = InputVStreamParams.make_from_network_group(
    #         network_group, quantized=False, format_type=FormatType.FLOAT32
    #     )
    #     output_vstreams_params = OutputVStreamParams.make_from_network_group(
    #         network_group, quantized=False, format_type=FormatType.FLOAT32
    #     )
    #
    #     with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
    #         cap = cv2.VideoCapture(0)
    #         cap.set(cv2.CAP_PROP_FPS, 30)
    #         if not cap.isOpened():
    #             print("Failed to open camera.")
    #             return
    #
    #         try:
    #             while True:
    #                 ret, frame = cap.read()
    #                 if not ret:
    #                     print("Failed to read frame from camera.")
    #                     break
    #
    #                 original_frame = frame.copy()
    #                 processed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess(frame, input_size=(input_width, input_height))
    #                 processed_frame = processed_frame.astype(np.float32)
    #                 processed_frame = np.expand_dims(processed_frame, axis=0)
    #
    #                 input_data = {input_vstream_info.name: processed_frame}
    #
    #                 with network_group.activate(network_group_params):
    #                     raw_detections = infer_pipeline.infer(input_data)
    #
    #                 outputs = {
    #                     'scrfd_10g/conv54': raw_detections[output_names['bbox_pred']],
    #                     'scrfd_10g/conv53': raw_detections[output_names['cls_pred']]
    #                 }
    #
    #                 faces = postprocess(outputs, img_w, img_h, scale, pad_w, pad_h)
    #
    #                 # Ensure all data types are compatible with JSON serialization
    #                 tensors = [{'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'score': float(score)} for
    #                            (x1, y1, x2, y2, score) in faces]