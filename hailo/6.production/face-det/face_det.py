#!/usr/bin/env python3

import numpy as np
import cv2
import argparse
import asyncio
import websockets
import json
import base64
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)

from face_util.processing import preproc, postprocess
import subprocess

RECONNECT_INTERVAL = 3
WS_URL = "ws://192.168.0.233:5000"
WEBSOCKET = None

def parse_args():
    parser = argparse.ArgumentParser(description='Face detection using scrfd_10g.hef with OpenCV camera feed')
    parser.add_argument("-n", "--net",
                        default='scrfd_10g.hef',
                        type=str,
                        help="Path for the .pth model.",
                        required=False
                        )

    args = parser.parse_args()
    return args


async def connect_to_websocket():
    global WEBSOCKET
    """Try to connect to the WebSocket server and return the connection if successful."""
    while True:
        try:
            WEBSOCKET = await websockets.connect(WS_URL)
            print("Connected to WebSocket server.")
            return WEBSOCKET
        except Exception as e:
            print(f"Warning: Unable to connect to WebSocket server: {e}")
            print(f"Reattempting connection in {RECONNECT_INTERVAL} seconds...")
            await asyncio.sleep(RECONNECT_INTERVAL)

async def run_bash_commands():
    """
    Run a list of bash commands and return their outputs.
    """
    commands = [
        "uname -a",
        "hailortcli fw-control identify"
    ]
    outputs = []
    for command in commands:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        outputs.append(result.stdout.strip())

    return outputs

async def send_data(frame, tensors, command_outputs=None):
    global WEBSOCKET
    if WEBSOCKET is None:
        return  # No WebSocket connection; skip sending data

    try:
        # Encode the frame as JPEG to reduce size
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Prepare data to send
        data = {
            'type': "detection_feed",
            'msgData': {
                'frame': frame_base64,
                'face_tensors': tensors,
                'face_landmark_tensors': None
            }
        }
        if command_outputs is not None:
            data['msgData']['commands'] = command_outputs

        await WEBSOCKET.send(json.dumps(data))
    except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidState) as e:
        # Force reconnect by setting websocket to None
        if WEBSOCKET is not None:
            WEBSOCKET = None
        print(f"Warning: Failed to send data via WebSocket: {e}")


async def main():
    global WEBSOCKET
    args = parse_args()

    # Run bash commands and capture their outputs (for initial connection only)
    command_outputs = await run_bash_commands()

    # Try to connect to the WebSocket server initially
    WEBSOCKET = await connect_to_websocket()

    devices = Device.scan()
    if not devices:
        print("No Hailo devices found")
        return

    hef = HEF(args.net)
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_infos = hef.get_output_vstream_infos()
    input_shape = input_vstream_info.shape
    input_height, input_width, _ = input_shape

    output_names = {
        'bbox_pred': 'scrfd_10g/conv54',
        'cls_pred': 'scrfd_10g/conv53'
    }
    first_connection = True

    with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                print("Failed to open camera.")
                return

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from camera.")
                        break

                    original_frame = frame.copy()
                    processed_frame, scale, pad_w, pad_h, img_w, img_h = preproc(frame, input_size=(input_width, input_height))
                    processed_frame = processed_frame.astype(np.float32)
                    processed_frame = np.expand_dims(processed_frame, axis=0)

                    input_data = {input_vstream_info.name: processed_frame}

                    with network_group.activate(network_group_params):
                        raw_detections = infer_pipeline.infer(input_data)

                    outputs = {
                        'scrfd_10g/conv54': raw_detections[output_names['bbox_pred']],
                        'scrfd_10g/conv53': raw_detections[output_names['cls_pred']]
                    }

                    faces = postprocess(outputs, img_w, img_h, scale, pad_w, pad_h)

                    # Ensure all data types are compatible with JSON serialization
                    tensors = [{'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'score': float(score)} for
                               (x1, y1, x2, y2, score) in faces]

                    # Draw bounding boxes
                    for (x1, y1, x2, y2, score) in faces:
                        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # cv2.putText(original_frame, f"Face box: {(x1, y1), (x2, y2)}", (x1, y1 - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(original_frame, f"Detection confidence: {score:.2f}%", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Attempt reconnection if WebSocket is not connected
                    if WEBSOCKET is None:
                        first_connection = True
                        print("Attempting to reconnect to WebSocket...")
                        WEBSOCKET = await connect_to_websocket()
                        if WEBSOCKET is None:
                            print("Reconnection failed. Retrying in 5 seconds.")
                            await asyncio.sleep(RECONNECT_INTERVAL)
                            continue

                    # Send frame, tensors, and command outputs (only on first connect)
                    if first_connection:
                        await send_data(original_frame, tensors, command_outputs)
                        first_connection = False  # Reset flag after first send
                    else:
                        await send_data(original_frame, tensors)

                    # Display the frame
                    cv2.imshow('Face Detection', original_frame)

                    # Exit if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())