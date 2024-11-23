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
import scipy.special
import subprocess

RECONNECT_INTERVAL = 3
WS_URL = "ws://192.168.0.233:5000"
WEBSOCKET = None

def parse_args():
    parser = argparse.ArgumentParser(description='Face detection using scrfd_10g.hef with OpenCV camera feed')
    parser.add_argument('hef', help="HEF file path")
    args = parser.parse_args()
    return args

def preproc(image, input_size=(640, 640)):
    """
    Preprocess the image: resize to (width, height) with padding to maintain aspect ratio.
    """
    img_h, img_w = image.shape[:2]
    input_w, input_h = input_size
    scale = min(input_w / img_w, input_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
    pad_w, pad_h = (input_w - new_w) // 2, (input_h - new_h) // 2
    padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized_image
    return padded_image, scale, pad_w, pad_h, img_w, img_h  # Return original image dimensions

def postprocess(outputs, img_w, img_h, scale, pad_w, pad_h, score_threshold=0.65, nms_threshold=0.4):
    """
    Postprocess the model outputs to extract face bounding boxes from the medium scale (40x40).
    """

    def decode_bboxes(bbox_pred, anchors, variances=[0.1, 0.2]):
        # bbox_pred: [N, 4], anchors: [N, 4]
        # anchors are in [cx, cy, w, h] format

        # Compute center coordinates and size
        boxes = np.zeros_like(bbox_pred)

        boxes[:, 0] = anchors[:, 0] + bbox_pred[:, 0] * variances[0] * anchors[:, 2]  # cx
        boxes[:, 1] = anchors[:, 1] + bbox_pred[:, 1] * variances[0] * anchors[:, 3]  # cy
        boxes[:, 2] = anchors[:, 2] * np.exp(bbox_pred[:, 2] * variances[1])  # w
        boxes[:, 3] = anchors[:, 3] * np.exp(bbox_pred[:, 3] * variances[1])  # h

        # Convert to corner coordinates
        x_min = boxes[:, 0] - boxes[:, 2] / 2  # x_min
        y_min = boxes[:, 1] - boxes[:, 3] / 2  # y_min
        x_max = boxes[:, 0] + boxes[:, 2] / 2  # x_max
        y_max = boxes[:, 1] + boxes[:, 3] / 2  # y_max

        boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        return boxes

    def generate_anchors(fm_sizes, input_size, steps, min_sizes):
        anchors = []
        for idx, fm_size in enumerate(fm_sizes):
            scale = input_size / steps[idx]
            fm_w, fm_h = fm_size
            for i in range(fm_h):
                for j in range(fm_w):
                    cx = (j + 0.5) / scale
                    cy = (i + 0.5) / scale
                    for min_size in min_sizes[idx]:
                        w = min_size / input_size
                        h = min_size / input_size
                        anchors.append([cx, cy, w, h])
        return np.array(anchors)

    def nms(boxes, scores, nms_threshold):
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold
        )
        return indices.flatten() if len(indices) > 0 else []

    # Model parameters for the medium scale
    input_size = 640
    variances = [0.1, 0.2]
    steps = [16]  # Medium scale step size
    min_sizes = [[64, 128]]  # Anchors for the medium scale
    score_threshold = 0.65
    nms_threshold = 0.4

    # Feature map sizes
    fm_sizes = [(input_size // step, input_size // step) for step in steps]

    # Generate anchors
    anchors = generate_anchors(fm_sizes, input_size, steps, min_sizes)

    # Extract outputs for the medium scale
    bbox_pred = outputs['scrfd_10g/conv54'][0]  # Shape: [40, 40, 8]
    cls_score = outputs['scrfd_10g/conv53'][0]  # Shape: [40, 40, 2]

    H, W, _ = bbox_pred.shape
    num_anchors = 2  # Since bbox_pred has 8 channels and 4 values per box

    # Reshape bbox_pred
    bbox_pred = bbox_pred.reshape(-1, 4)  # [H*W*num_anchors, 4]

    # Reshape cls_score
    cls_score = cls_score.reshape(-1)  # [H*W*num_anchors,]

    # Apply sigmoid activation
    scores = scipy.special.expit(cls_score)  # [H*W*num_anchors,]

    mask = scores > score_threshold

    # Now indexing will work
    bbox_pred = bbox_pred[mask]
    scores = scores[mask]
    anchors = anchors[mask]

    # Decode bounding boxes
    boxes = decode_bboxes(bbox_pred, anchors, variances)

    # print(boxes)
    # print(input_size)
    # Adjust boxes to input_size coordinates
    boxes[:, 0] *= input_size  # x_min
    boxes[:, 1] *= input_size  # y_min
    boxes[:, 2] *= input_size  # x_max
    boxes[:, 3] *= input_size  # y_max
    # print(boxes[:, 0] * input_size )
    # print(boxes[:, 1] * input_size )
    # print(boxes[:, 2] * input_size )
    # print(boxes[:, 3] * input_size )
    #
    # print(pad_w, pad_h)
    # Remove padding to get boxes in resized image coordinates
    boxes[:, 0] -= pad_w
    boxes[:, 1] -= pad_h
    boxes[:, 2] -= pad_w
    boxes[:, 3] -= pad_h

    boxes[:, 0] /= scale * 1.1
    boxes[:, 1] /= scale * 1.1
    boxes[:, 2] /= scale * 1.45
    boxes[:, 3] /= scale * 1.45

    # Clip coordinates to original image size
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)

    # Apply NMS
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # width
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # height
    indices = nms(boxes_xywh, scores, nms_threshold)
    boxes = boxes[indices]
    scores = scores[indices]

    # Convert to integers
    boxes = boxes.astype(int)

    results = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        results.append((x1, y1, x2, y2, score))

    return results

# ---------------------------------------------------------------------------------------------------------------------

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

    hef = HEF(args.hef)
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