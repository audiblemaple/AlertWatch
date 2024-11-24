#!/usr/bin/env python3

import numpy as np
import cv2
import argparse
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)
import time
import scipy.special

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
    return padded_image, scale, pad_w, pad_h

def postprocess(outputs, scale, pad_w, pad_h, score_threshold=0.7, nms_threshold=0.4):
    """
    Postprocess the model outputs to extract face bounding boxes from the smallest scale (20x20).
    """
    def decode_bboxes(bbox_pred, anchors, variances=[0.1, 0.2]):
        # bbox_pred: [N, 4], anchors: [N, 4]
        boxes = np.zeros_like(bbox_pred)
        boxes[:, 0] = anchors[:, 0] + bbox_pred[:, 0] * variances[0] * anchors[:, 2]
        boxes[:, 1] = anchors[:, 1] + bbox_pred[:, 1] * variances[0] * anchors[:, 3]
        boxes[:, 2] = anchors[:, 2] * np.exp(bbox_pred[:, 2] * variances[1])
        boxes[:, 3] = anchors[:, 3] * np.exp(bbox_pred[:, 3] * variances[1])
        # Convert to corner coordinates
        boxes[:, 0] -= boxes[:, 2] / 2  # x_min
        boxes[:, 1] -= boxes[:, 3] / 2  # y_min
        boxes[:, 2] += boxes[:, 0]      # x_max
        boxes[:, 3] += boxes[:, 1]      # y_max
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

    # Model parameters for the smallest scale
    input_size = 640
    variances = [0.1, 0.2]
    steps = [32]  # Smallest scale step size
    min_sizes = [[256, 512]]  # Anchors for the smallest scale
    score_threshold = 0.7
    nms_threshold = 0.4

    # Feature map sizes
    fm_sizes = [(input_size // step, input_size // step) for step in steps]
    # fm_sizes = [(20, 20)]  # For step 32

    # Generate anchors
    anchors = generate_anchors(fm_sizes, input_size, steps, min_sizes)
    # anchors.shape: [N, 4]

    # Extract outputs for the smallest scale
    bbox_pred = outputs['scrfd_10g/conv57'][0]  # Shape: [20, 20, 8]
    cls_score = outputs['scrfd_10g/conv56'][0]  # Shape: [20, 20, 2]

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

    # Adjust boxes to image scale
    boxes[:, 0] *= input_size  # x_min
    boxes[:, 1] *= input_size  # y_min
    boxes[:, 2] *= input_size  # x_max
    boxes[:, 3] *= input_size  # y_max

    # Apply NMS
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # width
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # height
    indices = nms(boxes_xywh, scores, nms_threshold)
    boxes = boxes[indices]
    scores = scores[indices]

    # Adjust boxes back to original image scale
    results = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        x1 = int((x1 - pad_w) / scale)
        y1 = int((y1 - pad_h) / scale)
        x2 = int((x2 - pad_w) / scale)
        y2 = int((y2 - pad_h) / scale)
        results.append((x1, y1, x2, y2, score))

    return results

def main():
    args = parse_args()

    devices = Device.scan()
    if not devices:
        print("No Hailo devices found")
        return

    hef = HEF(args.hef)

    # Get input and output info
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_infos = hef.get_output_vstream_infos()

    input_shape = input_vstream_info.shape  # Should be [height, width, channels]
    input_height, input_width, _ = input_shape

    # Collect output names for the smallest scale
    output_names = {
        'bbox_pred': 'scrfd_10g/conv57',  # Smallest scale bounding box predictions
        'cls_pred': 'scrfd_10g/conv56'    # Smallest scale classification scores
    }

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

                    # Preprocess the frame
                    processed_frame, scale, pad_w, pad_h = preproc(frame, input_size=(input_width, input_height))
                    processed_frame = processed_frame.astype(np.float32)
                    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

                    # Prepare input data
                    input_data = {input_vstream_info.name: processed_frame}

                    # Run inference
                    with network_group.activate(network_group_params):
                        raw_detections = infer_pipeline.infer(input_data)

                    # Collect outputs for the smallest scale
                    outputs = {
                        'scrfd_10g/conv57': raw_detections[output_names['bbox_pred']],
                        'scrfd_10g/conv56': raw_detections[output_names['cls_pred']]
                    }

                    # Postprocess detections
                    faces = postprocess(outputs, scale, pad_w, pad_h)

                    # Draw bounding boxes
                    for (x1, y1, x2, y2, score) in faces:
                        cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(original_frame, f"{score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Display the frame
                    cv2.imshow('Face Detection', original_frame)

                    # Exit if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
