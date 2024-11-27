#!/usr/bin/env python3

import numpy as np
import cv2
import argparse
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                            InputVStreamParams, OutputVStreamParams, FormatType)
from PIL import Image, ImageDraw
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Face detection using scrfd_10g.hef with OpenCV camera feed')
    parser.add_argument('hef', help="HEF file path")
    args = parser.parse_args()
    return args

def preproc(image, width=640, height=640):
    """
    Preprocess the image: resize to (width, height) with padding to maintain aspect ratio.
    """
    img_h, img_w = image.shape[:2]
    scale = min(width / img_w, height / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((height, width, 3), 114, dtype=np.uint8)
    pad_w, pad_h = (width - new_w) // 2, (height - new_h) // 2
    padded_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized_image
    return padded_image

def postprocess(detections, original_image, input_size):
    """
    Postprocess the model output to extract face bounding boxes and draw them on the image.
    """
    # Assuming detections are in the format [batch, num_detections, 5] -> [x1, y1, x2, y2, score]
    # Adjust this according to your model's actual output format.
    boxes = detections[0]  # Assuming batch size of 1
    h, w = original_image.shape[:2]
    input_h, input_w = input_size

    # Scale boxes back to original image size
    scale = min(input_w / w, input_h / h)
    pad_w, pad_h = (input_w - w * scale) / 2, (input_h - h * scale) / 2

    results = []
    for box in boxes:
        x1, y1, x2, y2, score = box
        if score < 0.5:
            continue
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
    output_vstream_info = hef.get_output_vstream_infos()[0]
    input_shape = input_vstream_info.shape  # Should be [height, width, channels]
    input_height, input_width, _ = input_shape

    with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef)
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
                    processed_frame = preproc(frame, width=input_width, height=input_height)
                    processed_frame = processed_frame.astype(np.float32)
                    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

                    # Prepare input data
                    input_data = {input_vstream_info.name: processed_frame}

                    # Run inference
                    with network_group.activate(network_group_params):
                        raw_detections = infer_pipeline.infer(input_data)

                    # Postprocess detections
                    detections = raw_detections[output_vstream_info.name]
                    faces = postprocess(detections, original_frame, (input_height, input_width))

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


# #!/usr/bin/env python3
#
# import cv2
# import numpy as np
# import os
# import sys
# import argparse
# import threading
# import queue
# from pathlib import Path
# from loguru import logger
# from PIL import Image
# from typing import List
# from face_det_utils import ObjectDetectionUtils
#
# # Add the parent directory to the system path to access utils module
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils import HailoAsyncInference
#
# def parse_args() -> argparse.Namespace:
#     """
#     Initialize argument parser for the script.
#
#     Returns:
#         argparse.Namespace: Parsed arguments.
#     """
#     parser = argparse.ArgumentParser(description="Detection Example")
#     parser.add_argument(
#         "-n", "--net",
#         help="Path for the network in HEF format.",
#         default="scrfd_10g.hef"
#     )
#     parser.add_argument(
#         "-i", "--input",
#         default="cam",
#         help="Path to the input - either an image or a folder of images."
#     )
#     parser.add_argument(
#         "-b", "--batch_size",
#         default=1,
#         type=int,
#         required=False,
#         help="Number of images in one batch"
#     )
#
#     args = parser.parse_args()
#
#     # Validate paths
#     if not os.path.exists(args.net):
#         raise FileNotFoundError(f"Network file not found: {args.net}")
#
#     return args
#
# def enqueue_images(
#     input_queue: queue.Queue,
#     width: int,
#     height: int,
#     utils: ObjectDetectionUtils
# ) -> None:
#     """
#     Capture frames from the camera, display them, preprocess them, and enqueue into the input queue.
#
#     Args:
#         input_queue (queue.Queue): Queue for input images.
#         width (int): Model input width.
#         height (int): Model input height.
#         utils (ObjectDetectionUtils): Utility class for object detection preprocessing.
#     """
#     cap = cv2.VideoCapture(0)  # Open the default camera (device index 0)
#     if not cap.isOpened():
#         logger.error("Could not open camera.")
#         return
#
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logger.error("Failed to read frame from camera.")
#                 break
#
#             # Display the camera feed
#             cv2.imshow('Camera Feed', frame)
#
#             # Exit if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#             # Convert frame from BGR (OpenCV default) to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(rgb_frame)
#
#             # Preprocess the image
#             processed_image = utils.preprocess(image, width, height)
#
#             # Enqueue the processed image as a batch of size 1
#             input_queue.put([processed_image])
#
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         input_queue.put(None)  # Signal the end of the stream
#
# def process_output(
#     output_queue: queue.Queue,
#     output_path: Path,
#     width: int,
#     height: int,
#     utils: ObjectDetectionUtils
# ) -> None:
#     """
#     Process and visualize the output results.
#
#     Args:
#         output_queue (queue.Queue): Queue for output results.
#         output_path (Path): Path to save the output images.
#         width (int): Image width.
#         height (int): Image height.
#         utils (ObjectDetectionUtils): Utility class for object detection visualization.
#     """
#     image_id = 0
#     while True:
#         result = output_queue.get()
#         if result is None:
#             break  # Exit the loop if sentinel value is received
#
#         processed_image, infer_results = result
#
#         # Handle output from HailoRT versions
#         if isinstance(infer_results, list) and len(infer_results) == 1:
#             infer_results = infer_results[0]
#
#         detections = utils.extract_detections(infer_results)
#
#         # Visualize detections on the image
#         utils.visualize(
#             detections, processed_image, image_id,
#             output_path, width, height
#         )
#
#         # Display the detection results
#         result_image = np.array(processed_image)
#         result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
#         cv2.imshow('Detections', result_image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#         print()
#         image_id += 1
#
#     cv2.destroyAllWindows()
#     output_queue.task_done()
#
# def infer(
#     net_path: str,
#     batch_size: int,
#     output_path: Path
# ) -> None:
#     """
#     Initialize queues, HailoAsyncInference instance, and run the inference.
#
#     Args:
#         net_path (str): Path to the HEF model file.
#         batch_size (int): Number of images per batch.
#         output_path (Path): Path to save the output images.
#     """
#     utils = ObjectDetectionUtils()
#     print("Utils initialized...")
#
#     input_queue = queue.Queue()
#     output_queue = queue.Queue()
#
#     hailo_inference = HailoAsyncInference(net_path, input_queue, output_queue, batch_size)
#     print("Hailo inference initialized...")
#
#     height, width, _ = hailo_inference.get_input_shape()
#     print("Input shape: ", height, width)
#
#     enqueue_thread = threading.Thread(
#         target=enqueue_images,
#         args=(input_queue, width, height, utils)
#     )
#
#     process_thread = threading.Thread(
#         target=process_output,
#         args=(output_queue, output_path, width, height, utils)
#     )
#
#     enqueue_thread.start()
#     print("enqueue_thread started")
#
#     process_thread.start()
#     print("process_thread started")
#
#     hailo_inference.run()
#
#     enqueue_thread.join()
#     output_queue.put(None)  # Signal process thread to exit
#     process_thread.join()
#
#     logger.info("Inference stopped. Exiting.")
#
# def main():
#     # Load the model
#     args = parse_args()
#
#     # Create output directory if it doesn't exist
#     output_path = Path('output_images')
#     output_path.mkdir(exist_ok=True)
#
#     # Start the inference
#     infer(args.net, args.batch_size, output_path)
#
# if __name__ == '__main__':
#     main()
