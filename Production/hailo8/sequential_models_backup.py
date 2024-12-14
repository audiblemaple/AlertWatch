#!/usr/bin/env python3
import time

import cv2
import asyncio
import numpy as np
from socketUtil import WebSocketClient
from drawUtil import draw_bounding_box
from inference import HailoInference_async, HailoInferenceAsyncMultiModel
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
from processingUtil import preprocess_faces, postprocess_faces, preprocess_face_landmarks

from util import init_cv_cap


WS_URL = "ws://192.168.0.233:5000"
RECONNECT_INTERVAL = 3

def face_detection(raw_detections, frame, input_width, input_height):
    output_names = {
        'bbox_pred': 'scrfd_10g/conv50',
        'cls_pred': 'scrfd_10g/conv49'
    }
    processed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_faces(
        frame, input_size=(input_width, input_height)
    )

    outputs = {
        'scrfd_10g/conv54': raw_detections[output_names['bbox_pred']],
        'scrfd_10g/conv53': raw_detections[output_names['cls_pred']]
    }

    return postprocess_faces(outputs, img_w, img_h, scale, pad_w, pad_h)


def run_face_detection(vdevice, input_image, model):
    # Load the face detection model
    face_detection_model = vdevice.create_infer_model(model)
    face_detection_model.input("input_layer1").set_format_type(FormatType.FLOAT32)
    face_detection_model.output().set_format_type(FormatType.FLOAT32)

    # Configure the model
    with face_detection_model.configure() as configured_model:
        # Prepare input and output buffers
        bindings = configured_model.create_bindings()
        bindings.input().set_buffer(input_image)
        output_buffer = np.empty(face_detection_model.output().shape, dtype=np.float32)
        bindings.output().set_buffer(output_buffer)

        # Run inference
        configured_model.wait_for_async_ready(timeout_ms=10000)
        configured_model.run([bindings])

        return output_buffer  # Bounding boxes


def run_face_landmarks(vdevice, cropped_face, model):
    # Load the face landmarks model
    face_landmarks_model = vdevice.create_infer_model(model)
    face_landmarks_model.input("input_layer1").set_format_type(FormatType.FLOAT32)
    face_landmarks_model.output().set_format_type(FormatType.FLOAT32)

    # Configure the model
    with face_landmarks_model.configure() as configured_model:
        # Prepare input and output buffers
        bindings = configured_model.create_bindings()
        bindings.input().set_buffer(cropped_face)
        output_buffer = np.empty(face_landmarks_model.output().shape, dtype=np.float32)
        bindings.output().set_buffer(output_buffer)

        # Run inference
        configured_model.wait_for_async_ready(timeout_ms=10000)
        configured_model.run([bindings])

        return output_buffer  # Landmarks


# def process_face(frame, bbox, input_shape):
#     """
#     Prepares the face region for inference.
#
#     Args:
#         frame (np.ndarray): Video frame.
#         bbox (tuple): Bounding box of the face (x, y, w, h).
#         input_shape (tuple): Shape of the model input (height, width, channels).
#
#     Returns:
#         np.ndarray: Preprocessed face region.
#         tuple: Adjusted bounding box for cropping.
#     """
#     x, y, w, h = bbox
#     x_start = max(0, x)
#     y_start = max(0, y)
#     x_end = min(frame.shape[1], x + w)
#     y_end = min(frame.shape[0], y + h)
#     face_roi = frame[y_start:y_end, x_start:x_end]
#
#     preprocessed_face = preprocess_face_landmarks(face_roi, target_size=input_shape[:2])
#     return preprocessed_face, (x, y, w, h)


def process_face(frame, bbox, input_shape):
    x1, y1, x2, y2 = bbox
    x_start = max(0, x1)
    y_start = max(0, y1)
    x_end = min(frame.shape[1], x2)
    y_end = min(frame.shape[0], y2)
    face_roi = frame[y_start:y_end, x_start:x_end]

    if face_roi.size == 0:
        print(f"Invalid face ROI: {bbox}")
        return None, bbox
    preprocessed_face = preprocess_face_landmarks(face_roi, target_size=input_shape[:2])
    return preprocessed_face, bbox


def adjust_landmarks(landmarks, bbox):
    """
    Adjusts landmark positions based on the face bounding box.

    Args:
        landmarks (np.ndarray): Predicted landmarks.
        bbox (tuple): Bounding box of the face (x, y, w, h).

    Returns:
        np.ndarray: Adjusted landmarks.
    """
    x, y, w, h = bbox
    return (landmarks + 0.5) * np.array([[w, h]]) + np.array([[x, y]])


def draw_results(frame, all_landmarks, fps, face_detected):
    """
    Draws landmarks, FPS, and detection status on the frame.

    Args:
        frame (np.ndarray): Video frame.
        all_landmarks (list): List of landmarks for all detected faces.
        fps (float): Current frames per second.
        face_detected (bool): Whether a face was detected.
    """
    # Draw landmarks
    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)


    # Display face detection status
    text = "Face Detected" if face_detected else "No Face Detected"
    color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)




















async def main():
    # model paths
    face_det = "../models/hailo8/scrfd_10g.hef"
    face_landmark = "../models/hailo8/face-landmarks-detection.hef"

    # Create WebSocket client
    # ws_client = WebSocketClient(WS_URL, RECONNECT_INTERVAL)

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN


    # Run bash commands and capture their outputs (for initial connection only)
    # command_outputs = await ws_client.run_bash_commands()

    # Try to connect to the WebSocket server initially
    # await ws_client.connect()

    # Capture video from webcam
    cap = init_cv_cap()
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    class_num = 136
    fps_start_time = 0
    fps = 0
    timeout_ms = 10000

    # Initialize the object with two models
    model_paths = (face_det, face_landmark)
    hailo_inference = HailoInferenceAsyncMultiModel(
        model_paths, input_types=('FLOAT32', 'UINT8'), output_types=('FLOAT32', 'FLOAT32')
    )

    # Get input shapes for the models
    face_detection_input_shape = hailo_inference.get_input_shape(model_id=1)
    face_landmarks_input_shape = hailo_inference.get_input_shape(model_id=2)
    print(f"face_detection_input_shape: {face_detection_input_shape}")
    print(f"face_landmarks_input_shape: {face_landmarks_input_shape}")


    # TODO: Find the out layer using the shape of the layers
    face_det_input_name = hailo_inference.models['model_1']['output_vstream_info']
    # TODO

    # TODO: Nothing this is good
    face_land_output_name = hailo_inference.models['model_2']['output_vstream_info'][0].name

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

        processed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_faces(
            frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
        )
        processed_frame = processed_frame.astype(np.float32)
        processed_frame = np.expand_dims(processed_frame, axis=0)

        # Run inference for Face Detection
        raw_faces = hailo_inference.run(model_id=1, input_data=processed_frame)
        faces = face_detection(raw_faces, frame, face_detection_input_shape[0], face_detection_input_shape[1])
        # print("Model 1 Output:", faces)

        all_landmarks = []

        if len(faces) == 0:
            print("No faces detected.")
            continue

        # Draw bounding boxes
        for (x1, y1, x2, y2, score) in faces:
            # Clip bounding box coordinates
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            draw_bounding_box(original_frame, score, (x1, y1), (x2, y2))

            preprocessed_face, adjusted_bbox = process_face(frame, (x1, y1, x2, y2), face_landmarks_input_shape)

            if preprocessed_face is None:
                continue

            # Display the preprocessed face
            if preprocessed_face is not None:
                # Reshape preprocessed face for display if needed
                preprocessed_face_display = preprocessed_face.squeeze()
                if len(preprocessed_face_display.shape) == 2:
                    # Convert grayscale to BGR for consistent display with OpenCV
                    preprocessed_face_display = cv2.cvtColor(preprocessed_face_display, cv2.COLOR_GRAY2BGR)
                cv2.imshow('Preprocessed Face', preprocessed_face_display)

            # Run inference Face Landmarks Detection
            landmarks = hailo_inference.run(model_id=2, input_data=preprocessed_face)
            landmarks_batch = landmarks.get(face_land_output_name, None)
            if landmarks_batch is None:
                print("Error: Landmarks output missing!")
                continue
            # print("Model 2 Output:", landmarks_batch)

            try:
                # Reshape and adjust landmarks
                landmarks = landmarks_batch[0].reshape(class_num // 2, 2)
                adjusted_landmarks = adjust_landmarks(landmarks, adjusted_bbox)
                all_landmarks.append(adjusted_landmarks)
            except ValueError as e:
                print(f"Error processing landmarks: {e}")
                continue



        # Draw results on the frame
        draw_results(original_frame, all_landmarks, fps, len(faces) > 0)

            # Ensure all data types are compatible with JSON serialization
            # tensors = [{'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'score': float(score)} for
            #            (x1, y1, x2, y2, score) in faces]


        # Display FPS
        cv2.putText(original_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)



        # Attempt reconnection if WebSocket is not connected
        # if ws_client.websocket is None:
        #     ws_client.first_connection = True
        #     print("Attempting to reconnect to WebSocket...")
        #     await ws_client.connect()
        #     if ws_client.websocket is None:
        #         print("Reconnection failed. Retrying in 5 seconds.")
        #         await asyncio.sleep(RECONNECT_INTERVAL)
        #         continue

        # Send frame, tensors, and command outputs (only on first connect)
        # if ws_client.first_connection:
        #     await ws_client.send_data(original_frame, tensors, command_outputs)
        #     ws_client.first_connection = False  # Reset flag after first send
        # else:
        #     await ws_client.send_data(original_frame, tensors)

        # Show the frame with landmarks
        cv2.imshow('Webcam Face Landmarks', original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())


#
# #!/usr/bin/env python3
# import time
#
# import cv2
# import argparse
# import asyncio
# import numpy as np
# from face_landmarks_detection.utils.utils import init_cv_cap
# from socketUtil import WebSocketClient
# from drawUtil import draw_bounding_box
# from inference import HailoInference
# from prePostProcessing import preprocess_faces, postprocess_faces, preprocess_face_landmarks
#
# WS_URL = "ws://192.168.0.233:5000"
# RECONNECT_INTERVAL = 3
#
# def init_cam_feed():
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     return cap
#
# def face_detection(hailo_object, frame, input_width, input_height):
#     output_names = {
#         'bbox_pred': 'scrfd_10g/conv54',
#         'cls_pred': 'scrfd_10g/conv53'
#     }
#     processed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_faces(
#         frame, input_size=(input_width, input_height)
#     )
#     processed_frame = processed_frame.astype(np.float32)
#     processed_frame = np.expand_dims(processed_frame, axis=0)
#
#     raw_detections = hailo_object.run(processed_frame)
#
#     outputs = {
#         'scrfd_10g/conv54': raw_detections[output_names['bbox_pred']],
#         'scrfd_10g/conv53': raw_detections[output_names['cls_pred']]
#     }
#
#     return postprocess_faces(outputs, img_w, img_h, scale, pad_w, pad_h)
#
#
# def landmarks_detection(hailo_object, frame, input_width, input_height):
#     print("dsd")
#     # results = hailo_object.run(frame)['face-landmarks-detection/fc1']
#     results = hailo_object.run(frame)
#     print(results)
#
#     # Scale and display landmarks
#     return results[0].reshape(-1, 2)
#
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Face detection using scrfd_10g.hef with OpenCV camera feed')
#     parser.add_argument("-fd", "--face_det",
#                         default='model/scrfd_10g.hef',
#                         type=str,
#                         help="Path for the .hef model for face detection.",
#                         required=False
#                         )
#     parser.add_argument("-fl", "--face_landmark",
#                         default="model/face-landmarks-detection.hef",
#                         type=str,
#                         help="Path for the .hef model for face landmark detection.",
#                         required=False
#                         )
#     args = parser.parse_args()
#     return args
#
# def init_hailo_obj(model, in_type, out_type):
#     # Load the HEF model for face detection
#     hailo_face_detection_inference = HailoInference(model, input_type=in_type, output_type=out_type)
#     # Get input shape for face detection
#     input_height, input_width, _ = hailo_face_detection_inference.get_input_shape()
#
#     return hailo_face_detection_inference, input_height, input_width
#
#
# async def main():
#     args = parse_args()
#
#     # Create WebSocket client
#     ws_client = WebSocketClient(WS_URL, RECONNECT_INTERVAL)
#
#     # Load the HEF model for face detection
#     hailo_face_detection_inference, input_height, input_width = init_hailo_obj(args.face_det, 'FLOAT32', 'FLOAT32')
#
#     # Load the HEF model for face landmarks detection
#     hailo_face_landmark_detection_inference, input_height, input_width = init_hailo_obj(args.face_landmark, 'UINT8',
#                                                                                         'UINT8')
#
#     # Run bash commands and capture their outputs (for initial connection only)
#     command_outputs = await ws_client.run_bash_commands()
#
#     # Try to connect to the WebSocket server initially
#     await ws_client.connect()
#
#     # Capture video from webcam
#     cap = init_cv_cap()
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
#
#     fps_start_time = 0
#     fps = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break
#
#         fps_end_time = time.time()
#         time_diff = fps_end_time - fps_start_time
#         fps = 1 / time_diff
#         fps_start_time = fps_end_time
#
#         original_frame = frame.copy()
#
#         ## Face detection
#         faces = face_detection(hailo_face_detection_inference, frame, input_width, input_height)
#
#         # Draw bounding boxes
#         for (x1, y1, x2, y2, score) in faces:
#             draw_bounding_box(original_frame, score, (x1, y1), (x2, y2))
#
#         # Ensure all data types are compatible with JSON serialization
#         tensors = [{'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'score': float(score)} for
#                    (x1, y1, x2, y2, score) in faces]
#
#         padding = 10
#         for (x, y, w, h, score) in faces:
#             # Add padding to the face region
#             x_start = max(0, x - padding)
#             y_start = max(0, y - padding)
#             x_end = min(frame.shape[1], x + w + padding)
#             y_end = min(frame.shape[0], y + h + padding)
#
#             face_roi = frame[y_start:y_end, x_start:x_end]
#
#             # Preprocess the face region
#             preprocessed_face = preprocess_face_landmarks(face_roi, target_size=(input_height, input_width))
#
#
#             # Run inference
#             results = landmarks_detection(hailo_face_landmark_detection_inference, preprocessed_face, input_width, input_height)
#
#
#             # Scale and display landmarks
#             landmarks = results[0].reshape(-1, 2)
#
#             for (lx, ly) in landmarks:
#                 cv2.circle(frame, (x + lx, y + ly), 2, (0, 255, 0), -1)
#
#         # Display FPS
#         cv2.putText(original_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)
#
#         # Attempt reconnection if WebSocket is not connected
#         if ws_client.websocket is None:
#             ws_client.first_connection = True
#             print("Attempting to reconnect to WebSocket...")
#             await ws_client.connect()
#             if ws_client.websocket is None:
#                 print("Reconnection failed. Retrying in 5 seconds.")
#                 await asyncio.sleep(RECONNECT_INTERVAL)
#                 continue
#
#         # Send frame, tensors, and command outputs (only on first connect)
#         if ws_client.first_connection:
#             await ws_client.send_data(original_frame, tensors, command_outputs)
#             ws_client.first_connection = False  # Reset flag after first send
#         else:
#             await ws_client.send_data(original_frame, tensors)
#
#         # Show the frame with landmarks
#         cv2.imshow('Webcam Face Landmarks', original_frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()
#     hailo_face_detection_inference.release_device()
#
# if __name__ == '__main__':
#     asyncio.run(main())
#
