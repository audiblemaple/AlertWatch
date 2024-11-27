#!/usr/bin/env python3
import time

import cv2
import argparse
import asyncio
import numpy as np
from face_landmarks_detection.utils.utils import init_cv_cap
from socketUtil import WebSocketClient
from drawUtil import draw_bounding_box
from inference import HailoInference, HailoInference_async
from processingUtil import preprocess_faces, postprocess_faces, preprocess_face_landmarks

WS_URL = "ws://192.168.0.233:5000"
RECONNECT_INTERVAL = 3

def init_cam_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def face_detection(hailo_object, frame, input_width, input_height):
    # output_names = {
    #     'bbox_pred': 'scrfd_10g/conv54',
    #     'cls_pred': 'scrfd_10g/conv53'
    # }

    output_names = {
        'bbox_pred': 'scrfd_10g/conv50',
        'cls_pred': 'scrfd_10g/conv49'
    }
    processed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_faces(
        frame, input_size=(input_width, input_height)
    )
    processed_frame = processed_frame.astype(np.float32)
    processed_frame = np.expand_dims(processed_frame, axis=0)

    raw_detections = hailo_object.run(processed_frame)

    # print("scrfd_10g/conv42")
    # print(raw_detections["scrfd_10g/conv42"].shape)
    # print("\n")
    #
    # print("scrfd_10g/conv43")
    # print(raw_detections["scrfd_10g/conv43"].shape)
    # print("\n")
    #
    # print("scrfd_10g/conv51")
    # print(raw_detections["scrfd_10g/conv51"].shape)
    # print("\n")
    #
    # print("scrfd_10g/conv49")
    # print(raw_detections["scrfd_10g/conv49"].shape)
    # print("\n")
    #
    # print("scrfd_10g/conv50")
    # print(raw_detections["scrfd_10g/conv50"].shape)
    # print("\n")
    #
    # print("scrfd_10g/conv57")
    # print(raw_detections["scrfd_10g/conv57"].shape)
    # print("\n")
    #
    # print("scrfd_10g/conv58")
    # print(raw_detections["scrfd_10g/conv58"].shape)
    # print("\n")
    #
    # print("scrfd_10g/conv41")
    # print(raw_detections["scrfd_10g/conv41"].shape)
    # print("\n")
    #
    # print("scrfd_10g/conv56")
    # print(raw_detections["scrfd_10g/conv56"].shape)
    # print("\n")
    #
    #
    # exit(1)

    outputs = {
        'scrfd_10g/conv54': raw_detections[output_names['bbox_pred']],
        'scrfd_10g/conv53': raw_detections[output_names['cls_pred']]
    }

    return postprocess_faces(outputs, img_w, img_h, scale, pad_w, pad_h)


def landmarks_detection(hailo_object, frame, input_width, input_height):
    results = hailo_object.run(frame)['face-landmarks-detection/fc1']

    # Scale and display landmarks
    return results[0].reshape(-1, 2)


def parse_args():
    parser = argparse.ArgumentParser(description='Face detection using scrfd_10g.hef with OpenCV camera feed')
    parser.add_argument("-fd", "--face_det",
                        default='../models/hailo8/scrfd_10g.hef',
                        type=str,
                        help="Path for the .hef model for face detection.",
                        required=False
                        )
    parser.add_argument("-fl", "--face_landmark",
                        default="../models/hailo8/face-landmarks-detection.hef",
                        type=str,
                        help="Path for the .hef model for face landmark detection.",
                        required=False
                        )
    args = parser.parse_args()
    return args

def init_hailo_obj(model, in_type, out_type):
    # Load the HEF model for face detection
    hailo_face_detection_inference = HailoInference_async(model, input_type=in_type, output_type=out_type)
    # hailo_face_detection_inference = HailoInference(model, input_type=in_type, output_type=out_type)
    # Get input shape for face detection
    input_height, input_width, _ = hailo_face_detection_inference.get_input_shape()

    return hailo_face_detection_inference, input_height, input_width


async def main():
    args = parse_args()

    # Create WebSocket client
    ws_client = WebSocketClient(WS_URL, RECONNECT_INTERVAL)

    # Load the HEF model for face detection
    inference, input_height, input_width = init_hailo_obj(args.face_det, 'FLOAT32', 'FLOAT32')

    # Run bash commands and capture their outputs (for initial connection only)
    command_outputs = await ws_client.run_bash_commands()

    # Try to connect to the WebSocket server initially
    await ws_client.connect()

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

        ## Face detection
        faces = face_detection(inference, frame, input_width, input_height)
        # inference.switch_model(args.face_landmark, 'UINT8', 'UINT8')

        # Draw bounding boxes
        for (x1, y1, x2, y2, score) in faces:
            draw_bounding_box(original_frame, score, (x1, y1), (x2, y2))

        # Ensure all data types are compatible with JSON serialization
        tensors = [{'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2), 'score': float(score)} for
                   (x1, y1, x2, y2, score) in faces]

        padding = 10
        for (x, y, w, h, score) in faces:
            # Add padding to the face region
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)

            face_roi = frame[y_start:y_end, x_start:x_end]

            # Preprocess the face region
            # preprocessed_face = preprocess_face_landmarks(face_roi, target_size=(224, 224))

            # Run inference
            # landmarks = landmarks_detection(inference, preprocessed_face, input_width, input_height)

            # for (lx, ly) in landmarks:
            #     cv2.circle(original_frame, (x + lx, y + ly), 2, (0, 255, 0), -1)

        # inference.switch_model(args.face_det, 'FLOAT32', 'FLOAT32')

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
    inference.release_device()

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
# from processingUtil import preprocess_faces, postprocess_faces, preprocess_face_landmarks
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
