import cv2
import torch
from PIL import Image
import numpy as np
import argparse
from face_landmarks_detection.utils.utils import init_cv_cap
from drawing_util import draw_landmarks, draw_head_pose, display_fps
from face_util import get_camera_intrinsics, estimate_head_pose_with_direction
from processing_util import get_preprocessing_pipeline
from eyes_util import handle_eye_aspect_ratio, calculate_blink_rate, calibrate_ear, calculate_ear
from loader_util import load_model, load_face_cascade

def parse_arguments():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Facial Landmarks Detection Testing")
    parser.add_argument("-n", "--net",
                        default='models/face_landmarks_epoch_9319.pth',
                        type=str,
                        help="Path for the .pth model.",
                        required=False)
    parser.add_argument("-m", "--mode",
                        default="face",
                        type=str,
                        help="face / eyes",
                        required=False)
    parser.add_argument("-d", "--dev",
                        default="cpu",
                        type=str,
                        help="cpu, cuda:0; rocm-(not implemented yet)",
                        required=False)
    return parser.parse_args()


def process_frame(frame, face_cascade, preprocess_pipeline, network, class_num, frame_count, fps, blink_counter,
                  current_blink_start, blink_durations):

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    all_landmarks = []
    face_tensors = []
    head_poses = []

    for (x, y, w, h) in faces:
        face = grayscale_image[y:y + h, x:x + w]
        face = preprocess_pipeline(Image.fromarray(face))
        face_tensors.append(face.unsqueeze(0))

    if face_tensors:
        face_batch = torch.cat(face_tensors)
        with torch.no_grad():
            landmarks_batch = network(face_batch)
        for i, (x, y, w, h) in enumerate(faces):
            landmarks = (landmarks_batch[i].view(int(class_num / 2), 2).detach().numpy() + 0.5) * np.array(
                [[w, h]]) + np.array([[x, y]])
            all_landmarks.append(landmarks)

            rotation_vector, translation_vector, direction = estimate_head_pose_with_direction(landmarks, frame.shape)
            head_poses.append((rotation_vector, translation_vector))

            cv2.putText(frame, f'Direction: {direction}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Process EAR, blink detection, and dynamic threshold calibration
            frame, blink_counter, current_blink_start, blink_durations = handle_eye_aspect_ratio(
                frame, landmarks, frame_count, fps, blink_counter, current_blink_start, blink_durations
            )
    else:
        cv2.putText(frame, f'No face detected...', (450, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return all_landmarks, head_poses, frame, blink_counter, current_blink_start, blink_durations

def main():
    args = parse_arguments()

    weights_path = args.net
    cascade_path = '../haarcascades/haarcascade_frontalface_alt.xml'
    class_num = 136 if args.mode == "face" else 24

    network = load_model(weights_path, class_num, args.dev)
    face_cascade = load_face_cascade(cascade_path)
    preprocess_pipeline = get_preprocessing_pipeline()
    cap = init_cv_cap(640, 640, 30)

    prev_time = cv2.getTickCount()
    cumulative_time = 0.0
    frame_count = 0

    # Variables for EAR calibration and blink tracking
    blink_counter = 0
    blink_durations = []
    current_blink_start = None
    ear_values = []
    calibrated_ear_threshold = 0.25

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        prev_time = current_time
        fps = 1 / time_diff if time_diff > 0 else 0

        cumulative_time += time_diff
        frame_count += 1
        avg_fps = frame_count / cumulative_time if cumulative_time > 0 else 0

        all_landmarks, head_poses, frame, blink_counter, current_blink_start, blink_durations = process_frame(
            frame, face_cascade, preprocess_pipeline, network, class_num, frame_count, fps, blink_counter,
            current_blink_start, blink_durations
        )

        for landmarks, (rotation_vector, translation_vector) in zip(all_landmarks, head_poses):
            draw_landmarks(frame, [landmarks])
            draw_head_pose(frame, rotation_vector, translation_vector, get_camera_intrinsics(frame.shape))

        # Update dynamic EAR threshold after calibration period (e.g., first 100 frames)
        if len(ear_values) < 100 and len(all_landmarks) > 0:
            left_eye = all_landmarks[0][36:42]
            right_eye = all_landmarks[0][42:48]
            avg_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            ear_values.append(avg_ear)
            calibrated_ear_threshold = calibrate_ear(ear_values)

        # Calculate blink rate
        blink_rate = calculate_blink_rate(blink_durations, frame_count, fps)

        # Display average FPS value
        cv2.putText(frame, f'AVG FPS: {blink_rate:.2f} B/M', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255),
                    2)

        # Display blink rate and calibrated EAR threshold
        cv2.putText(frame, f'Blink Rate: {avg_fps:.2f} FPS', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255),
                    2)
        cv2.putText(frame, f'EAR Threshold: {calibrated_ear_threshold:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (230, 230, 0), 2)

        display_fps(frame, fps)
        cv2.imshow('Webcam Face Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# import cv2
# import torch
# from PIL import Image
# import numpy as np
# import argparse
# from face_landmarks_detection.utils.utils import init_cv_cap
#
#
# from drawing_util import draw_landmarks, draw_head_pose, display_fps
# from faceUtil import estimate_head_pose, get_camera_intrinsics
# from processing_util import get_preprocessing_pipeline
# from production.eyes_util import handle_eye_aspect_ratio
# from production.loader_util import load_model, load_face_cascade
#
#
# def parse_arguments():
#     """
#     Parses command-line arguments for the script.
#     """
#     parser = argparse.ArgumentParser(description="Facial Landmarks Detection Testing")
#     parser.add_argument("-n", "--net",
#                         default='models/face_landmarks_epoch_9319.pth',
#                         type=str,
#                         help="Path for the .pth model.",
#                         required=False)
#     parser.add_argument("-m", "--mode",
#                         default="face",
#                         type=str,
#                         help="face / eyes",
#                         required=False)
#     parser.add_argument("-d", "--dev",
#                         default="cpu",
#                         type=str,
#                         help="cpu, cuda:0 or rocm-(not implemented yet)",
#                         required=False)
#     return parser.parse_args()
#
# def process_frame(frame, face_cascade, preprocess_pipeline, network, class_num, frame_count, fps, blink_counter,
# current_blink_start, blink_durations):
#     grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
#                                           flags=cv2.CASCADE_SCALE_IMAGE)
#     all_landmarks = []
#     face_tensors = []
#     head_poses = []
#
#     for (x, y, w, h) in faces:
#         face = grayscale_image[y:y + h, x:x + w]
#         face = preprocess_pipeline(Image.fromarray(face))
#         face_tensors.append(face.unsqueeze(0))
#
#     if face_tensors:
#         face_batch = torch.cat(face_tensors)
#         with torch.no_grad():
#             landmarks_batch = network(face_batch)
#         for i, (x, y, w, h) in enumerate(faces):
#             landmarks = (landmarks_batch[i].view(int(class_num / 2), 2).detach().numpy() + 0.5) * np.array(
#                 [[w, h]]) + np.array([[x, y]])
#             all_landmarks.append(landmarks)
#
#             rotation_vector, translation_vector = estimate_head_pose(landmarks, frame.shape)
#             head_poses.append((rotation_vector, translation_vector))
#
#             # Process EAR and blink detection
#             frame, blink_counter, current_blink_start, blink_durations = handle_eye_aspect_ratio(
#                 frame, landmarks, frame_count, fps, blink_counter, current_blink_start, blink_durations
#             )
#
#     return all_landmarks, head_poses, frame, blink_counter, current_blink_start, blink_durations
#
#
# def main():
#     args = parse_arguments()
#
#     weights_path = args.net
#     cascade_path = '../haarcascades/haarcascade_frontalface_alt.xml'
#     class_num = 136 if args.mode == "face" else 24
#
#     network = load_model(weights_path, class_num, args.dev)
#     face_cascade = load_face_cascade(cascade_path)
#     preprocess_pipeline = get_preprocessing_pipeline()
#     cap = init_cv_cap(640, 640, 30)
#
#     prev_time = cv2.getTickCount()
#     cumulative_time = 0.0
#     frame_count = 0
#
#     blink_counter = 0
#     blink_durations = []
#     current_blink_start = None
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_count += 1
#         current_time = cv2.getTickCount()
#         time_diff = (current_time - prev_time) / cv2.getTickFrequency()
#         prev_time = current_time
#         fps = 1 / time_diff if time_diff > 0 else 0
#
#         cumulative_time += time_diff
#         frame_count += 1
#         avg_fps = frame_count / cumulative_time if cumulative_time > 0 else 0
#
#         all_landmarks, head_poses, frame, blink_counter, current_blink_start, blink_durations = process_frame(
#             frame, face_cascade, preprocess_pipeline, network, class_num, frame_count, fps, blink_counter,
#             current_blink_start, blink_durations
#         )
#
#         for landmarks, (rotation_vector, translation_vector) in zip(all_landmarks, head_poses):
#             draw_landmarks(frame, [landmarks])
#             draw_head_pose(frame, rotation_vector, translation_vector, get_camera_intrinsics(frame.shape))
#
#         display_fps(frame, fps)
#         cv2.imshow('Webcam Face Landmarks', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#
