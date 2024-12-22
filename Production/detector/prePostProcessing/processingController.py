import cv2
import numpy as np


def preprocess_face_landmarks(frame, bbox, input_shape, gray=True):
    """
    Preprocesses the face ROI for landmark detection.

    Args:
        frame (np.ndarray): The original video frame.
        bbox (tuple): Bounding box of the face (x1, y1, x2, y2).
        input_shape (tuple): The target input shape for the model.
        gray (bool): Whether to convert the face ROI to grayscale.

    Returns:
        tuple: (preprocessed_face, adjusted_bbox) or (None, bbox) if invalid.
    """
    x1, y1, x2, y2 = bbox

    # Ensure the coordinates are integers and within image bounds
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(frame.shape[1], int(round(x2)))
    y2 = min(frame.shape[0], int(round(y2)))

    # Validate bounding box dimensions
    if x2 <= x1 or y2 <= y1:
        return None, bbox

    face_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0:
        return None, bbox

    if gray:
        # Convert to grayscale
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Original face ROI dimensions
    face_h, face_w = face_roi.shape[:2]
    target_h, target_w = input_shape[:2]

    # Validate face ROI dimensions
    if face_w <= 0 or face_h <= 0:
        return None, bbox

    # Calculate scaling factor to ensure the resized image is at least as big as the target size
    scale = max(target_w / face_w, target_h / face_h)

    # New dimensions after scaling
    new_w = int(face_w * scale)
    new_h = int(face_h * scale)

    # Validate new dimensions
    if new_w <= 0 or new_h <= 0:
        return None, bbox

    # Resize the face ROI
    resized_face = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Calculate offsets to crop the center part
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2

    # Handle cases where the resized image is smaller than the target size
    preprocessed_face = np.zeros((target_h, target_w), dtype=resized_face.dtype)

    x_start = max(0, -x_offset)
    y_start = max(0, -y_offset)
    x_end = x_start + min(new_w, target_w)
    y_end = y_start + min(new_h, target_h)

    resized_x_start = max(0, x_offset)
    resized_y_start = max(0, y_offset)
    resized_x_end = resized_x_start + (x_end - x_start)
    resized_y_end = resized_y_start + (y_end - y_start)

    preprocessed_face[y_start:y_end, x_start:x_end] = resized_face[resized_y_start:resized_y_end,
                                                      resized_x_start:resized_x_end]

    # Expand dimensions to fit model input
    if gray:
        preprocessed_face = preprocessed_face[np.newaxis, :, :, np.newaxis]
    else:
        preprocessed_face = preprocessed_face[np.newaxis, :, :, :]

    return preprocessed_face, bbox

# def preprocess_face_landmarks(frame, bbox, input_shape, gray=True):
#     """
#     Preprocesses the face ROI for landmark detection.
#
#     Args:
#         frame (np.ndarray): The original video frame, BGR format.
#         bbox (tuple): Bounding box of the face (x1, y1, x2, y2).
#         input_shape (tuple): The target input shape for the model (H, W, ...).
#         gray (bool): Whether to convert the face ROI to grayscale.
#
#     Returns:
#         tuple: (preprocessed_face, adjusted_bbox) or (None, bbox) if invalid.
#     """
#     x1, y1, x2, y2 = bbox
#     height, width = frame.shape[:2]
#     target_h, target_w = input_shape[:2]
#
#     # Ensure coordinates are within frame boundaries
#     x1 = max(0, int(round(x1)))
#     y1 = max(0, int(round(y1)))
#     x2 = min(width, int(round(x2)))
#     y2 = min(height, int(round(y2)))
#
#     # Validate bounding box
#     if x2 <= x1 or y2 <= y1:
#         return None, bbox
#
#     face_roi = frame[y1:y2, x1:x2]
#     if face_roi.size == 0:
#         return None, bbox
#
#     # Convert to grayscale if needed
#     if gray:
#         face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
#
#     face_h, face_w = face_roi.shape[:2]
#     if face_h == 0 or face_w == 0:
#         return None, bbox
#
#     # Calculate scale to make sure resized face >= target size
#     scale = max(target_w / face_w, target_h / face_h)
#
#     new_w = int(face_w * scale)
#     new_h = int(face_h * scale)
#
#     # Resize (INTER_NEAREST is very fast)
#     resized_face = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
#
#     # Center crop the resized image to the target size
#     start_x = (new_w - target_w) // 2
#     start_y = (new_h - target_h) // 2
#     preprocessed_face = resized_face[start_y:start_y + target_h, start_x:start_x + target_w]
#
#     # Add batch dimension and channel dimension if needed
#     if gray:
#         # Shape: (1, H, W, 1)
#         preprocessed_face = preprocessed_face[np.newaxis, :, :, np.newaxis]
#     else:
#         # For color images, ensure shape: (1, H, W, C)
#         if preprocessed_face.ndim == 2:
#             preprocessed_face = preprocessed_face[:, :, np.newaxis]
#         preprocessed_face = preprocessed_face[np.newaxis, :, :, :]
#
#     return preprocessed_face, bbox


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


# Define the preprocessing function with uint8 output
# def preprocess_face_landmarks(image, target_size=(224, 224), gray=True):
#     if gray:
#         # Convert to grayscale
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Resize image
#     resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
#
#     # Expand dimensions to fit the model input
#     # expanded_image = np.expand_dims(resized_image, axis=[0, -1])
#
#     # return expanded_image
#     return resized_image[np.newaxis, ..., np.newaxis]


# def preprocess_faces(image, input_size=(640, 640)):
#     img_h, img_w = image.shape[:2]
#     input_w, input_h = input_size
#
#     # Calculate scaling factor and new dimensions
#     scale = min(input_w / img_w, input_h / img_h)
#     new_w, new_h = round(img_w * scale), round(img_h * scale)
#
#     # Resize image
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#
#     # Pre-allocate output image with padding value
#     pad_w, pad_h = (input_w - new_w) // 2, (input_h - new_h) // 2
#     padded_image = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
#
#     # Place resized image in the center
#     padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image
#
#     return padded_image, scale, pad_w, pad_h, img_w, img_h

def preprocess_face_detection(image, input_size=(640, 640)):
    img_h, img_w = image.shape[:2]
    input_w, input_h = input_size

    # Calculate scaling factor and dimensions
    scale = min(input_w / img_w, input_h / img_h)
    new_size = (round(img_w * scale), round(img_h * scale))

    # Resize image efficiently
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

    # Create padded output using numpy
    pad_w, pad_h = (input_w - new_size[0]) // 2, (input_h - new_size[1]) // 2
    padded_image = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
    padded_image[pad_h:pad_h + new_size[1], pad_w:pad_w + new_size[0]] = resized_image

    return padded_image, scale, pad_w, pad_h, img_w, img_h


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

input_size = 640

steps = [16]  # Medium scale step size
min_sizes = [[64, 128]]  # Anchors for the medium scale

# Feature map sizes
# fm_sizes = [(input_size // step, input_size // step) for step in steps]
fm_sizes = [(40,40)]

PRECOMPUTED_ANCHORS = generate_anchors(fm_sizes, input_size, steps, min_sizes)

def postprocess_faces(outputs, pad_w, pad_h, score_threshold=0.67, nms_threshold=0.4) -> list[(int, int, int, int, float)] | None :
    """
    Postprocess the model outputs to extract face bounding boxes from the medium scale (40x40).
    """

    def decode_bboxes(bbox_pred, anchors, variances=[0.1, 0.2]):
        boxes = np.zeros_like(bbox_pred)

        # Center coordinates
        boxes[:, 0] = anchors[:, 0] + bbox_pred[:, 0] * variances[0] * anchors[:, 2]  # cx
        boxes[:, 1] = anchors[:, 1] + bbox_pred[:, 1] * variances[0] * anchors[:, 3]  # cy

        # Size
        boxes[:, 2] = anchors[:, 2] * np.exp(bbox_pred[:, 2] * variances[1])  # w
        boxes[:, 3] = anchors[:, 3] * np.exp(bbox_pred[:, 3] * variances[1])  # h

        # Convert to corner coordinates
        x_min = boxes[:, 0] - boxes[:, 2] / 2
        y_min = boxes[:, 1] - boxes[:, 3] / 2
        x_max = boxes[:, 0] + boxes[:, 2] / 2
        y_max = boxes[:, 1] + boxes[:, 3] / 2

        return np.stack([x_min, y_min, x_max, y_max], axis=1)

    # def nms(boxes, scores, nms_threshold):
    #     indices = cv2.dnn.NMSBoxes(
    #         bboxes=boxes.tolist(),
    #         scores=scores.tolist(),
    #         score_threshold=score_threshold,
    #         nms_threshold=nms_threshold
    #     )
    #     return indices.flatten() if len(indices) > 0 else []

    # ^^^^ alternative implementation of the function above ^^^^^^
    # def nms(boxes, scores, nms_threshold):
    #     # Convert boxes to numpy array
    #     boxes = np.array(boxes)
    #     scores = np.array(scores)
    #
    #     indices = []
    #     sorted_indices = np.argsort(scores)[::-1]  # Sort by scores in descending order
    #
    #     while len(sorted_indices) > 0:
    #         # Select the index with the highest score
    #         current = sorted_indices[0]
    #         indices.append(current)
    #
    #         # Compare IoU of this box with the rest
    #         others = sorted_indices[1:]
    #         ious = compute_iou(boxes[current], boxes[others])
    #
    #         # Keep boxes with IoU less than the threshold
    #         sorted_indices = sorted_indices[1:][ious < nms_threshold]
    #
    #     return indices

    # def compute_iou(box, other_boxes):
    #     x1 = np.maximum(box[0], other_boxes[:, 0])
    #     y1 = np.maximum(box[1], other_boxes[:, 1])
    #     x2 = np.minimum(box[2], other_boxes[:, 2])
    #     y2 = np.minimum(box[3], other_boxes[:, 3])
    #
    #     inter_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    #     box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    #     other_areas = (other_boxes[:, 2] - other_boxes[:, 0] + 1) * (other_boxes[:, 3] - other_boxes[:, 1] + 1)
    #
    #     union_area = box_area + other_areas - inter_area
    #     iou = inter_area / union_area
    #     return iou


    def sigmoid(x) -> float :
        """Replace scipy.special.expit with NumPy implementation."""
        return 1 / (1 + np.exp(-x))

    # Model parameters for the medium scale
    # input_size = 640
    ''' [(x,y), (w,h)] '''
    # variances = [0.1, 0.2]     # bad
    # variances = [0.001, 0.003] # gud
    # variances = [0.000001, 0.003] # great
    variances = [0.000001, 0.005] # also great
    # steps = [16]  # Medium scale step size
    # min_sizes = [[64, 128]]  # Anchors for the medium scale

    # Feature map sizes
    # fm_sizes = [(input_size // step, input_size // step) for step in steps]
    # fm_sizes = [(40,40)]

    # Generate anchors
    anchors = generate_anchors(fm_sizes, input_size, steps, min_sizes)

    # Extract outputs for the medium scale
    bbox_pred = outputs['scrfd_10g/conv50'][0]  # Shape: [40, 40, 8]
    cls_score = outputs['scrfd_10g/conv49'][0]  # Shape: [40, 40, 2]

    H, W, _ = bbox_pred.shape

    # Reshape bbox_pred
    bbox_pred = bbox_pred.reshape(-1, 4)  # [H*W*num_anchors, 4]

    # Reshape cls_score
    cls_score = cls_score.reshape(-1)  # [H*W*num_anchors,]

    # Apply sigmoid activation
    scores = sigmoid(cls_score)  # [H*W*num_anchors,]

    mask = scores > score_threshold

    bbox_pred = bbox_pred[mask]
    scores = scores[mask]
    anchors = anchors[mask]

    # Decode bounding boxes
    boxes = decode_bboxes(bbox_pred, anchors, variances)

    # Adjust boxes to input_size coordinates
    boxes[:, 0] *= input_size  # x_min
    boxes[:, 1] *= input_size  # y_min
    boxes[:, 2] *= input_size  # x_max
    boxes[:, 3] *= input_size  # y_max

    # Remove padding to get boxes in resized image coordinates
    boxes[:, 0] -= pad_w
    boxes[:, 1] -= pad_h
    boxes[:, 2] -= pad_w
    boxes[:, 3] -= pad_h

    ''' move left side right + '''
    boxes[:, 0] -= 5

    ''' move top-left side down + '''
    # boxes[:, 1] += 20

    ''' move right side right + '''
    boxes[:, 2] -= 5

    ''' move bottom-right side down + '''
    # boxes[:, 3] += 10

    # Apply NMS
    # boxes_xywh = boxes.copy()
    # boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # width
    # boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # height
    # indices = nms(boxes_xywh, scores, nms_threshold)
    # boxes = boxes[indices]
    # scores = scores[indices]
    #
    # boxes = boxes.astype(int)
    #
    #
    #
    # results = []
    # for box, score in zip(boxes, scores):
    #     x1, y1, x2, y2 = box
    #     results.append((x1, y1, x2, y2, score))
    #
    # return results


    if len(scores) < 1:
        return None

    # Convert scores to a NumPy array if they aren't already
    scores = np.array(scores)

    # Find the index of the highest score
    best_idx = np.argmax(scores)

    # Extract the best box and its score
    best_box = boxes[best_idx].astype(int)
    best_score = scores[best_idx]

    result = [(best_box[0], best_box[1], best_box[2], best_box[3], best_score)]

    return result


# working
# import cv2
# import numpy as np
# import scipy.special
#
# # Define the preprocessing function with uint8 output
# def preprocess_face_landmarks(image, target_size=(224, 224), gray=True):
#     if gray:
#         # Convert to grayscale
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Resize image
#     resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
#
#     # Expand dimensions to fit the model input
#     expanded_image = np.expand_dims(resized_image, axis=[0, -1])
#
#     # return expanded_image
#     return expanded_image
#
#
# def preprocess_faces(image, input_size=(640, 640)):
#     """
#     Preprocess the image: resize to (width, height) with padding to maintain aspect ratio.
#     """
#     img_h, img_w = image.shape[:2]
#     input_w, input_h = input_size
#
#     # Calculate scaling factor and new dimensions
#     scale = min(input_w / img_w, input_h / img_h)
#     new_w, new_h = round(img_w * scale), round(img_h * scale)
#
#     # Resize image
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#
#     # Pre-allocate output image with padding value
#     pad_w, pad_h = (input_w - new_w) // 2, (input_h - new_h) // 2
#     padded_image = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
#
#     # Place resized image in the center
#     padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image
#
#     return padded_image, scale, pad_w, pad_h, img_w, img_h
#
#
# def postprocess_faces(outputs, img_w, img_h, scale, pad_w, pad_h, score_threshold=0.67, nms_threshold=0.4):
#     """
#     Postprocess the model outputs to extract face bounding boxes from the medium scale (40x40).
#     """
#
#     def decode_bboxes(bbox_pred, anchors, variances=[0.1, 0.2]):
#         """
#         Decode bounding boxes with center-offset and size scaling.
#         """
#         boxes = np.zeros_like(bbox_pred)
#
#         # Center coordinates
#         boxes[:, 0] = anchors[:, 0] + bbox_pred[:, 0] * variances[0] * anchors[:, 2]  # cx
#         boxes[:, 1] = anchors[:, 1] + bbox_pred[:, 1] * variances[0] * anchors[:, 3]  # cy
#
#         # Size
#         boxes[:, 2] = anchors[:, 2] * np.exp(bbox_pred[:, 2] * variances[1])  # w
#         boxes[:, 3] = anchors[:, 3] * np.exp(bbox_pred[:, 3] * variances[1])  # h
#
#         # Convert to corner coordinates
#         x_min = boxes[:, 0] - boxes[:, 2] / 2
#         y_min = boxes[:, 1] - boxes[:, 3] / 2
#         x_max = boxes[:, 0] + boxes[:, 2] / 2
#         y_max = boxes[:, 1] + boxes[:, 3] / 2
#
#         return np.stack([x_min, y_min, x_max, y_max], axis=1)
#
#
#
#
#     def generate_anchors(fm_sizes, input_size, steps, min_sizes):
#         anchors = []
#         for idx, fm_size in enumerate(fm_sizes):
#             scale = input_size / steps[idx]
#             fm_w, fm_h = fm_size
#             for i in range(fm_h):
#                 for j in range(fm_w):
#                     cx = (j + 0.5) / scale
#                     cy = (i + 0.5) / scale
#                     for min_size in min_sizes[idx]:
#                         w = min_size / input_size
#                         h = min_size / input_size
#                         anchors.append([cx, cy, w, h])
#         return np.array(anchors)
#
#     def nms(boxes, scores, nms_threshold):
#         indices = cv2.dnn.NMSBoxes(
#             bboxes=boxes.tolist(),
#             scores=scores.tolist(),
#             score_threshold=score_threshold,
#             nms_threshold=nms_threshold
#         )
#         return indices.flatten() if len(indices) > 0 else []
#
#
#
#     # Model parameters for the medium scale
#     input_size = 640
#     variances = [0.1, 0.2]
#     steps = [16]  # Medium scale step size
#     min_sizes = [[64, 128]]  # Anchors for the medium scale
#
#     # Feature map sizes
#     fm_sizes = [(input_size // step, input_size // step) for step in steps]
#
#     # Generate anchors
#     anchors = generate_anchors(fm_sizes, input_size, steps, min_sizes)
#
#     # Extract outputs for the medium scale
#     bbox_pred = outputs['scrfd_10g/conv54'][0]  # Shape: [40, 40, 8]
#     cls_score = outputs['scrfd_10g/conv53'][0]  # Shape: [40, 40, 2]
#
#     H, W, _ = bbox_pred.shape
#
#     # Reshape bbox_pred
#     bbox_pred = bbox_pred.reshape(-1, 4)  # [H*W*num_anchors, 4]
#
#     # Reshape cls_score
#     cls_score = cls_score.reshape(-1)  # [H*W*num_anchors,]
#
#     # Apply sigmoid activation
#     scores = scipy.special.expit(cls_score)  # [H*W*num_anchors,]
#
#     mask = scores > score_threshold
#
#     # Now indexing will work
#     bbox_pred = bbox_pred[mask]
#     scores = scores[mask]
#     anchors = anchors[mask]
#
#     # Decode bounding boxes
#     boxes = decode_bboxes(bbox_pred, anchors, variances)
#
#     # Adjust boxes to input_size coordinates
#     boxes[:, 0] *= input_size  # x_min
#     boxes[:, 1] *= input_size  # y_min
#     boxes[:, 2] *= input_size  # x_max
#     boxes[:, 3] *= input_size  # y_max
#
#     # Remove padding to get boxes in resized image coordinates
#     boxes[:, 0] -= pad_w
#     boxes[:, 1] -= pad_h
#     boxes[:, 2] -= pad_w
#     boxes[:, 3] -= pad_h
#
#     #                                                      ^
#     #                                                      |
#     boxes[:, 1] /= scale * 1.5       # Top side  (      bigger )
#
#     #                                                      ^
#     #                                                      |
#     boxes[:, 3] /= scale * 1.55     # Bottom side (      bigger )
#
#     boxes[:, 0] /= scale * 1.05      # Left side (      bigger -->  )
#     boxes[:, 2] /= scale * 1.37     # Right side  (  <-- bigger)
#
#
#     # Apply NMS
#     boxes_xywh = boxes.copy()
#     boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # width
#     boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # height
#     indices = nms(boxes_xywh, scores, nms_threshold)
#     boxes = boxes[indices]
#     scores = scores[indices]
#
#     # Convert to integers
#     boxes = boxes.astype(int)
#
#     results = []
#     for box, score in zip(boxes, scores):
#         x1, y1, x2, y2 = box
#         results.append((x1, y1, x2, y2, score))
#
#     return results









































































#-----------------------------------------------------------------------------------------------------------------------
# def preprocess_face_landmarks(image, target_size=(120, 120), gray=False):
#     """
#     Preprocess the input image for the TDDFA model.
#
#     Parameters:
#     - image: Input image in BGR format (as loaded by OpenCV).
#     - target_size: Tuple indicating the desired output size (width, height).
#     - gray: Boolean indicating whether to convert the image to grayscale.
#
#     Returns:
#     - input_tensor: A NumPy array of shape [1, 3, 120, 120] ready for the model.
#     """
#     if gray:
#         # Convert to grayscale (not recommended for this model)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # Duplicate the grayscale image across 3 channels to match the model's input
#         image = np.stack((image, image, image), axis=-1)
#     else:
#         # Convert from BGR (OpenCV format) to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Resize image to the target size
#     resized_image = cv2.resize(image, target_size)
#
#     # Normalize image to have pixel values between approximately -1 and 1
#     normalized_image = (resized_image - 127.5) / 128.0
#
#     # Expand dimensions to add batch size: [Channels, Height, Width] -> [Batch Size, Channels, Height, Width]
#     input_tensor = np.expand_dims(normalized_image, axis=0)
#
#     input_tensor = np.ascontiguousarray(input_tensor)
#
#     return input_tensor