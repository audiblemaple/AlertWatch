import cv2
import numpy as np

# # Define the preprocessing function with uint8 output
def preprocess_face_landmarks(image, target_size=(224, 224), gray=True):
    if gray:
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

    # Expand dimensions to fit the model input
    # expanded_image = np.expand_dims(resized_image, axis=[0, -1])

    # return expanded_image
    return resized_image[np.newaxis, ..., np.newaxis]

# Define the preprocessing function with uint8 output
def preprocess_faces(image, input_size=(640, 640)):
    img_h, img_w = image.shape[:2]
    input_w, input_h = input_size

    # Calculate scaling factor and dimensions
    scale = min(input_w / img_w, input_h / img_h)
    new_size = (round(img_w * scale), round(img_h * scale))

    # Resize image efficiently
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)

    # Create padded output using numpy
    pad_w = (input_w - new_size[0]) // 2
    pad_h = (input_h - new_size[1]) // 2
    padded_image = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
    padded_image[pad_h:pad_h + new_size[1], pad_w:pad_w + new_size[0]] = resized_image

    return padded_image, scale, pad_w, pad_h, img_w, img_h

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_anchors(fm_sizes, input_size, steps, min_sizes):
    """
    Generate anchors for all feature map sizes and scales.
    Args:
        fm_sizes (list of tuples): Feature map sizes [(W1, H1), (W2, H2), (W3, H3)]
        input_size (int): Input image size (e.g., 640)
        steps (list of int): Step sizes for each scale
        min_sizes (list of list of int): Min sizes for anchors per scale
    Returns:
        np.ndarray: Generated anchors of shape [total_anchors, 4]
    """
    anchors = []
    for idx, fm_size in enumerate(fm_sizes):
        fm_w, fm_h = fm_size
        step = steps[idx]
        min_size = min_sizes[idx]
        for i in range(fm_h):
            for j in range(fm_w):
                cx = (j + 0.5) * step / input_size
                cy = (i + 0.5) * step / input_size
                for ms in min_size:
                    w = ms / input_size
                    h = ms / input_size
                    anchors.append([cx, cy, w, h])
    return np.array(anchors, dtype=np.float32)

def decode_bboxes(bbox_pred, anchors, variances=[0.1, 0.2]):
    """
    Decode bounding boxes from predictions and anchors.
    Args:
        bbox_pred (np.ndarray): Box predictions of shape [N, 4]
        anchors (np.ndarray): Anchors of shape [N, 4]
        variances (list of float): Variance values
    Returns:
        np.ndarray: Decoded boxes in [x_min, y_min, x_max, y_max] format
    """
    boxes = np.zeros_like(bbox_pred, dtype=np.float32)
    # Decode center-size
    boxes[:, 0] = anchors[:, 0] + bbox_pred[:, 0] * variances[0] * anchors[:, 2]  # cx
    boxes[:, 1] = anchors[:, 1] + bbox_pred[:, 1] * variances[0] * anchors[:, 3]  # cy
    boxes[:, 2] = anchors[:, 2] * np.exp(bbox_pred[:, 2] * variances[1])         # w
    boxes[:, 3] = anchors[:, 3] * np.exp(bbox_pred[:, 3] * variances[1])         # h

    # Convert center-size to corner coordinates
    x_min = boxes[:, 0] - boxes[:, 2] / 2
    y_min = boxes[:, 1] - boxes[:, 3] / 2
    x_max = boxes[:, 0] + boxes[:, 2] / 2
    y_max = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x_min, y_min, x_max, y_max], axis=1)

def nms(boxes, scores, score_threshold, nms_threshold):
    """
    Perform Non-Maximum Suppression using OpenCV.
    Args:
        boxes (np.ndarray): Bounding boxes in [x_min, y_min, x_max, y_max] format
        scores (np.ndarray): Confidence scores
        score_threshold (float): Minimum score threshold
        nms_threshold (float): NMS IoU threshold
    Returns:
        list: Indices of boxes to keep
    """
    # Convert boxes to [x, y, w, h] as required by cv2.dnn.NMSBoxes
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

    # cv2.dnn.NMSBoxes expects boxes as a list of [x, y, w, h]
    # and returns indices as a list of lists, so we need to flatten it
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh.tolist(),
        scores=scores.tolist(),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold
    )
    if len(indices) > 0:
        return indices.flatten()
    return []

def postprocess_faces(outputs, img_w, img_h, scale, pad_w, pad_h,
                      input_size=640, variances=[0.1, 0.2],
                      steps=[8, 16, 32],
                      min_sizes=[[16, 32], [64, 128], [256, 512]],
                      score_threshold=0.67, nms_threshold=0.99):
    """
    Postprocess the model outputs to extract face bounding boxes from all scales.
    Args:
        outputs (dict): Dictionary of model outputs keyed by layer names.
        img_w (int): Original image width.
        img_h (int): Original image height.
        scale (float): Scaling factor used during preprocessing.
        pad_w (int): Padding width added during preprocessing.
        pad_h (int): Padding height added during preprocessing.
        input_size (int): Size of the input image to the model.
        variances (list of float): Variance values for decoding.
        steps (list of int): Step sizes for each scale.
        min_sizes (list of list of int): Min sizes for anchors per scale.
        score_threshold (float): Confidence score threshold.
        nms_threshold (float): NMS IoU threshold.
    Returns:
        list: List of detected faces with bounding boxes and scores.
    """
    # Layer names for SCRFD-10G variant
    BOXES_LAYERS = ["scrfd_10g/conv57", "scrfd_10g/conv50", "scrfd_10g/conv42"]     # 20x20x8, 40x40x8, 80x80x8
    CLASSES_LAYERS = ["scrfd_10g/conv56", "scrfd_10g/conv49", "scrfd_10g/conv41"]  # 20x20x2, 40x40x2, 80x80x2

    # Feature map sizes for each scale
    fm_sizes = [(input_size // step, input_size // step) for step in steps]  # [(80,80), (40,40), (20,20)]

    # Generate anchors for all scales
    all_anchors = generate_anchors(fm_sizes, input_size, steps, min_sizes)  # [total_anchors, 4]

    # Initialize lists to collect boxes and scores from all scales
    all_boxes = []
    all_scores = []

    # Calculate the starting index for anchors in each scale
    scale_offsets = []
    current_offset = 0
    for idx in range(len(steps)):
        num_anchors = len(min_sizes[idx])  # 2 anchors per location
        fm_w, fm_h = fm_sizes[idx]
        scale_offsets.append(current_offset)
        current_offset += fm_w * fm_h * num_anchors  # Total anchors up to this scale

    for idx, (box_layer, cls_layer) in enumerate(zip(BOXES_LAYERS, CLASSES_LAYERS)):
        fm_w, fm_h = fm_sizes[idx]
        step = steps[idx]
        min_size = min_sizes[idx]

        # Number of anchors per location for this scale
        C_boxes = outputs[box_layer].shape[-1]
        num_anchors = C_boxes // 4  # 8 /4 =2
        C_classes = outputs[cls_layer].shape[-1]
        expected_classes_channels = num_anchors  # 1 class score per anchor

        if C_classes != expected_classes_channels:
            raise ValueError(f"Mismatch in number of anchors for scale {idx}: classes layer has {C_classes} channels but expected {expected_classes_channels} (num_anchors={num_anchors})")

        # Extract and reshape box predictions
        bbox_pred = outputs[box_layer][0].astype(np.float32)  # Remove batch dimension: [H, W, 8]
        H, W, C = bbox_pred.shape
        bbox_pred = bbox_pred.reshape(-1, 4)  # [H * W * 2, 4]

        # Extract and reshape class predictions
        cls_pred = outputs[cls_layer][0].astype(np.float32)  # Remove batch dimension: [H, W, 2]
        cls_pred = cls_pred.reshape(-1)  # [H * W * 2]

        # Apply sigmoid to class scores
        scores = sigmoid(cls_pred)  # [H * W * 2]

        # Filter out boxes with low scores
        mask = scores > score_threshold  # [H * W * 2]
        if not np.any(mask):
            continue  # No detections in this scale

        bbox_pred = bbox_pred[mask]  # [num_keep, 4]
        scores = scores[mask]        # [num_keep]

        # Calculate indices in all_anchors
        # Each scale has fm_w * fm_h * num_anchors anchors
        scale_offset = scale_offsets[idx]
        anchor_indices = scale_offset + np.nonzero(mask)[0]
        anchors = all_anchors[anchor_indices]  # [num_keep, 4]

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

        # Adjust for scale to map back to original image size
        boxes /= scale

        # Clip boxes to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)

        all_boxes.append(boxes)
        all_scores.append(scores)

    if len(all_boxes) == 0:
        return []  # No detections

    # Concatenate all boxes and scores from all scales
    all_boxes = np.concatenate(all_boxes, axis=0)  # [total_detections, 4]
    all_scores = np.concatenate(all_scores, axis=0)  # [total_detections]

    # Perform Non-Maximum Suppression (NMS)
    keep_indices = nms(all_boxes, all_scores, score_threshold, nms_threshold)
    if len(keep_indices) == 0:
        return []

    boxes_final = all_boxes[keep_indices].astype(int)
    scores_final = all_scores[keep_indices]

    # Prepare the final list of detections
    results = []
    for box, score in zip(boxes_final, scores_final):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        results.append({
            "label": "face",
            "confidence": float(score),
            "bbox": [int(x1), int(y1), int(w), int(h)]
        })

    # return [results[0]['bbox'][0], results[0]['bbox'][1], results[0]['bbox'][2], results[0]['bbox'][0], results[0]['confidence']]

    return results




# import cv2
# import numpy as np
#
# # Define the preprocessing function with uint8 output
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
#
# def preprocess_faces(image, input_size=(640, 640)):
#     img_h, img_w = image.shape[:2]
#     input_w, input_h = input_size
#
#     # Calculate scaling factor and dimensions
#     scale = min(input_w / img_w, input_h / img_h)
#     new_size = (round(img_w * scale), round(img_h * scale))
#
#     # Resize image efficiently
#     resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
#
#     # Create padded output using numpy
#     pad_w, pad_h = (input_w - new_size[0]) // 2, (input_h - new_size[1]) // 2
#     padded_image = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
#     padded_image[pad_h:pad_h + new_size[1], pad_w:pad_w + new_size[0]] = resized_image
#
#     return padded_image, scale, pad_w, pad_h, img_w, img_h
#
#
# def postprocess_faces(outputs, img_w, img_h, scale, pad_w, pad_h, score_threshold=0.67, nms_threshold=0.99):
#     """
#     Postprocess the model outputs to extract face bounding boxes from the medium scale (40x40).
#     """
#
#     def decode_bboxes(bbox_pred, anchors, variances=[0.1, 0.2]):
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
#     def compute_iou(box, other_boxes):
#         x1 = np.maximum(box[0], other_boxes[:, 0])
#         y1 = np.maximum(box[1], other_boxes[:, 1])
#         x2 = np.minimum(box[2], other_boxes[:, 2])
#         y2 = np.minimum(box[3], other_boxes[:, 3])
#
#         inter_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
#         box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
#         other_areas = (other_boxes[:, 2] - other_boxes[:, 0] + 1) * (other_boxes[:, 3] - other_boxes[:, 1] + 1)
#
#         union_area = box_area + other_areas - inter_area
#         iou = inter_area / union_area
#         return iou
#
#
#     def sigmoid(x):
#         """Replace scipy.special.expit with NumPy implementation."""
#         return 1 / (1 + np.exp(-x))
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
#     bbox_pred = outputs['scrfd_10g/conv50'][0]  # Shape: [40, 40, 8]
#     cls_score = outputs['scrfd_10g/conv49'][0]  # Shape: [40, 40, 2]
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
#     scores = sigmoid(cls_score)  # [H*W*num_anchors,]
#
#     mask = scores > score_threshold
#
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
#     # Adjust for scale
#     boxes[:, 1] /= scale * 1.5  # Top side
#     boxes[:, 3] /= scale * 1.55  # Bottom side
#     boxes[:, 0] /= scale * 1.05  # Left side
#     boxes[:, 2] /= scale * 1.37  # Right side
#
#     # Apply NMS
#     boxes_xywh = boxes.copy()
#     boxes_xywh[:, 2] -= boxes_xywh[:, 0]  # width
#     boxes_xywh[:, 3] -= boxes_xywh[:, 1]  # height
#     indices = nms(boxes_xywh, scores, nms_threshold)
#     boxes = boxes[indices]
#     scores = scores[indices]
#
#     boxes = boxes.astype(int)
#
#     results = []
#     for box, score in zip(boxes, scores):
#         x1, y1, x2, y2 = box
#         results.append((x1, y1, x2, y2, score))
#
#     return results
#
#
