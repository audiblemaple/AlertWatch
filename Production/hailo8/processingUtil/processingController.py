import cv2
import numpy as np
import scipy.special

# def preprocess_faces(image, input_size=(640, 640)):
#     """
#     Preprocess the image: resize to (width, height) with padding to maintain aspect ratio.
#     """
#     img_h, img_w = image.shape[:2]
#     input_w, input_h = input_size
#     scale = min(input_w / img_w, input_h / img_h)
#     new_w, new_h = int(img_w * scale), int(img_h * scale)
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#     padded_image = np.full((input_h, input_w, 3), 127, dtype=np.uint8)
#     pad_w, pad_h = (input_w - new_w) // 2, (input_h - new_h) // 2
#     padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w, :] = resized_image
#     return padded_image, scale, pad_w, pad_h, img_w, img_h  # Return original image dimensions

def preprocess_faces(image, input_size=(640, 640)):
    """
    Preprocess the image: resize to (width, height) with padding to maintain aspect ratio.
    """
    img_h, img_w = image.shape[:2]
    input_w, input_h = input_size

    # Calculate scaling factor and new dimensions
    scale = min(input_w / img_w, input_h / img_h)
    new_w, new_h = round(img_w * scale), round(img_h * scale)

    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pre-allocate output image with padding value
    pad_w, pad_h = (input_w - new_w) // 2, (input_h - new_h) // 2
    padded_image = np.full((input_h, input_w, 3), 127, dtype=np.uint8)

    # Place resized image in the center
    padded_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_image

    return padded_image, scale, pad_w, pad_h, img_w, img_h


def postprocess_faces(outputs, img_w, img_h, scale, pad_w, pad_h, score_threshold=0.60, nms_threshold=0.4):
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

    # Feature map sizes
    fm_sizes = [(input_size // step, input_size // step) for step in steps]

    # Generate anchors
    anchors = generate_anchors(fm_sizes, input_size, steps, min_sizes)

    # Extract outputs for the medium scale
    bbox_pred = outputs['scrfd_10g/conv54'][0]  # Shape: [40, 40, 8]
    cls_score = outputs['scrfd_10g/conv53'][0]  # Shape: [40, 40, 2]

    H, W, _ = bbox_pred.shape

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

    boxes[:, 0] /= scale * 1      # Left side   (bigger -> bigger)
    boxes[:, 1] /= scale * 1        # Top side    (bigger -> bigger)
    boxes[:, 2] /= scale * 1.25     # Right side  (bigger -> smaller)
    boxes[:, 3] /= scale * 1.5     # Bottom side (bigger -> smaller)

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


# Define the preprocessing function with uint8 output
def preprocess_face_landmarks(image, target_size=(224, 224), gray=True):
    if gray:
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Expand dimensions to fit the model input
    expanded_image = np.expand_dims(resized_image, axis=[0, -1])

    # return expanded_image
    return expanded_image







































































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