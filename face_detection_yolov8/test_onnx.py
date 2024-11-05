import cv2
import numpy as np
import onnxruntime as ort


def generate_anchors(feature_map_size, stride):
    anchors = []
    for y in range(feature_map_size[1]):  # Height dimension
        for x in range(feature_map_size[0]):  # Width dimension
            anchors.append([(x + 0.5) * stride, (y + 0.5) * stride])
    return np.array(anchors)


def decode_predictions(scores, bboxes, threshold):
    print("scores.shape: ", scores.shape)
    print("bboxes.shape: ", bboxes.shape)
    print("threshold: ", threshold)

    # Remove batch dimension
    scores = scores[0]  # Shape: (N, num_classes) or (N, 1)
    bboxes = bboxes[0]  # Shape: (N, 4)

    print("scores.shape after removing batch dimension:", scores.shape)
    print("bboxes.shape after removing batch dimension:", bboxes.shape)

    # Handle different shapes of scores
    if scores.shape[1] > 1:
        face_scores = scores[:, 1]  # Assuming face class at index 1
    else:
        face_scores = scores.squeeze()

    # Apply confidence threshold
    indices = np.where(face_scores > threshold)[0]
    if len(indices) == 0:
        return np.array([]), np.array([])

    print("indices: ", indices)
    scores = face_scores[indices]
    bboxes = bboxes[indices]

    # Ensure proper ordering of coordinates: (x1, y1) should be the top-left corner
    # and (x2, y2) should be the bottom-right corner
    bboxes[:, [0, 2]] = np.sort(bboxes[:, [0, 2]], axis=1)  # Sort x1, x2
    bboxes[:, [1, 3]] = np.sort(bboxes[:, [1, 3]], axis=1)  # Sort y1, y2

    return bboxes, scores


def post_process(outputs, img_shape, conf_threshold=0.5, nms_threshold=0.4):
    stride = 32
    feature_map_size = (img_shape[1] // stride, img_shape[0] // stride)  # (width, height)
    print('Feature map size:', feature_map_size)

    scores = outputs['score_32']
    bboxes = outputs['bbox_32']

    anchors = generate_anchors(feature_map_size, stride)
    print('Anchors shape:', anchors.shape)

    bboxes_decoded, scores_decoded = decode_predictions(
        scores, bboxes, conf_threshold
    )

    if bboxes_decoded.size == 0:
        return np.array([]), np.array([])

    # Convert bboxes to x, y, w, h for NMS
    bboxes_xywh = np.zeros_like(bboxes_decoded)
    bboxes_xywh[:, 0] = bboxes_decoded[:, 0]
    bboxes_xywh[:, 1] = bboxes_decoded[:, 1]
    bboxes_xywh[:, 2] = bboxes_decoded[:, 2] - bboxes_decoded[:, 0]
    bboxes_xywh[:, 3] = bboxes_decoded[:, 3] - bboxes_decoded[:, 1]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes_xywh.tolist(), scores_decoded.tolist(), conf_threshold, nms_threshold
    )
    if len(indices) == 0:
        return np.array([]), np.array([])

    indices = indices.flatten()

    final_bboxes = bboxes_decoded[indices]
    final_scores = scores_decoded[indices]

    return final_bboxes, final_scores


def adjust_scale(bboxes, scale_x, scale_y):
    # Adjust bounding boxes based on the scaling factors
    bboxes[:, [0, 2]] /= scale_x  # Scale x-coordinates back to original size
    bboxes[:, [1, 3]] /= scale_y  # Scale y-coordinates back to original size
    return bboxes


# Load the ONNX model
model_path = 'scrfd_10g_bnkps.onnx'
session = ort.InferenceSession(model_path)

# Get model input and output details
input_name = session.get_inputs()[0].name
print(f'input_name: {input_name}')
input_shape = session.get_inputs()[0].shape
print(f'input_shape: {input_shape}')
output_names = [output.name for output in session.get_outputs()]
print(f'output_names: {output_names}')


# Initialize video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_height, orig_width = frame.shape[:2]
    img = cv2.resize(frame, (640, 640))
    scale_x = 640 / orig_width
    scale_y = 640 / orig_height

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
    img_expanded = np.expand_dims(img_transposed, axis=0)

    # Run inference
    ort_inputs = {input_name: img_expanded}
    ort_outputs = session.run(['score_32', 'bbox_32'], ort_inputs)

    # Prepare outputs dictionary
    outputs_dict = {'score_32': ort_outputs[0], 'bbox_32': ort_outputs[1]}

    # Print shapes for debugging
    print('scores shape:', outputs_dict['score_32'].shape)
    print('bboxes shape:', outputs_dict['bbox_32'].shape)

    # Post-processing
    final_bboxes, final_scores = post_process(
        outputs_dict, img_shape=(640, 640), conf_threshold=0.6, nms_threshold=0.4
    )

    # Adjust coordinates back to original image scale
    final_bboxes = adjust_scale(final_bboxes, scale_x, scale_y)

    # Draw bounding boxes
    # Draw bounding boxes
    for bbox, score in zip(final_bboxes, final_scores):
        x1, y1, x2, y2 = bbox.astype(int)
        print(f"(x1, y1), (x2, y2): ({x1}, {y1}), ({x2}, {y2})")

        # Draw rectangle using the top-left and bottom-right points
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put confidence score on top of the bounding box
        cv2.putText(frame, f'{score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('SCRFD Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
