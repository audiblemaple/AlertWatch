import cv2

def draw_bounding_box(frame, score, p1, p2, color=(0, 255, 0), size=2):
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Detection confidence: {score:.2f}%",
        (p1[0], p1[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )