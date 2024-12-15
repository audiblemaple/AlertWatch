import sys
import cv2

def init_cv_cap(width=640, height=640, fps=30, videopath=None):
    if videopath is None:
        if 'win' in sys.platform:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videopath)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

