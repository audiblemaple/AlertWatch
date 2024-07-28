import socket
import cv2
import pickle
import struct

HOST = '0.0.0.0'
PORT = 8485

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn, addr = s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

while True:
    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_frame_size = data[:payload_size]
    data = data[payload_size:]
    frame_size = struct.unpack(">L", packed_frame_size)[0]

    while len(data) < frame_size:
        data += conn.recv(4096)

    frame_data = data[:frame_size]
    data = data[frame_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    while len(data) < payload_size:
        data += conn.recv(4096)

    packed_tensor_size = data[:payload_size]
    data = data[payload_size:]
    tensor_size = struct.unpack(">L", packed_tensor_size)[0]

    while len(data) < tensor_size:
        data += conn.recv(4096)

    tensor_data = data[:tensor_size]
    data = data[tensor_size:]

    tensors = pickle.loads(tensor_data, fix_imports=True, encoding="bytes")
    # print("Received tensor array:", tensors)  # Debug print

    # print("Received frame of shape: {}".format(frame.shape))
    # print("Received tensor with {} landmarks".format(len(tensors)))

    # Overlay the landmarks on the frame
    for landmarks in tensors:
        for (lx, ly) in landmarks:
            cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)

    cv2.imshow('ImageWindow', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
cv2.destroyAllWindows()
print("Server closed")


# import socket
# import cv2
# import pickle
# import numpy as np
# import struct
#
# HOST = ''
# PORT = 8485
#
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# print('Socket created')
#
# s.bind((HOST, PORT))
# print('Socket bind complete')
# s.listen(10)
# print('Socket now listening')
#
# conn, addr = s.accept()
#
# data = b""
# payload_size = struct.calcsize(">L")
# print("payload_size: {}".format(payload_size))
#
# while True:
#     while len(data) < payload_size:
#         data += conn.recv(4096)
#
#     packed_frame_size = data[:payload_size]
#     data = data[payload_size:]
#     frame_size = struct.unpack(">L", packed_frame_size)[0]
#
#     while len(data) < frame_size:
#         data += conn.recv(4096)
#
#     frame_data = data[:frame_size]
#     data = data[frame_size:]
#
#     frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
#     frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
#
#     while len(data) < payload_size:
#         data += conn.recv(4096)
#
#     packed_tensor_size = data[:payload_size]
#     data = data[payload_size:]
#     tensor_size = struct.unpack(">L", packed_tensor_size)[0]
#
#     while len(data) < tensor_size:
#         data += conn.recv(4096)
#
#     tensor_data = data[:tensor_size]
#     data = data[tensor_size:]
#
#     tensors = pickle.loads(tensor_data, fix_imports=True, encoding="bytes")
#     print("Received tensor array:", tensors)  # Debug print
#
#     print("Received frame of shape: {}".format(frame.shape))
#     print("Received tensor with {} landmarks".format(len(tensors)))
#
#     cv2.imshow('ImageWindow', frame)
#     cv2.waitKey(1)
#
# conn.close()
# print("Server closed")
