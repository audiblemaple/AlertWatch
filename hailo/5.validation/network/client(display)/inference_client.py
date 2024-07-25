import cv2
import numpy as np
import socket
import struct

def main():
    server_address = '0.0.0.0'  # Listen on all network interfaces
    server_port = 60000

    # Set up server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_address, server_port))
    server_socket.listen(5)
    print(f"Listening on {server_address}:{server_port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        data = b""
        payload_size = struct.calcsize("L")

        while True:
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet

            if len(data) < payload_size:
                print("Incomplete packet received")
                break

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            print(f"Message size: {msg_size}")

            while len(data) < msg_size:
                print("message size: ", msg_size)
                print("data length: ", str(len(data)))
                packet = client_socket.recv(4096)
                if not packet:
                    break
                data += packet

            if len(data) < msg_size:
                print("Incomplete data received")
                break

            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Decode the frame
            np_frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            if frame is None or frame.size == 0:
                print("Received empty frame")
                continue

            # Display the frame
            cv2.imshow('Received Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



# import cv2
# import numpy as np
# import socket
# import struct
#
# def main():
#     server_address = '0.0.0.0'  # Listen on all network interfaces
#     server_port = 60000
#
#     # Set up server socket
#     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_socket.bind((server_address, server_port))
#     server_socket.listen(5)
#     print(f"Listening on {server_address}:{server_port}")
#
#     while True:
#         client_socket, addr = server_socket.accept()
#         print(f"Connection from {addr}")
#
#         data = b""
#         payload_size = struct.calcsize("L")
#
#         while True:
#             while len(data) < payload_size:
#                 packet = client_socket.recv(4096)
#                 if not packet:
#                     break
#                 data += packet
#
#             if len(data) < payload_size:
#                 print("Incomplete packet received")
#                 break
#
#             packed_msg_size = data[:payload_size]
#             data = data[payload_size:]
#             msg_size = struct.unpack("L", packed_msg_size)[0]
#             print(f"Message size: {msg_size}")
#
#             while len(data) < msg_size:
#                 packet = client_socket.recv(4096)
#                 if not packet:
#                     break
#                 data += packet
#
#             if len(data) < msg_size:
#                 print("Incomplete data received")
#                 break
#
#             frame_data = data[:msg_size]
#             data = data[msg_size:]
#
#             # Decode the frame
#             np_frame = np.frombuffer(frame_data, dtype=np.uint8)
#             frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
#
#             if frame is None or frame.size == 0:
#                 print("Received empty frame")
#                 continue
#
#             # Display the frame
#             cv2.imshow('Received Frame', frame)
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         client_socket.close()
#         cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
