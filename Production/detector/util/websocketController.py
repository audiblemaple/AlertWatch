"""
WebSocket Client for Sending Video and Tensor Data

This module provides a WebSocket client implementation for connecting to a server,
sending video frames, tensors, and additional data. It supports reconnecting on failure.

Classes:
    - WebSocketClient: A client for managing WebSocket connections and sending data.

Functions:
    - initialize_websocket: Initializes a `WebSocketClient` instance and connects to the server.

Dependencies:
    - asyncio: For managing asynchronous WebSocket connections and tasks.
    - websockets: For WebSocket communication.
    - cv2: For video frame encoding.
    - json, base64: For preparing and encoding data to send over WebSocket.

Usage:
    Use `initialize_websocket` to set up a WebSocket connection and start sending data.

Author:
    Lior Jigalo

License:
    MIT
"""

import json
import base64

import cv2
import websockets
import asyncio

class WebSocketClient:
    """
    WebSocket client for connecting to a server and sending data.

    Attributes:
        ws_url (str): The WebSocket server URL.
        reconnect_interval (int): Time in seconds between reconnection attempts.
        websocket (websockets.WebSocketClientProtocol | None): The WebSocket connection instance.
        first_connection (bool): Indicates whether this is the first connection attempt.

    Methods:
        connect: Establishes a WebSocket connection with retries.
        run_bash_commands: Placeholder for executing bash commands during initialization.
        send_data: Sends video frames, tensors, and additional data to the server.
    """
    def __init__(self, ws_url, reconnect_interval=2):
        """
        Initializes the WebSocketClient instance.

        Args:
            ws_url (str): The WebSocket server URL.
            reconnect_interval (int): Time in seconds between reconnection attempts (default: 2).
        """
        self.ws_url = ws_url
        self.reconnect_interval = reconnect_interval
        self.websocket = None
        self.first_connection = True

    async def connect(self):
        """
        Establishes a WebSocket connection to the server.

        Retries if the connection fails, up to a maximum of 2 attempts.
        """
        i: int = 0
        while self.websocket is None:
            if i == 2:
                break
            try:
                self.websocket = await websockets.connect(self.ws_url)
                print("WebSocket connection established.")
            except Exception as e:
                print(f"Connection failed: {e}. Retrying in {self.reconnect_interval} seconds.")
                await asyncio.sleep(self.reconnect_interval)
                i = i + 1

    async def send_data(self, frame, tensors, command_outputs=None):
        """
        Sends video frames and tensor data to the WebSocket server.

        Args:
            frame (np.ndarray): The video frame to send.
            tensors (dict): A dictionary of tensor data to send.
            command_outputs (dict | None): Optional command outputs to include in the message.

        Returns:
            None
        """
        if self.websocket is None:
            return  # No WebSocket connection; skip sending data

        try:
            # Encode the frame as JPEG to reduce size
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare data to send
            data = {
                'type': "detection_feed",
                'msgData': {
                    'frame': frame_base64,
                    'face_tensors': tensors,
                    'face_landmark_tensors': None  # Update if you have landmark tensors
                }
            }
            if command_outputs is not None:
                data['msgData']['commands'] = command_outputs

            await self.websocket.send(json.dumps(data))
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidState) as e:
            # Force reconnect by setting websocket to None
            if self.websocket is not None:
                self.websocket = None
            print(f"Warning: Failed to send data via WebSocket: {e}")

async def initialize_websocket(ws_url, reconnect_interval) -> WebSocketClient:
    """
    Initializes the WebSocket client and connects to the server.

    Args:
        ws_url (str): The WebSocket server URL.
        reconnect_interval (int): Time in seconds between reconnection attempts.

    Returns:
        WebSocketClient: An instance of the initialized WebSocket client.
    """
    ws_client = WebSocketClient(ws_url, reconnect_interval)
    await ws_client.connect()
    return ws_client



# # socketUtil/websocketController.py
#
# import json
# import base64
# import cv2
# import websockets
# import asyncio
#
# class WebSocketClient:
#     def __init__(self, ws_url, reconnect_interval=2):
#         self.ws_url = ws_url
#         self.reconnect_interval = reconnect_interval
#         self.websocket = None
#         self.first_connection = True
#
#     async def connect(self):
#         while self.websocket is None:
#             try:
#                 self.websocket = await websockets.connect(self.ws_url)
#                 print("WebSocket connection established.")
#             except Exception as e:
#                 print(f"Connection failed: {e}. Retrying in {self.reconnect_interval} seconds.")
#                 await asyncio.sleep(self.reconnect_interval)
#
#     async def run_bash_commands(self):
#         """
#         Placeholder for running initial bash commands.
#         Replace this method's content with actual bash command executions if needed.
#         """
#         # Example: return {"command_output": "success"}
#         return {"command_output": "initial_commands_executed"}
#
#     async def send_data(self, frame, tensors, command_outputs=None):
#         if self.websocket is None:
#             return  # No WebSocket connection; skip sending data
#
#         try:
#             # Encode the frame as JPEG to reduce size
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame_base64 = base64.b64encode(buffer).decode('utf-8')
#
#             # Prepare data to send
#             data = {
#                 'type': "detection_feed",
#                 'msgData': {
#                     'frame': frame_base64,
#                     'face_tensors': tensors,
#                     'face_landmark_tensors': None  # Update if you have landmark tensors
#                 }
#             }
#             if command_outputs is not None:
#                 data['msgData']['commands'] = command_outputs
#
#             await self.websocket.send(json.dumps(data))
#         except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidState) as e:
#             # Force reconnect by setting websocket to None
#             if self.websocket is not None:
#                 self.websocket = None
#             print(f"Warning: Failed to send data via WebSocket: {e}")
#
#
# async def initialize_websocket(WS_URL, RECONNECT_INTERVAL) -> WebSocketClient:
#     """
#     Initializes the WebSocket client.
#
#     Returns:
#         WebSocketClient: An instance of the WebSocketClient.
#     """
#     ws_client = WebSocketClient(WS_URL, RECONNECT_INTERVAL)
#     await ws_client.run_bash_commands()  # Run initial bash commands
#     await ws_client.connect()
#     return ws_client