import asyncio
import websockets
import json
import base64
import cv2

class WebSocketClient:
    def __init__(self, ws_url, reconnect_interval=3):
        self.ws_url = ws_url
        self.reconnect_interval = reconnect_interval
        self.websocket = None
        self.first_connection = True

    async def connect(self):
        """Try to connect to the WebSocket server and return the connection if successful."""
        while True:
            try:
                self.websocket = await websockets.connect(self.ws_url)
                print("Connected to WebSocket server.")
                return self.websocket
            except Exception as e:
                print(f"Warning: Unable to connect to WebSocket server: {e}")
                print(f"Reattempting connection in {self.reconnect_interval} seconds...")
                await asyncio.sleep(self.reconnect_interval)

    async def send_data(self, frame, tensors, command_outputs=None):
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
                    'face_landmark_tensors': None
                }
            }
            if command_outputs is not None:
                data['msgData']['commands'] = command_outputs

            await self.websocket.send(json.dumps(data))
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidState) as e:
            # Force reconnect by setting websocket to None
            if self.websocket is not None:
                await self.websocket.close()
                self.websocket = None
            print(f"Warning: Failed to send data via WebSocket: {e}")

    async def ensure_connection(self):
        """Ensure that the websocket connection is active."""
        if self.websocket is None or self.websocket.closed:
            await self.connect()
            self.first_connection = True

    def reset_first_connection(self):
        self.first_connection = False

    def is_first_connection(self):
        return self.first_connection
