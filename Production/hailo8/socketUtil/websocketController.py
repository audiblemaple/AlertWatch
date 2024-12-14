# socketUtil/websocketController.py

import json
import base64
import cv2
import websockets
import asyncio

class WebSocketClient:
    def __init__(self, ws_url, reconnect_interval=2):
        self.ws_url = ws_url
        self.reconnect_interval = reconnect_interval
        self.websocket = None
        self.first_connection = True

    async def connect(self):
        while self.websocket is None:
            try:
                self.websocket = await websockets.connect(self.ws_url)
                print("WebSocket connection established.")
            except Exception as e:
                print(f"Connection failed: {e}. Retrying in {self.reconnect_interval} seconds.")
                await asyncio.sleep(self.reconnect_interval)

    async def run_bash_commands(self):
        """
        Placeholder for running initial bash commands.
        Replace this method's content with actual bash command executions if needed.
        """
        # Example: return {"command_output": "success"}
        return {"command_output": "initial_commands_executed"}

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
