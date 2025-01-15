import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import numpy as np

from . import (
    WebSocketClient,
    initialize_websocket
)


@pytest.mark.asyncio
async def test_websocket_client_init():
    """
    Test the initialization of the WebSocketClient.
    """
    ws_url = "ws://localhost:8765"
    reconnect_interval = 5

    client = WebSocketClient(ws_url, reconnect_interval)

    assert client.ws_url == ws_url
    assert client.reconnect_interval == reconnect_interval
    assert client.websocket is None
    assert client.first_connection is True


@pytest.mark.asyncio
async def test_websocket_client_connect_success():
    """
    Test that connect() successfully connects on the first attempt.
    """
    ws_url = "ws://localhost:8765"
    client = WebSocketClient(ws_url)

    # Mock websockets.connect to simulate a successful connection
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.return_value = AsyncMock()  # Simulate a valid websocket

        await client.connect()

        # Assert that websockets.connect was called exactly once
        mock_connect.assert_called_once_with(ws_url)
        # The client.websocket attribute should not be None now
        assert client.websocket is not None


@pytest.mark.asyncio
async def test_websocket_client_connect_failure_retries():
    """
    Test that connect() retries connection up to 2 times on failure.
    """
    ws_url = "ws://localhost:8765"
    client = WebSocketClient(ws_url, reconnect_interval=0)  # 0 for faster tests

    # Mock websockets.connect to always raise an exception
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        mock_connect.side_effect = Exception("Connection failed")

        await client.connect()

        # websockets.connect should be called up to 2 times
        assert mock_connect.call_count == 2
        # After failing, client.websocket should remain None
        assert client.websocket is None


@pytest.mark.asyncio
async def test_websocket_client_send_data_no_connection():
    """
    Test that send_data() does nothing when there is no active WebSocket connection.
    """
    client = WebSocketClient("ws://localhost:8765")

    # frame can be anything; we're not actually encoding in this test
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_tensors = {"tensor_key": "tensor_value"}

    # send_data should return immediately if websocket is None
    await client.send_data(frame=fake_frame, tensors=fake_tensors)
    # No errors should occur, and there's nothing to assert here other than it doesn't raise.


@pytest.mark.asyncio
async def test_websocket_client_send_data_success():
    """
    Test that send_data() sends the correct JSON data over the WebSocket.
    """
    client = WebSocketClient("ws://localhost:8765")
    
    # Mock an active websocket
    mock_websocket = AsyncMock()
    client.websocket = mock_websocket

    # Create a fake frame
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_tensors = {"tensor_key": "tensor_value"}
    command_outputs = {"command": "output"}

    with patch("cv2.imencode") as mock_imencode:
        # Simulate cv2.imencode returning success and a buffer
        mock_imencode.return_value = (True, b"fake_jpeg_buffer")

        await client.send_data(frame=fake_frame, tensors=fake_tensors, command_outputs=command_outputs)

        # Check cv2.imencode was called with '.jpg'
        mock_imencode.assert_called_with('.jpg', fake_frame)

        # Extract what was sent
        sent_data = mock_websocket.send.call_args[0][0]

        # Verify the JSON structure
        import json
        data_dict = json.loads(sent_data)

        assert data_dict["type"] == "detection_feed"
        assert "frame" in data_dict["msgData"]
        # Since it's base64-encoded, it should be a string
        assert isinstance(data_dict["msgData"]["frame"], str)
        assert data_dict["msgData"]["face_tensors"] == fake_tensors
        assert data_dict["msgData"]["commands"] == command_outputs


@pytest.mark.asyncio
async def test_initialize_websocket():
    """
    Test that initialize_websocket creates a WebSocketClient and calls connect().
    """
    ws_url = "ws://localhost:8765"
    reconnect_interval = 2

    with patch.object(WebSocketClient, 'connect', new_callable=AsyncMock) as mock_connect:
        ws_client = await initialize_websocket(ws_url, reconnect_interval)

        # connect() should be called exactly once
        mock_connect.assert_awaited_once()
        # ws_client should be an instance of WebSocketClient
        assert isinstance(ws_client, WebSocketClient)
        assert ws_client.ws_url == ws_url
        assert ws_client.reconnect_interval == reconnect_interval
