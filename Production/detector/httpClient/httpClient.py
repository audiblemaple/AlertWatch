# httpClient.py

import aiohttp
import asyncio
import json

async def send_drowsiness_alert(url: str, payload: dict, headers: dict = None, retries: int = 3, backoff_factor: float = 0.5):
    """
    Sends an HTTP POST request to the specified URL with the given payload and headers.
    Retries the request on failure based on the specified retry count and backoff factor.

    Args:
        url (str): The endpoint URL to send the request to.
        payload (dict): The JSON payload to include in the request body.
        headers (dict, optional): Additional headers to include in the request.
        retries (int): Number of retry attempts on failure.
        backoff_factor (float): Factor by which the delay increases after each retry.

    Returns:
        bool: True if the request was successful (status code 200-299), False otherwise.
    """
    attempt = 0
    print("sending alert")
    while attempt < retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if 200 <= response.status < 300:
                        print(f"[HTTP] Successfully sent drowsiness alert. Status Code: {response.status}")
                        return True
                    else:
                        print(f"[HTTP] Failed to send drowsiness alert. Status Code: {response.status}")
        except aiohttp.ClientError as e:
            print(f"[HTTP] ClientError: {e}")
        except asyncio.TimeoutError:
            print("[HTTP] Request timed out.")
        except Exception as e:
            print(f"[HTTP] Unexpected error: {e}")

        attempt += 1
        delay = backoff_factor * (2 ** (attempt - 1))
        print(f"[HTTP] Retrying in {delay} seconds... (Attempt {attempt}/{retries})")
        await asyncio.sleep(delay)

    print("[HTTP] All retry attempts failed. Could not send drowsiness alert.")
    return False
