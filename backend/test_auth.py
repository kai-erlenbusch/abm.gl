import asyncio
import websockets

async def test_unauthorized():
    uri = "ws://localhost:8000/ws"
    print("Testing unauthorized connection...")
    try:
        async with websockets.connect(uri) as websocket:
            print("ERROR: Connection succeeded unexpectedly without a token.")
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"SUCCESS: Connection rejected with HTTP status {e.status_code}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"SUCCESS: Connection closed with WS code {e.code}")
    except Exception as e:
        print(f"ERROR or server not running: {e}")

async def test_authorized():
    uri = "ws://localhost:8000/ws?token=dev_secret_token"
    print("Testing authorized connection...")
    try:
        async with websockets.connect(uri) as websocket:
            print("SUCCESS: Connection succeeded with valid token.")
    except Exception as e:
        print(f"ERROR: Failed to connect with token: {e}")

async def main():
    await test_unauthorized()
    print("-" * 30)
    await test_authorized()

if __name__ == "__main__":
    asyncio.run(main())
