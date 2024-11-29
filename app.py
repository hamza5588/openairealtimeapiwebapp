from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import os
import json
from typing import Dict
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from pynput import keyboard
from openai_realtime_client import RealtimeClient, AudioHandler, InputHandler, TurnDetectionMode
from llama_index.core.tools import FunctionTool

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active client connections
clients: Dict[str, RealtimeClient] = {}
audio_handlers: Dict[str, AudioHandler] = {}

def get_phone_number(name: str) -> str:
    """Get my phone number."""
    if name == "Jerry":
        return "1234567890"
    elif name == "Logan":
        return "0987654321"
    else:
        return "Unknown"

tools = [FunctionTool.from_defaults(fn=get_phone_number)]

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    print(f"New WebSocket connection established for client: {client_id}")
    
    try:
        # Initialize handlers
        audio_handler = AudioHandler()
        input_handler = InputHandler()
        input_handler.loop = asyncio.get_running_loop()
        
        # Initialize client
        client = RealtimeClient(
            api_key=os.environ.get("OPENAI_API_KEY"),
            on_text_delta=lambda text: asyncio.create_task(websocket.send_text(json.dumps({"type": "text", "content": text}))),
            on_audio_delta=lambda audio: audio_handler.play_audio(audio),
            on_interrupt=lambda: audio_handler.stop_playback_immediately(),
            turn_detection_mode=TurnDetectionMode.SERVER_VAD,
            tools=tools,
        )
        
        # Store client and handler
        clients[client_id] = client
        audio_handlers[client_id] = audio_handler
        
        print(f"Connecting to OpenAI for client: {client_id}")
        await client.connect()
        print(f"Connected to OpenAI successfully for client: {client_id}")
        
        message_handler = asyncio.create_task(client.handle_messages())
        streaming_task = asyncio.create_task(audio_handler.start_streaming(client))
        
        # Listen for WebSocket messages
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                print(f"Received message from client {client_id}: {data['type']}")
                
                if data["type"] == "audio":
                    # Handle incoming audio data
                    audio_bytes = bytes(data["content"])
                    print(f"Received audio chunk of size: {len(audio_bytes)} bytes")
                    await client.send_audio(audio_bytes)
                elif data["type"] == "command" and data["content"] == "stop":
                    print(f"Received stop command from client: {client_id}")
                    break
                    
            except json.JSONDecodeError as e:
                print(f"Error decoding message from client {client_id}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in WebSocket connection for client {client_id}: {e}")
        await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
    finally:
        print(f"Cleaning up connection for client: {client_id}")
        # Cleanup
        if client_id in clients:
            audio_handlers[client_id].stop_streaming()
            audio_handlers[client_id].cleanup()
            await clients[client_id].close()
            del clients[client_id]
            del audio_handlers[client_id]

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("static/index.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

if __name__ == "__main__":
    print("Starting FastAPI server...")
    if not os.environ.get("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set!")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")