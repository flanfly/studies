import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from receive import exchange_data, PairState

global_pair_state: dict[str,PairState]  = { }

async def run_exchange_stream():

    global global_pair_state
    await exchange_data(global_pair_state)


@asynccontextmanager
async def exchange_stream(app: FastAPI):
    t = asyncio.create_task(run_exchange_stream())
    yield
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=exchange_stream)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global global_pair_state

    await websocket.accept()
    try:
        while True:
            msg = {k: v.model_dump(mode='json') for k,v in global_pair_state.items()}
            await websocket.send_json(msg)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


app_dir = Path(__file__).parent / "web" / "dist"

if (app_dir / "assets").exists():
    app.mount("/assets", StaticFiles(directory=app_dir / "assets"), name="assets")


@app.get("/{full_path:path}")
async def serve_vite_app(full_path: str):
    global app_dir
    requested_file = app_dir / full_path
    if requested_file.is_file():
        return FileResponse(requested_file)

    return FileResponse(app_dir / "index.html")
