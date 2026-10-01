"""WebSocket /ws/live stub: accepts and closes. Live logic arrives in story 1.3."""

from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.close(code=1000)
