from collections import defaultdict
from fastapi import WebSocket
from ai_factory.schemas.websocket import TelemetryMessage


class ConnectionManager:
    def __init__(self):
        self.active: set[WebSocket] = set()
        self.subscriptions: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        for subs in self.subscriptions.values():
            subs.discard(ws)

    async def broadcast(self, message: TelemetryMessage):
        payload = message.model_dump_json()
        dead = set()
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_to_subscribers(self, topic: str, message: TelemetryMessage):
        payload = message.model_dump_json()
        for ws in self.subscriptions.get(topic, set()):
            try:
                await ws.send_text(payload)
            except Exception:
                pass


manager = ConnectionManager()
