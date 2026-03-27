from __future__ import annotations

from datetime import datetime, timezone

from ai_factory.core.instances.models import InstanceManifest
from ai_factory.core.instances.store import FileInstanceStore


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


class InstanceQueryService:
    def __init__(self, store: FileInstanceStore):
        self.store = store

    def list_instances(
        self,
        *,
        instance_type: str | None = None,
        status: str | None = None,
        parent_instance_id: str | None = None,
    ) -> list[InstanceManifest]:
        items = self.store.list_instances()
        if instance_type:
            items = [item for item in items if item.type == instance_type]
        if status:
            items = [item for item in items if item.status == status]
        if parent_instance_id:
            items = [item for item in items if item.parent_instance_id == parent_instance_id]
        return sorted(items, key=lambda item: _parse_ts(item.updated_at), reverse=True)

    def get_children(self, parent_instance_id: str) -> list[InstanceManifest]:
        return self.list_instances(parent_instance_id=parent_instance_id)
