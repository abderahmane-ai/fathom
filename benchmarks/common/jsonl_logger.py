"""JSON-line structured logger for per-step benchmark events.

Writes one JSON object per line to a `.jsonl` file, suitable for streaming
ingest by `scripts/ingest/collect.py`.  Each event has a `ts` (ISO UTC), an
`event` (str), and any additional fields the caller provides.

This is intentionally NOT a replacement for the Lightning CSVLogger — it
complements it.  Use this for *run-level* events (start, end, NaN, errors)
and let Lightning handle *per-step training metrics* (loss, lr, tps).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append-only JSON-line writer.  Thread-safe (single lock).

    Args:
        path: output file (overwritten on first open).  Parent dir is
            created if missing.
        run_id: optional run_id stored as a top-level field on every event.
    """

    def __init__(self, path: Path | str, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self.run_id = run_id
        self.n_events = 0

    def emit(self, event: str, **fields: Any) -> None:
        """Write one event line.  Each call adds `ts` and `event` automatically.

        `fields` is the arbitrary per-event payload (e.g. step, loss, lr, tps).
        Non-JSON-serializable values are coerced to their `repr`.
        """
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        if self.run_id is not None:
            record["run_id"] = self.run_id
        for key, value in fields.items():
            try:
                json.dumps(value)
                record[key] = value
            except (TypeError, ValueError):
                record[key] = repr(value)
        with self._lock:
            self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fp.flush()
        self.n_events += 1

    def close(self) -> None:
        """Close the underlying file.  Idempotent."""
        with self._lock:
            if not self._fp.closed:
                self._fp.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
