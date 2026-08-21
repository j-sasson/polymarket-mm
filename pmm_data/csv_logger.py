"""Tiny append-only CSV writer, flushed on every row so logs are safe to tail
and survive a crash without losing the last write."""
from __future__ import annotations

import csv
from pathlib import Path


class CsvLogger:
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        new_file = not path.exists()
        self._fh = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fields)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict):
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        self._fh.close()
