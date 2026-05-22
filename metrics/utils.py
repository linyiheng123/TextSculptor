"""Shared utility functions for TextSculpt-Bench evaluation."""

import json
import os
from typing import List


def load_jsonl_lines(file_path: str) -> List[dict]:
    """Load all valid JSON objects from a JSONL file."""
    lines = []
    if not os.path.exists(file_path):
        return lines

    with open(file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                lines.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return lines


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
