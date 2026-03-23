from __future__ import annotations

from pathlib import Path

from owywvad.utils import load_yaml


def load_prompt_groups(path: Path) -> dict[str, list[str]]:
    payload = load_yaml(path)
    return {
        "objects": list(payload.get("objects", [])),
        "states": list(payload.get("states", [])),
        "rules": list(payload.get("rules", [])),
    }

