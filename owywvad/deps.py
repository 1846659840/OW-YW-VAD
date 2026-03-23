from __future__ import annotations

import subprocess
from pathlib import Path

from owywvad.utils import ensure_dir, load_yaml, project_root


def install_dependencies() -> list[str]:
    root = project_root()
    deps_cfg = load_yaml(root / "deps.yaml").get("dependencies", {})
    messages: list[str] = []
    for name, spec in deps_cfg.items():
        repo = spec["repo"]
        commit = spec["commit"]
        target = root / spec["target_dir"]
        ensure_dir(target.parent)
        if not target.exists():
            subprocess.run(["git", "clone", repo, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "fetch", "--all", "--tags"], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", commit], check=True)
        messages.append(f"{name}: {target} @ {commit}")
    return messages

