"""Filesystem settings for MCP document access.

Manages allowed directory allowlist for document ingestion.
Settings stored on server at /home/exx/myCode/genai-agent-core/data/settings/.
"""

import json
from pathlib import Path

# Settings stored on server (not client)
SETTINGS_PATH = Path("data/settings/filesystem.json")

DEFAULT_ALLOWED_ROOTS = [
    str(Path.home() / "Documents"),
    str(Path.home() / "Downloads"),
    "/Volumes",  # macOS network shares
    "/home/exx/Documents",  # Linux server documents
    "/mnt",  # Linux mount points
]


def load_allowed_roots() -> list[str]:
    """Load user-configured allowed roots from JSON file."""
    if not SETTINGS_PATH.exists():
        return DEFAULT_ALLOWED_ROOTS

    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        # Fallback to defaults if file is corrupted
        return DEFAULT_ALLOWED_ROOTS


def save_allowed_roots(roots: list[str]) -> None:
    """Save allowed roots to JSON file on server."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(roots, indent=2))


def is_path_allowed(path: str) -> bool:
    """Check if a path is within allowed roots."""
    resolved = Path(path).resolve()
    allowed_roots = load_allowed_roots()

    return any(
        str(resolved).startswith(str(Path(root).resolve()))
        for root in allowed_roots
    )
