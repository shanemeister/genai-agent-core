"""MCP Filesystem Integration — Privacy Zone 1 compliant.

Provides controlled access to local/network filesystems for document ingestion.
Files are read-only accessed and never transmitted beyond the local machine.

All file access is restricted to user-configured allowed directories.
"""

from pathlib import Path
from typing import List, Dict
import hashlib

from core.settings.filesystem_settings import load_allowed_roots


# Supported file extensions for document ingestion
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".csv",
    ".jpg", ".jpeg", ".png", ".svg"
}


class FilesystemAccess:
    """Privacy-first filesystem access for document ingestion.

    All operations check against user-configured allowlist.
    """

    def list_files(self, directory: str, recursive: bool = False) -> List[Dict]:
        """List files in directory (only within allowed roots).

        Args:
            directory: Path to directory to list
            recursive: If True, search subdirectories recursively

        Returns:
            List of file metadata dicts:
            [
                {
                    "filename": "document.pdf",
                    "path": "/full/path/to/document.pdf",
                    "size": 12345,
                    "type": "pdf"
                },
                ...
            ]

        Raises:
            PermissionError: If directory is not in allowed roots
        """
        dir_path = Path(directory)

        if not self._is_allowed_path(dir_path):
            raise PermissionError(f"Access denied: {directory}")

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        files = []

        try:
            for path in dir_path.glob(pattern):
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                    files.append({
                        "filename": path.name,
                        "path": str(path),
                        "size": path.stat().st_size,
                        "type": path.suffix[1:],  # Remove leading dot
                    })
        except PermissionError as e:
            # Some subdirectories might not be accessible
            raise PermissionError(f"Cannot access directory: {e}")

        return files

    def read_file(self, filepath: str) -> bytes:
        """Read file contents (only allowed paths).

        Args:
            filepath: Full path to file

        Returns:
            File contents as bytes

        Raises:
            PermissionError: If file is not in allowed roots
            ValueError: If file extension is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(filepath)

        if not self._is_allowed_path(path):
            raise PermissionError(f"Access denied: {filepath}")

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if not path.is_file():
            raise ValueError(f"Not a file: {filepath}")

        return path.read_bytes()

    def compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash for deduplication.

        Args:
            content: File contents

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(content).hexdigest()

    def get_allowed_roots(self) -> List[str]:
        """Get list of currently allowed directory roots.

        Returns:
            List of allowed directory paths
        """
        return load_allowed_roots()

    def _is_allowed_path(self, path: Path) -> bool:
        """Check if path is within user-configured allowed roots.

        Args:
            path: Path to check

        Returns:
            True if path is within an allowed root, False otherwise
        """
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            # Can't resolve path (broken symlink, etc.)
            return False

        allowed_roots = load_allowed_roots()

        for root in allowed_roots:
            try:
                root_resolved = Path(root).resolve()
                # Check if resolved path starts with allowed root
                if str(resolved).startswith(str(root_resolved)):
                    return True
            except (OSError, RuntimeError):
                # Can't resolve root - skip it
                continue

        return False
