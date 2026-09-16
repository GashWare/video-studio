"""
project_io.py — Project save/load for PyDev NLE video editor.

Project files use the `.nleproj` extension and are ZIP archives containing:
  - project.json   : all clip metadata (numpy masks excluded)
  - masks/          : one .npy file per mask entry (<clip_id>_<frame_idx>.npy)
"""

import io
import json
import logging
import os
import zipfile
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_VERSION = 2
RECENT_PROJECTS_KEY = "recent_projects"
_MAX_RECENT = 8

# Sequence-level setting defaults used when a project file omits them.
_SEQ_DEFAULTS: dict[str, Any] = {
    "proj_w": 1920,
    "proj_h": 1080,
    "fps": 30.0,
}


# ── Save ──────────────────────────────────────────────────────────────────────

def save_project(
    filepath: str,
    project_clips: list[dict],
    proj_w: int,
    proj_h: int,
    fps: float,
) -> None:
    """Save a project to *filepath* (a ``.nleproj`` ZIP archive).

    Raises
    ------
    OSError
        If the destination path cannot be written.
    ValueError
        If *project_clips* is not a list.
    """
    if not isinstance(project_clips, list):
        raise ValueError("project_clips must be a list of clip dicts.")

    # Build the serialisable clip list, extracting numpy masks on the side.
    serialisable_clips: list[dict] = []
    # collected_masks: list of (clip_id, frame_idx_str, ndarray)
    collected_masks: list[tuple[str, str, np.ndarray]] = []

    for clip in project_clips:
        clip_copy = dict(clip)  # shallow copy — we only mutate custom_masks
        clip_id = str(clip_copy.get("id", ""))

        raw_masks: dict = clip_copy.pop("custom_masks", {}) or {}
        mask_keys: list[str] = []  # frame indices that have been saved

        for frame_idx, arr in raw_masks.items():
            if arr is None:
                continue
            frame_str = str(frame_idx)
            collected_masks.append((clip_id, frame_str, arr))
            mask_keys.append(frame_str)

        # Store only the list of frame indices so load_project knows what to
        # look for; the arrays themselves live in the masks/ directory.
        clip_copy["custom_masks"] = mask_keys
        serialisable_clips.append(clip_copy)

    project_doc = {
        "version": PROJECT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "proj_w": proj_w,
        "proj_h": proj_h,
        "fps": fps,
        "clips": serialisable_clips,
    }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Write the JSON manifest.
            json_bytes = json.dumps(project_doc, indent=2, ensure_ascii=False).encode("utf-8")
            zf.writestr("project.json", json_bytes)

            # Write each mask as a .npy file inside masks/.
            for clip_id, frame_str, arr in collected_masks:
                buf = io.BytesIO()
                np.save(buf, arr)
                mask_name = f"masks/{clip_id}_{frame_str}.npy"
                zf.writestr(mask_name, buf.getvalue())

    except OSError as exc:
        raise OSError(f"Failed to write project file '{filepath}': {exc}") from exc

    logger.info(
        "Project saved to '%s' (%d clip(s), %d mask(s)).",
        filepath,
        len(serialisable_clips),
        len(collected_masks),
    )


# ── Load ──────────────────────────────────────────────────────────────────────

def load_project(filepath: str) -> tuple[list, int, int, float]:
    """Load a project from *filepath* and return ``(clips, proj_w, proj_h, fps)``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If the file contains an unsupported project version.
    zipfile.BadZipFile
        If the file is not a valid ZIP / .nleproj archive.
    json.JSONDecodeError
        If ``project.json`` inside the archive is malformed.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Project file not found: '{filepath}'")

    with zipfile.ZipFile(filepath, "r") as zf:
        # ── Read & validate the JSON manifest ─────────────────────────────
        try:
            json_bytes = zf.read("project.json")
        except KeyError as exc:
            raise ValueError(
                f"'{filepath}' is missing 'project.json' — not a valid .nleproj file."
            ) from exc

        project_doc: dict = json.loads(json_bytes.decode("utf-8"))

        version = project_doc.get("version")
        if version != PROJECT_VERSION:
            # Graceful degradation: warn but attempt load for forward-compatible minor changes.
            logger.warning(
                "Project version mismatch: file has version %r, expected %r. "
                "Attempting load anyway — some features may be missing.",
                version,
                PROJECT_VERSION,
            )

        # ── Sequence settings (with safe defaults) ─────────────────────────
        proj_w = int(project_doc.get("proj_w", _SEQ_DEFAULTS["proj_w"]))
        proj_h = int(project_doc.get("proj_h", _SEQ_DEFAULTS["proj_h"]))
        fps    = float(project_doc.get("fps",    _SEQ_DEFAULTS["fps"]))

        # Build a set of available mask paths for fast look-up.
        available_masks: set[str] = {name for name in zf.namelist() if name.startswith("masks/")}

        # ── Reconstruct clips ──────────────────────────────────────────────
        raw_clips: list[dict] = project_doc.get("clips", [])
        project_clips: list[dict] = []

        for clip in raw_clips:
            clip_id = str(clip.get("id", ""))
            # custom_masks was stored as a list of frame-index strings.
            mask_keys: list[str] = clip.pop("custom_masks", []) or []
            reconstructed_masks: dict[str | int, np.ndarray] = {}

            for frame_str in mask_keys:
                mask_path = f"masks/{clip_id}_{frame_str}.npy"
                if mask_path not in available_masks:
                    logger.warning(
                        "Mask file '%s' missing from archive — skipping.", mask_path
                    )
                    continue
                try:
                    buf = io.BytesIO(zf.read(mask_path))
                    arr = np.load(buf, allow_pickle=False)
                    # Restore numeric key if possible, else keep as string.
                    try:
                        key: str | int = int(frame_str)
                    except ValueError:
                        key = frame_str
                    reconstructed_masks[key] = arr
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Could not load mask '%s': %s — skipping.", mask_path, exc
                    )

            clip["custom_masks"] = reconstructed_masks
            project_clips.append(clip)

    logger.info(
        "Project loaded from '%s' — version %s, %d clip(s).",
        filepath,
        version,
        len(project_clips),
    )
    return project_clips, proj_w, proj_h, fps


# ── Recent-projects helpers ───────────────────────────────────────────────────

def get_recent_projects(config_mgr) -> list[str]:
    """Return up to ``_MAX_RECENT`` recent project paths from *config_mgr*.

    *config_mgr* is expected to expose a ``get(key, default)`` interface
    (e.g. the app's existing ConfigManager).  Returns an empty list if no
    recent projects have been stored yet.
    """
    recent = config_mgr.get(RECENT_PROJECTS_KEY, [])
    if not isinstance(recent, list):
        return []
    return recent[:_MAX_RECENT]


def add_recent_project(config_mgr, filepath: str) -> None:
    """Prepend *filepath* to the recent-projects list in *config_mgr*.

    Deduplicates (case-sensitive path comparison) and trims to
    ``_MAX_RECENT`` entries.  Persists via ``config_mgr.set(key, value)``.
    """
    recent: list[str] = get_recent_projects(config_mgr)

    # Remove any existing occurrence of this path so the new entry is first.
    recent = [p for p in recent if p != filepath]
    recent.insert(0, filepath)
    recent = recent[:_MAX_RECENT]

    config_mgr.set(RECENT_PROJECTS_KEY, recent)
