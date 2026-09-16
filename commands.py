"""
commands.py — Global undo/redo command stack for PyDev NLE.

Usage
-----
    from commands import CommandStack, MoveClipCommand, ...

    # In your App class __init__:
    self.cmd_stack = CommandStack(self)

    # To perform an undoable move:
    self.cmd_stack.push(MoveClipCommand(self, clip_id, old_start, new_start))

    # To undo / redo (e.g. from Edit menu or keyboard shortcut):
    self.cmd_stack.undo()
    self.cmd_stack.redo()

After every push / undo / redo the stack calls ``app._after_command()``.
Define that method on your App class to refresh the UI (timeline draw,
proxy rebuild, inspector update, etc.).  Commands themselves must NOT call
``rebuild_proxy_cache()`` or ``timeline.draw()`` directly.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Maximum undo levels retained in memory.
# ---------------------------------------------------------------------------
_MAX_LEVELS: int = 50


# ===========================================================================
# Abstract base
# ===========================================================================

class Command(ABC):
    """Abstract base for every undoable action."""

    #: Human-readable label shown in the Edit menu (e.g. "Move Clip").
    description: str = "Action"

    @abstractmethod
    def execute(self) -> None:
        """Perform (or re-perform) the action."""

    @abstractmethod
    def undo(self) -> None:
        """Reverse the action, restoring previous state."""


# ===========================================================================
# Command stack
# ===========================================================================

class CommandStack:
    """
    Global undo/redo manager.

    Parameters
    ----------
    app:
        The main application instance.  Must expose:
          - ``project_clips: list[dict]``
          - ``_after_command() -> None``
    """

    def __init__(self, app: Any) -> None:
        self._app: Any = app
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, command: Command) -> None:
        """Execute *command* and add it to the undo stack.

        If execution raises an exception the command is discarded and
        the exception is re-raised so the caller can show an error dialog.
        """
        command.execute()  # may raise — do not modify stacks on failure
        self._undo_stack.append(command)
        # Trim to maximum depth
        if len(self._undo_stack) > _MAX_LEVELS:
            self._undo_stack.pop(0)
        # A new action always invalidates the redo branch
        self._redo_stack.clear()
        self._app._after_command()

    def undo(self) -> None:
        """Undo the most recent command, if any."""
        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        self._app._after_command()

    def redo(self) -> None:
        """Re-execute the most recently undone command, if any."""
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        self._app._after_command()

    def clear(self) -> None:
        """Discard both stacks.  Call on project new / open."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        """True when there is at least one command to undo."""
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        """True when there is at least one command to redo."""
        return bool(self._redo_stack)

    @property
    def undo_label(self) -> str:
        """Menu label, e.g. ``"Undo Move Clip"`` or ``"Nothing to undo"``."""
        if self._undo_stack:
            return f"Undo {self._undo_stack[-1].description}"
        return "Nothing to undo"

    @property
    def redo_label(self) -> str:
        """Menu label, e.g. ``"Redo Delete Clip"`` or ``"Nothing to redo"``."""
        if self._redo_stack:
            return f"Redo {self._redo_stack[-1].description}"
        return "Nothing to redo"


# ===========================================================================
# Concrete commands
# ===========================================================================

def _find_clip(clips: list[dict], clip_id: str) -> dict:
    """Return the clip dict with the given id, or raise ``KeyError``."""
    clip = next((c for c in clips if c["id"] == clip_id), None)
    if clip is None:
        raise KeyError(f"Clip '{clip_id}' not found in project_clips.")
    return clip


# ---------------------------------------------------------------------------
# 1. Move clip
# ---------------------------------------------------------------------------

class MoveClipCommand(Command):
    """Slide a clip (and its linked companion if present) to a new timeline start position."""

    description = "Move Clip"

    def __init__(self, app: Any, clip_id: str, old_start: float, new_start: float, sync_linked: bool = True) -> None:
        self._app = app
        self._clip_id = clip_id
        self._old_start = old_start
        self._new_start = new_start
        self._sync_linked = sync_linked
        self._linked_id: str | None = None
        self._linked_old_start: float | None = None

        # Capture linked companion if present
        if self._sync_linked:
            try:
                c = _find_clip(self._app.project_clips, self._clip_id)
                lid = c.get("linked_clip_id")
                if lid:
                    lc = next((x for x in self._app.project_clips if x["id"] == lid), None)
                    if lc:
                        self._linked_id = lid
                        self._linked_old_start = lc["start"]
            except Exception:
                pass

    def execute(self) -> None:
        delta = self._new_start - self._old_start
        _find_clip(self._app.project_clips, self._clip_id)["start"] = self._new_start
        if self._linked_id and self._linked_old_start is not None:
            lc = next((x for x in self._app.project_clips if x["id"] == self._linked_id), None)
            if lc:
                lc["start"] = max(0.0, self._linked_old_start + delta)

    def undo(self) -> None:
        _find_clip(self._app.project_clips, self._clip_id)["start"] = self._old_start
        if self._linked_id and self._linked_old_start is not None:
            lc = next((x for x in self._app.project_clips if x["id"] == self._linked_id), None)
            if lc:
                lc["start"] = self._linked_old_start


# ---------------------------------------------------------------------------
# 2. Trim clip
# ---------------------------------------------------------------------------

class TrimClipCommand(Command):
    """Change a clip's (and linked companion's) trim-in / trim-out points."""

    description = "Trim Clip"

    def __init__(
        self,
        app: Any,
        clip_id: str,
        old_in: float,
        old_out: float,
        new_in: float,
        new_out: float,
        sync_linked: bool = True,
    ) -> None:
        self._app = app
        self._clip_id = clip_id
        self._old_in = old_in
        self._old_out = old_out
        self._new_in = new_in
        self._new_out = new_out
        self._sync_linked = sync_linked
        self._linked_id: str | None = None
        self._linked_old_in: float | None = None
        self._linked_old_out: float | None = None
        self._linked_old_start: float | None = None

        if self._sync_linked:
            try:
                c = _find_clip(self._app.project_clips, self._clip_id)
                lid = c.get("linked_clip_id")
                if lid:
                    lc = next((x for x in self._app.project_clips if x["id"] == lid), None)
                    if lc:
                        self._linked_id = lid
                        self._linked_old_in = lc["trim_in"]
                        self._linked_old_out = lc["trim_out"]
                        self._linked_old_start = lc["start"]
            except Exception:
                pass

    def execute(self) -> None:
        c = _find_clip(self._app.project_clips, self._clip_id)
        c["trim_in"] = self._new_in
        c["trim_out"] = self._new_out
        if self._linked_id:
            lc = next((x for x in self._app.project_clips if x["id"] == self._linked_id), None)
            if lc:
                lc["trim_in"] = self._new_in
                lc["trim_out"] = self._new_out

    def undo(self) -> None:
        c = _find_clip(self._app.project_clips, self._clip_id)
        c["trim_in"] = self._old_in
        c["trim_out"] = self._old_out
        if self._linked_id and self._linked_old_in is not None and self._linked_old_out is not None:
            lc = next((x for x in self._app.project_clips if x["id"] == self._linked_id), None)
            if lc:
                lc["trim_in"] = self._linked_old_in
                lc["trim_out"] = self._linked_old_out


# ---------------------------------------------------------------------------
# 3. Delete clip
# ---------------------------------------------------------------------------

class DeleteClipCommand(Command):
    """Remove a clip (and optionally its linked companion) from the timeline."""

    description = "Delete Clip"

    def __init__(self, app: Any, clip_data_copy: dict, sync_linked: bool = True) -> None:
        self._app = app
        self._clip_data = copy.deepcopy(clip_data_copy)
        self._sync_linked = sync_linked
        self._linked_data: dict | None = None

        if self._sync_linked and self._clip_data.get("linked_clip_id"):
            lid = self._clip_data["linked_clip_id"]
            lc = next((c for c in self._app.project_clips if c["id"] == lid), None)
            if lc:
                self._linked_data = copy.deepcopy(lc)

    def execute(self) -> None:
        del_ids = {self._clip_data["id"]}
        if self._linked_data:
            del_ids.add(self._linked_data["id"])
        self._app.project_clips = [
            c for c in self._app.project_clips if c["id"] not in del_ids
        ]

    def undo(self) -> None:
        if not any(c["id"] == self._clip_data["id"] for c in self._app.project_clips):
            self._app.project_clips.append(copy.deepcopy(self._clip_data))
        if self._linked_data and not any(c["id"] == self._linked_data["id"] for c in self._app.project_clips):
            self._app.project_clips.append(copy.deepcopy(self._linked_data))


# ---------------------------------------------------------------------------
# 4. Add clip
# ---------------------------------------------------------------------------

class AddClipCommand(Command):
    """Append a new clip to the timeline (inverse of delete)."""

    description = "Add Clip"

    def __init__(self, app: Any, clip_data: dict) -> None:
        self._app = app
        self._clip_data = copy.deepcopy(clip_data)

    def execute(self) -> None:
        if not any(c["id"] == self._clip_data["id"] for c in self._app.project_clips):
            self._app.project_clips.append(copy.deepcopy(self._clip_data))

    def undo(self) -> None:
        self._app.project_clips = [
            c for c in self._app.project_clips if c["id"] != self._clip_data["id"]
        ]


# ---------------------------------------------------------------------------
# 5. Split clip
# ---------------------------------------------------------------------------

class SplitClipCommand(Command):
    """
    Split a clip (and its linked companion if present) at *split_t* into two contiguous halves.
    """

    description = "Split Clip"

    def __init__(
        self,
        app: Any,
        original_clip_copy: dict,
        left_id: str,
        right_id: str,
        split_t: float,
        linked_right_id: str | None = None,
    ) -> None:
        self._app = app
        self._original = copy.deepcopy(original_clip_copy)
        self._left_id = left_id
        self._right_id = right_id
        self._split_t = split_t

        self._linked_original: dict | None = None
        self._linked_id: str | None = original_clip_copy.get("linked_clip_id")
        self._linked_right_id: str | None = linked_right_id

        if self._linked_id:
            lc = next((c for c in self._app.project_clips if c["id"] == self._linked_id), None)
            if lc:
                self._linked_original = copy.deepcopy(lc)
                if not self._linked_right_id:
                    import uuid
                    self._linked_right_id = str(uuid.uuid4())

    def execute(self) -> None:
        clips = self._app.project_clips

        # 1. Main clip split
        left = next((c for c in clips if c["id"] == self._left_id), None)
        if left is None:
            left = copy.deepcopy(self._original)
            left["id"] = self._left_id
            clips.append(left)

        # Remove stale right half on redo
        self._app.project_clips = [c for c in self._app.project_clips if c["id"] != self._right_id]

        left = _find_clip(self._app.project_clips, self._left_id)
        right = copy.deepcopy(left)
        right["id"] = self._right_id
        right["trim_in"] = self._split_t
        right["start"] = left["start"] + (self._split_t - left["trim_in"])
        left["trim_out"] = self._split_t

        # 2. Linked companion split
        if self._linked_id and self._linked_original and self._linked_right_id:
            l_left = next((c for c in self._app.project_clips if c["id"] == self._linked_id), None)
            if l_left is None:
                l_left = copy.deepcopy(self._linked_original)
                l_left["id"] = self._linked_id
                self._app.project_clips.append(l_left)

            self._app.project_clips = [c for c in self._app.project_clips if c["id"] != self._linked_right_id]
            l_left = _find_clip(self._app.project_clips, self._linked_id)
            l_right = copy.deepcopy(l_left)
            l_right["id"] = self._linked_right_id
            l_right["trim_in"] = self._split_t
            l_right["start"] = l_left["start"] + (self._split_t - l_left["trim_in"])
            l_left["trim_out"] = self._split_t

            # Pair left with left, right with right
            left["linked_clip_id"] = l_left["id"]
            l_left["linked_clip_id"] = left["id"]
            right["linked_clip_id"] = l_right["id"]
            l_right["linked_clip_id"] = right["id"]
            self._app.project_clips.append(l_right)

        self._app.project_clips.append(right)

    def undo(self) -> None:
        del_ids = {self._right_id}
        if self._linked_right_id:
            del_ids.add(self._linked_right_id)

        self._app.project_clips = [
            c for c in self._app.project_clips if c["id"] not in del_ids
        ]
        left = next((c for c in self._app.project_clips if c["id"] == self._left_id), None)
        if left is not None:
            left.update(copy.deepcopy(self._original))
        else:
            self._app.project_clips.append(copy.deepcopy(self._original))

        if self._linked_id and self._linked_original:
            l_left = next((c for c in self._app.project_clips if c["id"] == self._linked_id), None)
            if l_left is not None:
                l_left.update(copy.deepcopy(self._linked_original))
            else:
                self._app.project_clips.append(copy.deepcopy(self._linked_original))


# ---------------------------------------------------------------------------
# 6. Separate Audio Command
# ---------------------------------------------------------------------------

class SeparateAudioCommand(Command):
    """Extract audio from a video clip onto a dedicated audio track and link them."""

    description = "Separate Audio"

    def __init__(self, app: Any, video_clip_id: str, audio_clip_data: dict) -> None:
        self._app = app
        self._video_id = video_clip_id
        self._audio_data = copy.deepcopy(audio_clip_data)
        self._old_mute: bool = False
        self._old_linked_id: str | None = None

        v = next((c for c in self._app.project_clips if c["id"] == self._video_id), None)
        if v:
            self._old_mute = v.get("mute_audio", False)
            self._old_linked_id = v.get("linked_clip_id")

    def execute(self) -> None:
        v = _find_clip(self._app.project_clips, self._video_id)
        v["mute_audio"] = True
        v["linked_clip_id"] = self._audio_data["id"]
        self._audio_data["linked_clip_id"] = self._video_id

        if not any(c["id"] == self._audio_data["id"] for c in self._app.project_clips):
            self._app.project_clips.append(copy.deepcopy(self._audio_data))

    def undo(self) -> None:
        self._app.project_clips = [
            c for c in self._app.project_clips if c["id"] != self._audio_data["id"]
        ]
        v = next((c for c in self._app.project_clips if c["id"] == self._video_id), None)
        if v:
            v["mute_audio"] = self._old_mute
            v["linked_clip_id"] = self._old_linked_id


# ---------------------------------------------------------------------------
# 7. Toggle Link Command
# ---------------------------------------------------------------------------

class ToggleLinkCommand(Command):
    """Link or unlink two companion clips."""

    description = "Toggle Link"

    def __init__(self, app: Any, clip_a_id: str, clip_b_id: str | None = None) -> None:
        self._app = app
        self._clip_a_id = clip_a_id
        self._clip_b_id = clip_b_id
        self._old_a_linked: str | None = None
        self._old_b_linked: str | None = None

        ca = next((c for c in self._app.project_clips if c["id"] == self._clip_a_id), None)
        if ca:
            self._old_a_linked = ca.get("linked_clip_id")
            if not self._clip_b_id and self._old_a_linked:
                self._clip_b_id = self._old_a_linked

        if self._clip_b_id:
            cb = next((c for c in self._app.project_clips if c["id"] == self._clip_b_id), None)
            if cb:
                self._old_b_linked = cb.get("linked_clip_id")

    def execute(self) -> None:
        ca = next((c for c in self._app.project_clips if c["id"] == self._clip_a_id), None)
        cb = next((c for c in self._app.project_clips if c["id"] == self._clip_b_id), None) if self._clip_b_id else None

        if self._old_a_linked:
            # Currently linked → unlink
            if ca: ca["linked_clip_id"] = None
            if cb: cb["linked_clip_id"] = None
        else:
            # Currently unlinked → link
            if ca and self._clip_b_id: ca["linked_clip_id"] = self._clip_b_id
            if cb: cb["linked_clip_id"] = self._clip_a_id

    def undo(self) -> None:
        ca = next((c for c in self._app.project_clips if c["id"] == self._clip_a_id), None)
        cb = next((c for c in self._app.project_clips if c["id"] == self._clip_b_id), None) if self._clip_b_id else None
        if ca: ca["linked_clip_id"] = self._old_a_linked
        if cb: cb["linked_clip_id"] = self._old_b_linked


# ---------------------------------------------------------------------------
# 6. Duplicate clip
# ---------------------------------------------------------------------------

class DuplicateClipCommand(Command):
    """Duplicate a clip, placing the copy immediately after the source."""

    description = "Duplicate Clip"

    def __init__(self, app: Any, source_id: str, new_clip_id: str) -> None:
        self._app = app
        self._source_id = source_id
        self._new_clip_id = new_clip_id
        # Snapshot of the duplicate's data is built lazily on first execute.
        self._dup_data: dict | None = None

    def execute(self) -> None:
        if self._dup_data is None:
            # First execution: derive the duplicate from the live source.
            src = _find_clip(self._app.project_clips, self._source_id)
            dup = copy.deepcopy(src)
            dup["id"] = self._new_clip_id
            dup["start"] = src["start"] + (src["trim_out"] - src["trim_in"])
            self._dup_data = dup
        # Re-insert on redo (guard against already present).
        if not any(c["id"] == self._new_clip_id for c in self._app.project_clips):
            self._app.project_clips.append(copy.deepcopy(self._dup_data))

    def undo(self) -> None:
        self._app.project_clips = [
            c for c in self._app.project_clips if c["id"] != self._new_clip_id
        ]


# ---------------------------------------------------------------------------
# 7. Inspector change
# ---------------------------------------------------------------------------

class InspectorChangeCommand(Command):
    """
    Record a generic dict-diff for any inspector field change.

    Parameters
    ----------
    old_props:
        Mapping of ``{field: old_value}`` for each changed key.
    new_props:
        Mapping of ``{field: new_value}`` for each changed key.
    """

    description = "Edit Clip Properties"

    def __init__(
        self,
        app: Any,
        clip_id: str,
        old_props: dict[str, Any],
        new_props: dict[str, Any],
    ) -> None:
        self._app = app
        self._clip_id = clip_id
        self._old_props = copy.deepcopy(old_props)
        self._new_props = copy.deepcopy(new_props)

    def execute(self) -> None:
        c = _find_clip(self._app.project_clips, self._clip_id)
        c.update(self._new_props)

    def undo(self) -> None:
        c = _find_clip(self._app.project_clips, self._clip_id)
        c.update(self._old_props)


# ---------------------------------------------------------------------------
# 8. Ripple Delete Clip
# ---------------------------------------------------------------------------

class RippleDeleteClipCommand(Command):
    """Delete a clip (and its linked companion) and shift subsequent clips left to close the gap."""

    description = "Ripple Delete Clip"

    def __init__(self, app: Any, clip_data_copy: dict, sync_linked: bool = True) -> None:
        self._app = app
        self._clip_data = copy.deepcopy(clip_data_copy)
        self._sync_linked = sync_linked
        self._linked_data: dict | None = None
        self._duration = self._clip_data["trim_out"] - self._clip_data["trim_in"]
        self._start_t = self._clip_data["start"]
        self._affected_tracks = {self._clip_data.get("track")}

        if self._sync_linked and self._clip_data.get("linked_clip_id"):
            lid = self._clip_data["linked_clip_id"]
            lc = next((c for c in self._app.project_clips if c["id"] == lid), None)
            if lc:
                self._linked_data = copy.deepcopy(lc)
                self._affected_tracks.add(lc.get("track"))

        self._shifted_clip_ids = []
        for c in self._app.project_clips:
            if c["id"] != self._clip_data["id"] and (not self._linked_data or c["id"] != self._linked_data["id"]):
                if c.get("track") in self._affected_tracks and c["start"] >= self._start_t + self._duration - 0.001:
                    self._shifted_clip_ids.append(c["id"])

    def execute(self) -> None:
        del_ids = {self._clip_data["id"]}
        if self._linked_data:
            del_ids.add(self._linked_data["id"])
        self._app.project_clips = [c for c in self._app.project_clips if c["id"] not in del_ids]
        for c in self._app.project_clips:
            if c["id"] in self._shifted_clip_ids:
                c["start"] = max(0.0, c["start"] - self._duration)

    def undo(self) -> None:
        for c in self._app.project_clips:
            if c["id"] in self._shifted_clip_ids:
                c["start"] += self._duration
        if not any(c["id"] == self._clip_data["id"] for c in self._app.project_clips):
            self._app.project_clips.append(copy.deepcopy(self._clip_data))
        if self._linked_data and not any(c["id"] == self._linked_data["id"] for c in self._app.project_clips):
            self._app.project_clips.append(copy.deepcopy(self._linked_data))


# ---------------------------------------------------------------------------
# 9. Slip Clip
# ---------------------------------------------------------------------------

class SlipClipCommand(Command):
    """Slip a clip: change trim_in and trim_out by delta while maintaining timeline start and duration."""

    description = "Slip Clip"

    def __init__(self, app: Any, clip_id: str, old_in: float, old_out: float, new_in: float, new_out: float) -> None:
        self._app = app
        self._clip_id = clip_id
        self._old_in = old_in
        self._old_out = old_out
        self._new_in = new_in
        self._new_out = new_out

    def execute(self) -> None:
        c = _find_clip(self._app.project_clips, self._clip_id)
        c["trim_in"] = self._new_in
        c["trim_out"] = self._new_out

    def undo(self) -> None:
        c = _find_clip(self._app.project_clips, self._clip_id)
        c["trim_in"] = self._old_in
        c["trim_out"] = self._old_out


# ---------------------------------------------------------------------------
# 10. Slide Clip
# ---------------------------------------------------------------------------

class SlideClipCommand(Command):
    """Slide a clip: move target clip start while trimming previous neighbor out and next neighbor in."""

    description = "Slide Clip"

    def __init__(self, app: Any, clip_id: str, delta: float, prev_id: str | None, next_id: str | None) -> None:
        self._app = app
        self._clip_id = clip_id
        self._delta = delta
        self._prev_id = prev_id
        self._next_id = next_id

    def execute(self) -> None:
        c = _find_clip(self._app.project_clips, self._clip_id)
        c["start"] += self._delta
        if self._prev_id:
            try:
                prev_c = _find_clip(self._app.project_clips, self._prev_id)
                prev_c["trim_out"] = max(prev_c["trim_in"] + 0.1, prev_c["trim_out"] + self._delta)
            except Exception:
                pass
        if self._next_id:
            try:
                next_c = _find_clip(self._app.project_clips, self._next_id)
                next_c["start"] += self._delta
                next_c["trim_in"] = max(0.0, next_c["trim_in"] + self._delta)
            except Exception:
                pass

    def undo(self) -> None:
        c = _find_clip(self._app.project_clips, self._clip_id)
        c["start"] -= self._delta
        if self._prev_id:
            try:
                prev_c = _find_clip(self._app.project_clips, self._prev_id)
                prev_c["trim_out"] = max(prev_c["trim_in"] + 0.1, prev_c["trim_out"] - self._delta)
            except Exception:
                pass
        if self._next_id:
            try:
                next_c = _find_clip(self._app.project_clips, self._next_id)
                next_c["start"] -= self._delta
                next_c["trim_in"] = max(0.0, next_c["trim_in"] - self._delta)
            except Exception:
                pass
