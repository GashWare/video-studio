# autosave.py
"""
Auto-save and crash recovery.

AutoSaver: background thread that saves project state every N seconds.
Crash recovery: on startup, if .autosave.nleproj is newer than last manual save,
                offer to restore it.
"""
import threading
import os
import time
import logging
from tkinter import messagebox

from config import config_mgr
import project_io

AUTOSAVE_FILE = ".autosave.nleproj"

logger = logging.getLogger(__name__)


class AutoSaver:
    """
    Periodically saves project state to AUTOSAVE_FILE.
    Started when a project is open, stopped when app closes.
    """

    def __init__(self, app):
        self._app = app
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        """Start the autosave thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="AutoSaver", daemon=True)
        self._thread.start()
        logger.info("AutoSaver started.")

    def stop(self):
        """Stop and wait for the thread."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("AutoSaver stopped.")

    def _run(self):
        """Thread body: sleep N seconds, call save."""
        interval = int(config_mgr.get("autosave_interval") or 60)
        # Use short sleep increments so we react quickly to stop requests.
        slept = 0
        while not self._stop.is_set():
            time.sleep(1)
            slept += 1
            if slept >= interval:
                slept = 0
                self._save_now()
                # Re-read interval in case the user changed the preference.
                interval = int(config_mgr.get("autosave_interval") or 60)

    def _save_now(self):
        """
        Save current app.project_clips to AUTOSAVE_FILE using project_io.save_project.
        Silently swallows exceptions (never crash the app from autosave).
        """
        try:
            clips = list(getattr(self._app, "project_clips", []) or [])
            proj_w = int(getattr(self._app, "proj_w", 1920) or 1920)
            proj_h = int(getattr(self._app, "proj_h", 1080) or 1080)
            fps    = float(getattr(self._app, "fps",    30.0) or 30.0)
            project_io.save_project(AUTOSAVE_FILE, clips, proj_w, proj_h, fps)
            logger.debug("Autosave written to '%s'.", AUTOSAVE_FILE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Autosave failed (suppressed): %s", exc)


def check_autosave_recovery(app) -> bool:
    """
    Check if AUTOSAVE_FILE exists and is newer than the last known save.
    If so, ask the user whether to restore.
    Returns True if recovery was performed.
    Called from VideoStudioApp.__init__ after UI is built.
    """
    if not os.path.isfile(AUTOSAVE_FILE):
        return False

    autosave_mtime = os.path.getmtime(AUTOSAVE_FILE)

    # Compare against the current project file mtime (if any).
    project_path = getattr(app, "current_project_path", None)
    if project_path and os.path.isfile(project_path):
        project_mtime = os.path.getmtime(project_path)
        if autosave_mtime <= project_mtime:
            # The manual save is at least as recent — nothing to recover.
            return False

    answer = messagebox.askyesno(
        "Crash Recovery",
        "An autosave file was found that is newer than your last manual save.\n\n"
        "Do you want to restore the autosaved project?",
    )
    if not answer:
        return False

    try:
        clips, proj_w, proj_h, fps = project_io.load_project(AUTOSAVE_FILE)
        app.project_clips = clips
        app.proj_w = proj_w
        app.proj_h = proj_h
        app.fps = fps
        # Let the app refresh its UI if it exposes a method for that.
        if hasattr(app, "_refresh_timeline"):
            app._refresh_timeline()
        logger.info("Autosave recovery completed from '%s'.", AUTOSAVE_FILE)
        return True
    except Exception as exc:
        logger.error("Autosave recovery failed: %s", exc)
        messagebox.showerror("Recovery Failed",
                             f"Could not restore the autosave:\n{exc}")
        return False


def clear_autosave():
    """Delete AUTOSAVE_FILE if it exists. Called after successful manual save."""
    try:
        if os.path.isfile(AUTOSAVE_FILE):
            os.remove(AUTOSAVE_FILE)
            logger.debug("Autosave file '%s' removed.", AUTOSAVE_FILE)
    except OSError as exc:
        logger.warning("Could not remove autosave file: %s", exc)
