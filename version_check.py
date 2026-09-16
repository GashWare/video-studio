# version_check.py
"""
Version check against GitHub releases API.
Non-blocking: runs in a background thread, shows a banner in the status bar if
a newer version is available.
"""
import threading
import logging
import urllib.request
import json
import webbrowser

import customtkinter as ctk

APP_VERSION  = "1.0.0"
GITHUB_REPO  = "GashWare/video-studio"

_API_URL     = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"

logger = logging.getLogger(__name__)


def _parse_version(tag: str) -> tuple[int, ...]:
    """Convert a version tag (e.g. 'v1.2.3' or '1.2.3') to a comparable tuple."""
    tag = tag.lstrip("v").strip()
    try:
        return tuple(int(x) for x in tag.split("."))
    except ValueError:
        return (0,)


def _is_newer(remote_tag: str, current: str = APP_VERSION) -> bool:
    """Return True if *remote_tag* represents a version newer than *current*."""
    return _parse_version(remote_tag) > _parse_version(current)


def check_for_updates(app, status_label=None):
    """
    Run a background thread to check for updates.
    If a newer version is found, call app.after(0, _show_update_banner, app, version).
    Silently swallows all network errors.
    """
    def _worker():
        try:
            req = urllib.request.Request(
                _API_URL,
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": f"video-studio/{APP_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            tag_name = data.get("tag_name", "")
            if not tag_name:
                return

            if _is_newer(tag_name):
                clean = tag_name.lstrip("v")
                logger.info("Update available: %s (current: %s)", clean, APP_VERSION)
                app.after(0, _show_update_banner, app, clean)
            else:
                logger.debug("No update found (latest: %s, current: %s).", tag_name, APP_VERSION)

        except Exception as exc:  # noqa: BLE001
            # Network errors, 404s, JSON errors — all silenced.
            logger.debug("Version check failed (suppressed): %s", exc)

    t = threading.Thread(target=_worker, name="VersionCheck", daemon=True)
    t.start()


def _show_update_banner(app, new_version: str):
    """
    Show a non-intrusive update notification.
    Adds a small CTkLabel to the app's status bar with a link to the release page.
    Clicking it opens the URL in the default browser.
    """
    # Try to attach to a known status-bar frame; fall back to the root window.
    parent = None
    for attr in ("status_bar", "statusbar", "_status_bar", "status_frame"):
        candidate = getattr(app, attr, None)
        if candidate is not None:
            parent = candidate
            break
    if parent is None:
        parent = app

    banner_text = (
        f"  ✨ Version {new_version} available — "
        f"download at github.com/{GITHUB_REPO}  "
    )
    banner = ctk.CTkLabel(
        parent,
        text=banner_text,
        fg_color="#fef08a",   # soft yellow
        text_color="#713f12",
        corner_radius=4,
        cursor="hand2",
        font=ctk.CTkFont(size=12),
    )
    banner.pack(side="right", padx=6, pady=2)

    def _open_url(_event=None):
        webbrowser.open(_RELEASE_URL)

    banner.bind("<Button-1>", _open_url)

    logger.info("Update banner shown for version %s.", new_version)


def get_current_version() -> str:
    return APP_VERSION
