# crash_reporter.py
"""
Global exception handler.
Wraps app.mainloop() to catch uncaught exceptions.
Writes a timestamped crash report and offers to show the log.
"""
import sys
import os
import traceback
import logging
import platform
import subprocess
from datetime import datetime
import tkinter as tk

CRASH_DIR = "crash_reports"
LOG_FILE  = "app.log"

logger = logging.getLogger(__name__)


def install_crash_handler(app):
    """
    Install sys.excepthook to catch unhandled exceptions.
    Writes crash report file and shows a dialog.
    Call this before app.mainloop().
    """
    def _hook(exc_type, exc_value, exc_tb):
        # Always log it first so app.log is up-to-date before we read it.
        logger.critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        try:
            report_path = write_crash_report(exc_type, exc_value, exc_tb)
            error_summary = f"{exc_type.__name__}: {exc_value}"
            # show_crash_dialog must run on the main thread — use after() if
            # the exception fires from a background thread.
            try:
                app.after(0, show_crash_dialog, app, report_path, error_summary)
            except Exception:
                show_crash_dialog(app, report_path, error_summary)
        except Exception as inner:
            # Last-resort fallback: never let the crash handler itself crash.
            print(f"[crash_reporter] Failed to handle exception: {inner}", file=sys.stderr)

        # Call the original excepthook so Python still prints the traceback.
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def write_crash_report(exc_type, exc_value, exc_tb) -> str:
    """
    Write a crash report to crash_reports/<timestamp>.txt.
    Returns the report file path.
    Contents: timestamp, Python version, platform, traceback, last 50 lines of app.log.
    """
    os.makedirs(CRASH_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(CRASH_DIR, f"crash_{timestamp}.txt")

    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text  = "".join(tb_lines)

    # Grab the last 50 lines of app.log if it exists.
    log_tail = ""
    if os.path.isfile(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as fh:
                all_lines = fh.readlines()
            log_tail = "".join(all_lines[-50:])
        except Exception as exc:
            log_tail = f"(could not read log: {exc})"

    report_lines = [
        "=" * 72,
        f"CRASH REPORT — {datetime.now().isoformat(sep=' ')}",
        "=" * 72,
        f"Python  : {sys.version}",
        f"Platform: {platform.platform()}",
        f"CWD     : {os.getcwd()}",
        "",
        "── Traceback ──────────────────────────────────────────────────────────",
        tb_text,
        "── Last 50 lines of app.log ───────────────────────────────────────────",
        log_tail if log_tail else "(log file not found)",
        "=" * 72,
    ]

    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report_lines))
        logger.info("Crash report written to '%s'.", report_path)
    except Exception as exc:
        logger.error("Could not write crash report: %s", exc)

    return report_path


def show_crash_dialog(app, report_path: str, error_summary: str):
    """
    Show a non-blocking Toplevel dialog telling the user what happened,
    the report path, and offering 'View Report' and 'Close' buttons.
    """
    win = tk.Toplevel()
    win.title("Application Error")
    win.resizable(False, False)
    win.grab_set()

    # ── Message ──────────────────────────────────────────────────────────────
    header = tk.Label(
        win,
        text="The application encountered an unexpected error.",
        font=("Segoe UI", 11, "bold"),
        wraplength=420,
        justify="left",
    )
    header.pack(padx=20, pady=(16, 4), anchor="w")

    detail = tk.Label(
        win,
        text=error_summary,
        font=("Courier New", 9),
        fg="#cc0000",
        wraplength=420,
        justify="left",
    )
    detail.pack(padx=20, pady=(0, 8), anchor="w")

    path_lbl = tk.Label(
        win,
        text=f"Crash report saved to:\n{os.path.abspath(report_path)}",
        font=("Segoe UI", 9),
        wraplength=420,
        justify="left",
    )
    path_lbl.pack(padx=20, pady=(0, 12), anchor="w")

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_frame = tk.Frame(win)
    btn_frame.pack(fill="x", padx=20, pady=(0, 14))

    def _view():
        abs_path = os.path.abspath(report_path)
        try:
            if sys.platform == "win32":
                os.startfile(abs_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", abs_path])
            else:
                subprocess.Popen(["xdg-open", abs_path])
        except Exception as exc:
            tk.messagebox.showerror("Cannot Open File", str(exc), parent=win)

    tk.Button(btn_frame, text="View Report", width=12, command=_view).pack(side="left")
    tk.Button(btn_frame, text="Close", width=12, command=win.destroy).pack(side="right")

    # Centre on screen.
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")
