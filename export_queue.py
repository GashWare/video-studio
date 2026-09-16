# export_queue.py
"""
Batch render queue manager for Video Studio.
Opens via File → Export Queue.
"""

import copy
import threading
import uuid
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# ---------------------------------------------------------------------------
# ExportJob — data model for a single queued render
# ---------------------------------------------------------------------------

class ExportJob:
    """One queued render job."""

    STATUSES = ("queued", "running", "done", "failed", "cancelled")

    def __init__(
        self,
        job_id: str,
        label: str,
        project_clips: list,
        out_path: str,
        proj_w: int,
        proj_h: int,
        fps: int,
        codec: str,
        bitrate=None,
        track_volumes=None,
        track_pans=None,
    ):
        self.job_id        = job_id
        self.label         = label
        self.project_clips = copy.deepcopy(project_clips)
        self.out_path      = out_path
        self.proj_w        = proj_w
        self.proj_h        = proj_h
        self.fps           = fps
        self.codec         = codec
        self.bitrate       = bitrate
        self.track_volumes = dict(track_volumes) if track_volumes else {}
        self.track_pans    = dict(track_pans) if track_pans else {}
        self.status        = "queued"   # "queued"|"running"|"done"|"failed"|"cancelled"
        self.progress      = 0.0        # 0.0 – 100.0
        self.error         = None
        self._task         = None       # RenderTask instance while running

    def cancel(self):
        """Request cancellation of the running task (if any)."""
        self.status = "cancelled"
        if self._task is not None:
            self._task.cancel()

    def __repr__(self) -> str:
        return f"<ExportJob id={self.job_id!r} label={self.label!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# _JobRow — one row widget inside the queue window
# ---------------------------------------------------------------------------

class _JobRow(ctk.CTkFrame):
    """Visual row for a single ExportJob inside the queue window."""

    _STATUS_COLORS = {
        "queued":    "#6b7280",
        "running":   "#3b82f6",
        "done":      "#22c55e",
        "failed":    "#ef4444",
        "cancelled": "#f59e0b",
    }

    def __init__(self, parent, job: ExportJob, cancel_cb):
        super().__init__(parent, fg_color="#2a2a2a", corner_radius=6)
        self._job       = job
        self._cancel_cb = cancel_cb
        self._build()

    def _build(self):
        # Label
        self._label_lbl = ctk.CTkLabel(
            self, text=self._job.label, anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"), width=200
        )
        self._label_lbl.grid(row=0, column=0, padx=8, pady=(6, 2), sticky="w")

        # Output path (truncated)
        self._path_lbl = ctk.CTkLabel(
            self, text=self._short_path(self._job.out_path),
            anchor="w", font=ctk.CTkFont(size=10), text_color="#888888", width=200
        )
        self._path_lbl.grid(row=1, column=0, padx=8, pady=(0, 4), sticky="w")

        # Status badge
        self._status_lbl = ctk.CTkLabel(
            self, text=self._job.status.upper(),
            fg_color=self._STATUS_COLORS.get(self._job.status, "#6b7280"),
            corner_radius=4, width=80, font=ctk.CTkFont(size=10, weight="bold"),
        )
        self._status_lbl.grid(row=0, column=1, padx=8, pady=(6, 2))

        # Progress bar
        self._progress_bar = ctk.CTkProgressBar(self, width=200)
        self._progress_bar.set(0.0)
        self._progress_bar.grid(row=1, column=1, padx=8, pady=(0, 4))

        # Cancel button
        self._cancel_btn = ctk.CTkButton(
            self, text="✕", width=30, height=28,
            fg_color="#6b7280",
            command=lambda: self._cancel_cb(self._job),
        )
        self._cancel_btn.grid(row=0, column=2, rowspan=2, padx=8, pady=4)

    @staticmethod
    def _short_path(path: str, max_len: int = 45) -> str:
        if len(path) <= max_len:
            return path
        return "…" + path[-(max_len - 1):]

    def refresh(self):
        """Update displayed state from the job model."""
        status = self._job.status
        self._status_lbl.configure(
            text=status.upper(),
            fg_color=self._STATUS_COLORS.get(status, "#6b7280"),
        )
        self._progress_bar.set(self._job.progress / 100.0)
        # Disable cancel for terminal states
        if status in ("done", "failed", "cancelled"):
            self._cancel_btn.configure(state="disabled", fg_color="#333333")


# ---------------------------------------------------------------------------
# ExportQueueWindow
# ---------------------------------------------------------------------------

class ExportQueueWindow(ctk.CTkToplevel):
    """
    Floating queue window listing all render jobs.
    Opens from File → Export Queue.

    Jobs run sequentially (one RenderTask at a time).
    Jobs persist in memory for the session.
    """

    _POLL_MS = 200   # UI refresh interval

    def __init__(self, app):
        super().__init__(app)
        self.title("Export Queue")
        self.geometry("620x480")
        self.resizable(True, True)

        self._app   = app
        self._jobs: list[ExportJob]      = []
        self._rows: dict[str, _JobRow]   = {}   # job_id → _JobRow
        self._running_job: ExportJob | None = None

        self._build_ui()
        self._start_poll()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top toolbar
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkButton(
            toolbar, text="＋ Add Current Project",
            command=self._add_current_project, width=180,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            toolbar, text="Clear Finished",
            command=self._clear_finished,
            fg_color="gray40", width=130,
        ).pack(side="left", padx=4)

        self._running_lbl = ctk.CTkLabel(toolbar, text="", text_color="#3b82f6")
        self._running_lbl.pack(side="right", padx=8)

        # Scrollable job list
        self._scroll = ctk.CTkScrollableFrame(self, label_text="")
        self._scroll.pack(fill="both", expand=True, padx=8, pady=4)

        # Overall progress label
        self._overall_lbl = ctk.CTkLabel(self, text="0 / 0 jobs done",
                                          font=ctk.CTkFont(size=11))
        self._overall_lbl.pack(pady=(2, 6))

    def _rebuild_rows(self):
        """Recreate all row widgets from the jobs list."""
        for w in self._scroll.winfo_children():
            w.destroy()
        self._rows.clear()
        for job in self._jobs:
            row = _JobRow(self._scroll, job, cancel_cb=self._cancel_job)
            row.pack(fill="x", padx=4, pady=3)
            self._rows[job.job_id] = row

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_job(self, job: ExportJob):
        """Append a job to the queue and start it if nothing is running."""
        self._jobs.append(job)
        row = _JobRow(self._scroll, job, cancel_cb=self._cancel_job)
        row.pack(fill="x", padx=4, pady=3)
        self._rows[job.job_id] = row
        self._update_overall_label()
        self._run_next()

    # ------------------------------------------------------------------
    # Internal job management
    # ------------------------------------------------------------------

    def _run_next(self):
        """Start the next queued job if none is currently running."""
        if self._running_job is not None:
            return
        for job in self._jobs:
            if job.status == "queued":
                self._start_job(job)
                return

    def _start_job(self, job: ExportJob):
        from engine import RenderTask

        self._running_job = job
        job.status   = "running"
        job.progress = 0.0

        def on_progress(pct: float):
            job.progress = float(pct)

        def on_success():
            job.status   = "done"
            job.progress = 100.0
            self._running_job = None
            self.after(0, self._run_next)

        def on_error(msg: str):
            job.status = "failed"
            job.error  = msg
            self._running_job = None
            self.after(0, self._run_next)

        task = RenderTask(
            project_clips=job.project_clips,
            out_path=job.out_path,
            proj_w=job.proj_w,
            proj_h=job.proj_h,
            fps=job.fps,
            codec=job.codec,
            bitrate=job.bitrate,
            on_success=on_success,
            on_error=on_error,
            on_progress=on_progress,
            track_volumes=job.track_volumes,
            track_pans=job.track_pans,
        )
        job._task = task
        task.start()

    def _cancel_job(self, job: ExportJob):
        """Cancel a job — stops the running thread or removes it from the queue."""
        if job.status == "running":
            job.cancel()
            self._running_job = None
            # Give a short delay then proceed to next
            self.after(500, self._run_next)
        elif job.status == "queued":
            job.status = "cancelled"

    def _clear_finished(self):
        """Remove done/failed/cancelled jobs from the visible list."""
        self._jobs = [j for j in self._jobs if j.status in ("queued", "running")]
        self._rebuild_rows()
        self._update_overall_label()

    # ------------------------------------------------------------------
    # "Add Current Project" dialog
    # ------------------------------------------------------------------

    def _add_current_project(self):
        """Build a new ExportJob from the current app state and prompt for output path."""
        app = self._app

        # Detect required app attributes
        project_clips = getattr(app, "project_clips", [])
        if not project_clips:
            messagebox.showwarning("No Clips", "There are no clips in the current project.",
                                   parent=self)
            return

        proj_w  = getattr(app, "proj_w",  1920)
        proj_h  = getattr(app, "proj_h",  1080)
        fps     = getattr(app, "fps",     30)

        # Read export settings from the Inspector Export panel if available
        try:
            res_str = app.exp_res.get()   # e.g. "1920x1080"
            pw, ph  = (int(v) for v in res_str.split('x'))
            proj_w, proj_h = pw, ph
        except Exception:
            pass
        try:
            fps = float(app.exp_fps.get())
        except Exception:
            pass
        codec   = getattr(app.exp_codec, 'get', lambda: "libx264")()
        bitrate = getattr(app, "bitrate", None)
        t_vols  = getattr(app.timeline, 'track_volumes', None) if hasattr(app, 'timeline') else None
        t_pans  = getattr(app.timeline, 'track_pans', None) if hasattr(app, 'timeline') else None

        # Ask for output path
        out_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export to…",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("WebM video", "*.webm"),
                       ("All files", "*.*")],
        )
        if not out_path:
            return

        # Ask for a label
        dlg = _LabelDialog(self)
        self.wait_window(dlg)
        label = dlg.result or f"Export {len(self._jobs) + 1}"

        job = ExportJob(
            job_id=str(uuid.uuid4()),
            label=label,
            project_clips=project_clips,
            out_path=out_path,
            proj_w=proj_w,
            proj_h=proj_h,
            fps=fps,
            codec=codec,
            bitrate=bitrate,
            track_volumes=t_vols,
            track_pans=t_pans,
        )
        self.add_job(job)

    # ------------------------------------------------------------------
    # Polling / refresh
    # ------------------------------------------------------------------

    def _start_poll(self):
        self._poll()

    def _poll(self):
        """Periodically refresh all job rows."""
        for job_id, row in self._rows.items():
            row.refresh()
        self._update_overall_label()
        if self._running_job:
            self._running_lbl.configure(
                text=f"Rendering: {self._running_job.label}  "
                     f"{self._running_job.progress:.1f}%"
            )
        else:
            queued_count = sum(1 for j in self._jobs if j.status == "queued")
            self._running_lbl.configure(
                text=f"{queued_count} job(s) queued" if queued_count else ""
            )
        self.after(self._POLL_MS, self._poll)

    def _update_overall_label(self):
        total = len(self._jobs)
        done  = sum(1 for j in self._jobs if j.status in ("done", "failed", "cancelled"))
        self._overall_lbl.configure(text=f"{done} / {total} jobs done")


# ---------------------------------------------------------------------------
# Small helper: label input dialog
# ---------------------------------------------------------------------------

class _LabelDialog(ctk.CTkToplevel):
    """Tiny dialog to ask for a job label."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Job Label")
        self.geometry("320x120")
        self.resizable(False, False)
        self.grab_set()
        self.result = None

        ctk.CTkLabel(self, text="Label for this export job:").pack(padx=12, pady=(12, 4))
        self._entry = ctk.CTkEntry(self, width=260)
        self._entry.pack(padx=12)
        self._entry.focus()

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=8)
        ctk.CTkButton(btn_row, text="OK", width=80, command=self._ok).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Cancel", width=80,
                      fg_color="gray40", command=self.destroy).pack(side="left", padx=4)
        self.bind("<Return>", lambda _e: self._ok())

    def _ok(self):
        self.result = self._entry.get().strip() or None
        self.destroy()
