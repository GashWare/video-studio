# smart_features.py
"""
UI dialogs and orchestration for AI features.
Opens as top-level dialogs from the application menu.

Depends on:
  - ai_features.py  (WhisperSubtitler, SceneDetector, SmartReframer)
  - customtkinter   (CTkToplevel, CTkButton, etc.)
"""

import os
import threading
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

from ai_features import WhisperSubtitler, SceneDetector, SmartReframer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _unavailable_dialog(parent, package_name: str, pip_cmd: str):
    """
    Shows a simple unavailability notice inside a CTkToplevel.
    Returns the toplevel so callers can decide whether to show more UI.
    """
    win = ctk.CTkToplevel(parent)
    win.title(f"Package Missing: {package_name}")
    win.geometry("460x180")
    win.resizable(False, False)

    ctk.CTkLabel(
        win,
        text=f"⚠  {package_name} is not installed.",
        font=("Segoe UI", 14, "bold"),
    ).pack(pady=(24, 4))
    ctk.CTkLabel(
        win,
        text=f"Install it from a terminal with:\n{pip_cmd}",
        font=("Segoe UI", 12),
        justify="center",
    ).pack(pady=4)
    ctk.CTkButton(win, text="Close", width=100, command=win.destroy).pack(pady=16)
    return win


# ---------------------------------------------------------------------------
# SubtitleDialog
# ---------------------------------------------------------------------------

class SubtitleDialog(ctk.CTkToplevel):
    """
    Dialog for auto-generating subtitles via Whisper.
    Shows model selector, language selector, and a progress bar.
    On completion: 'Insert Subtitles' appends title clips to app.project_clips.
    """

    _MODELS = ["tiny", "base", "small", "medium", "large"]
    _LANGUAGES = [
        "Auto-detect", "English", "Spanish", "French", "German",
        "Italian", "Portuguese", "Russian", "Japanese", "Chinese",
        "Korean", "Arabic", "Hindi", "Turkish", "Dutch", "Polish",
    ]
    _LANG_CODES = {
        "Auto-detect": None, "English": "en", "Spanish": "es",
        "French": "fr", "German": "de", "Italian": "it",
        "Portuguese": "pt", "Russian": "ru", "Japanese": "ja",
        "Chinese": "zh", "Korean": "ko", "Arabic": "ar",
        "Hindi": "hi", "Turkish": "tr", "Dutch": "nl", "Polish": "pl",
    }

    def __init__(self, app, clip: dict):
        super().__init__(app)
        self.app = app
        self.clip = clip
        self._segments = []

        self.title("Auto-Generate Subtitles (Whisper)")
        self.geometry("500x400")
        self.resizable(False, False)

        if not WhisperSubtitler.is_available():
            self.destroy()
            _unavailable_dialog(
                app,
                "openai-whisper",
                "pip install openai-whisper",
            )
            return

        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 20, "pady": 6}

        ctk.CTkLabel(
            self, text="🎙 Auto-Generate Subtitles",
            font=("Segoe UI", 16, "bold"),
        ).pack(**pad, pady=(20, 4))

        clip_name = os.path.basename(self.clip.get("path", "clip"))
        ctk.CTkLabel(
            self, text=f"Clip: {clip_name}",
            font=("Segoe UI", 11), text_color="gray",
        ).pack(padx=20, pady=(0, 10))

        # Model selector
        row1 = ctk.CTkFrame(self, fg_color="transparent")
        row1.pack(fill="x", **pad)
        ctk.CTkLabel(row1, text="Model:", width=80, anchor="w").pack(side="left")
        self._model_var = ctk.StringVar(value="base")
        ctk.CTkComboBox(
            row1, variable=self._model_var,
            values=self._MODELS, width=160,
        ).pack(side="left", padx=6)

        # Language selector
        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(fill="x", **pad)
        ctk.CTkLabel(row2, text="Language:", width=80, anchor="w").pack(side="left")
        self._lang_var = ctk.StringVar(value="Auto-detect")
        ctk.CTkComboBox(
            row2, variable=self._lang_var,
            values=self._LANGUAGES, width=160,
        ).pack(side="left", padx=6)

        # Progress bar + status label
        self._status_lbl = ctk.CTkLabel(self, text="Ready.", text_color="gray")
        self._status_lbl.pack(**pad)
        self._progress = ctk.CTkProgressBar(self, mode="indeterminate", width=380)
        self._progress.pack(padx=20, pady=4)
        self._progress.stop()

        # Result label (hidden until done)
        self._result_lbl = ctk.CTkLabel(self, text="", text_color="#2ecc71")
        self._result_lbl.pack(padx=20, pady=2)

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=16)
        self._run_btn = ctk.CTkButton(
            btn_frame, text="▶ Run Whisper",
            command=self._run, fg_color="teal",
        )
        self._run_btn.pack(side="left", padx=4)
        self._insert_btn = ctk.CTkButton(
            btn_frame, text="⬇ Insert Subtitles",
            command=self._insert, state="disabled",
        )
        self._insert_btn.pack(side="left", padx=4)
        ctk.CTkButton(
            btn_frame, text="Close",
            command=self.destroy, fg_color="gray",
        ).pack(side="right", padx=4)

    def _run(self):
        self._run_btn.configure(state="disabled")
        self._insert_btn.configure(state="disabled")
        self._result_lbl.configure(text="")
        self._status_lbl.configure(text="Transcribing… this may take a moment.")
        self._progress.configure(mode="indeterminate")
        self._progress.start()

        model_size = self._model_var.get()
        lang_name = self._lang_var.get()
        language = self._LANG_CODES.get(lang_name)
        media_path = self.clip.get("path", "")

        def worker():
            try:
                segments = WhisperSubtitler.transcribe(
                    media_path, model_size=model_size, language=language
                )
                self.after(0, lambda: self._on_done(segments))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, segments):
        self._segments = segments
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(1.0)
        n = len(segments)
        self._status_lbl.configure(text="Transcription complete.", text_color="#2ecc71")
        self._result_lbl.configure(
            text=f"Found {n} subtitle segment{'s' if n != 1 else ''}. "
                 f"Click 'Insert Subtitles' to add them to the timeline."
        )
        self._run_btn.configure(state="normal")
        if n > 0:
            self._insert_btn.configure(state="normal")

    def _on_error(self, msg):
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(0)
        self._status_lbl.configure(text=f"Error: {msg}", text_color="#e74c3c")
        self._run_btn.configure(state="normal")

    def _insert(self):
        if not self._segments:
            return

        # Find existing subtitle / title track or create S1
        target_track = None
        if hasattr(self.app, "timeline") and hasattr(self.app.timeline, "track_keys"):
            for tk in self.app.timeline.track_keys:
                if tk.startswith("S") or tk.startswith("T"):
                    target_track = tk
                    break
            if target_track is None:
                if hasattr(self.app.timeline, "add_subtitle_track"):
                    self.app.timeline.add_subtitle_track()
                    target_track = "S1"
                else:
                    target_track = "S1"
        else:
            target_track = "S1"

        title_clips = WhisperSubtitler.segments_to_title_clips(self._segments, track=target_track)
        for tc in title_clips:
            self.app.project_clips.append(tc)
        # Trigger timeline refresh
        try:
            self.app.rebuild_proxy_cache()
            self.app.timeline.draw()
        except Exception:
            pass
        n = len(title_clips)
        self._result_lbl.configure(
            text=f"✔ {n} subtitle clip{'s' if n != 1 else ''} inserted into track {target_track}.",
            text_color="#2ecc71",
        )
        self._insert_btn.configure(state="disabled")


# ---------------------------------------------------------------------------
# SceneDetectDialog
# ---------------------------------------------------------------------------

class SceneDetectDialog(ctk.CTkToplevel):
    """
    Dialog for auto-splitting a clip at scene changes.
    """

    def __init__(self, app, clip: dict):
        super().__init__(app)
        self.app = app
        self.clip = clip
        self._scene_times = []

        self.title("Auto-Split at Scene Changes")
        self.geometry("480x420")
        self.resizable(False, False)

        if not SceneDetector.is_available():
            self.destroy()
            _unavailable_dialog(
                app,
                "PySceneDetect",
                "pip install scenedetect[opencv]",
            )
            return

        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 20, "pady": 6}

        ctk.CTkLabel(
            self, text="🎬 Auto-Split at Scene Changes",
            font=("Segoe UI", 16, "bold"),
        ).pack(**pad, pady=(20, 4))

        clip_name = os.path.basename(self.clip.get("path", "clip"))
        ctk.CTkLabel(
            self, text=f"Clip: {clip_name}",
            font=("Segoe UI", 11), text_color="gray",
        ).pack(padx=20, pady=(0, 6))

        # Threshold slider
        thr_frame = ctk.CTkFrame(self, fg_color="transparent")
        thr_frame.pack(fill="x", **pad)
        ctk.CTkLabel(thr_frame, text="Threshold:", width=90, anchor="w").pack(side="left")
        self._thr_lbl = ctk.CTkLabel(thr_frame, text="27", width=36)
        self._thr_lbl.pack(side="right")
        self._thr_var = ctk.DoubleVar(value=27.0)
        ctk.CTkSlider(
            thr_frame, variable=self._thr_var,
            from_=10, to=60,
            command=lambda v: self._thr_lbl.configure(text=f"{int(v)}"),
        ).pack(side="left", fill="x", expand=True, padx=6)

        # Progress + status
        self._status_lbl = ctk.CTkLabel(self, text="Ready.", text_color="gray")
        self._status_lbl.pack(**pad)
        self._progress = ctk.CTkProgressBar(self, mode="indeterminate", width=380)
        self._progress.pack(padx=20, pady=4)
        self._progress.stop()

        # Scene list
        list_frame = ctk.CTkFrame(self, fg_color="transparent")
        list_frame.pack(fill="both", expand=True, padx=20, pady=4)
        ctk.CTkLabel(list_frame, text="Detected scene changes (seconds):").pack(anchor="w")
        self._listbox = tk.Listbox(
            list_frame, bg="#2b2b2b", fg="white",
            selectbackground="#1f538d", height=5,
        )
        self._listbox.pack(fill="both", expand=True, pady=4)

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=12)
        self._detect_btn = ctk.CTkButton(
            btn_frame, text="🔍 Detect Scenes",
            command=self._detect, fg_color="teal",
        )
        self._detect_btn.pack(side="left", padx=4)
        self._split_btn = ctk.CTkButton(
            btn_frame, text="✂ Split All",
            command=self._split_all, state="disabled",
        )
        self._split_btn.pack(side="left", padx=4)
        ctk.CTkButton(
            btn_frame, text="Close",
            command=self.destroy, fg_color="gray",
        ).pack(side="right", padx=4)

    def _detect(self):
        self._detect_btn.configure(state="disabled")
        self._split_btn.configure(state="disabled")
        self._listbox.delete(0, "end")
        self._status_lbl.configure(text="Analysing video…", text_color="gray")
        self._progress.configure(mode="indeterminate")
        self._progress.start()

        video_path = self.clip.get("path", "")
        threshold = float(self._thr_var.get())

        def worker():
            try:
                times = SceneDetector.detect_scenes(video_path, threshold=threshold)
                self.after(0, lambda: self._on_done(times))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, times):
        self._scene_times = times
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(1.0)
        n = len(times)
        self._status_lbl.configure(
            text=f"Found {n} scene change{'s' if n != 1 else ''}.",
            text_color="#2ecc71",
        )
        self._listbox.delete(0, "end")
        for t in times:
            self._listbox.insert("end", f"  {t:.3f} s")
        self._detect_btn.configure(state="normal")
        if n > 0:
            self._split_btn.configure(state="normal")

    def _on_error(self, msg):
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(0)
        self._status_lbl.configure(text=f"Error: {msg}", text_color="#e74c3c")
        self._detect_btn.configure(state="normal")

    def _split_all(self):
        if not self._scene_times:
            return
        sub_clips = SceneDetector.auto_split_clip(self.clip, self._scene_times)
        if len(sub_clips) <= 1:
            self._status_lbl.configure(
                text="No usable split points within clip.", text_color="orange"
            )
            return

        # Replace the original clip with the split sub-clips
        clips = self.app.project_clips
        idx = next((i for i, c in enumerate(clips) if c["id"] == self.clip["id"]), None)
        if idx is not None:
            clips[idx: idx + 1] = sub_clips
        else:
            clips.extend(sub_clips)

        try:
            self.app.rebuild_proxy_cache()
            self.app.timeline.draw()
        except Exception:
            pass

        n = len(sub_clips)
        self._status_lbl.configure(
            text=f"✔ Clip replaced with {n} sub-clips.",
            text_color="#2ecc71",
        )
        self._split_btn.configure(state="disabled")


# ---------------------------------------------------------------------------
# SmartReframeDialog
# ---------------------------------------------------------------------------

class SmartReframeDialog(ctk.CTkToplevel):
    """
    Dialog for AI smart reframe (9:16 crop tracking via MediaPipe).
    """

    _RATIOS = {
        "9:16 (Portrait)": (9, 16),
        "1:1 (Square)":    (1, 1),
        "4:3 (Classic)":   (4, 3),
    }

    def __init__(self, app, clip: dict):
        super().__init__(app)
        self.app = app
        self.clip = clip
        self._crop_path = []
        self._preview_image = None  # Keep reference to avoid GC

        self.title("AI Smart Reframe")
        self.geometry("540x480")
        self.resizable(False, False)

        if not SmartReframer.is_available():
            self.destroy()
            _unavailable_dialog(
                app,
                "mediapipe",
                "pip install mediapipe opencv-python",
            )
            return

        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 20, "pady": 6}

        ctk.CTkLabel(
            self, text="🤖 AI Smart Reframe",
            font=("Segoe UI", 16, "bold"),
        ).pack(**pad, pady=(20, 4))

        clip_name = os.path.basename(self.clip.get("path", "clip"))
        ctk.CTkLabel(
            self, text=f"Clip: {clip_name}",
            font=("Segoe UI", 11), text_color="gray",
        ).pack(padx=20, pady=(0, 6))

        # Target ratio
        ratio_frame = ctk.CTkFrame(self, fg_color="transparent")
        ratio_frame.pack(fill="x", **pad)
        ctk.CTkLabel(ratio_frame, text="Target ratio:", width=100, anchor="w").pack(side="left")
        self._ratio_var = ctk.StringVar(value="9:16 (Portrait)")
        ctk.CTkComboBox(
            ratio_frame, variable=self._ratio_var,
            values=list(self._RATIOS.keys()), width=200,
        ).pack(side="left", padx=6)

        # Progress + status
        self._status_lbl = ctk.CTkLabel(self, text="Ready.", text_color="gray")
        self._status_lbl.pack(**pad)
        self._progress = ctk.CTkProgressBar(self, mode="indeterminate", width=420)
        self._progress.pack(padx=20, pady=4)
        self._progress.stop()

        # Preview canvas (first frame + crop overlay)
        preview_frame = ctk.CTkFrame(self, fg_color="#111111", corner_radius=6)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=6)
        self._canvas = tk.Canvas(
            preview_frame, bg="#111111", highlightthickness=0,
            width=380, height=160,
        )
        self._canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self._draw_first_frame()

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=12)
        self._analyse_btn = ctk.CTkButton(
            btn_frame, text="🔍 Analyse",
            command=self._analyse, fg_color="teal",
        )
        self._analyse_btn.pack(side="left", padx=4)
        self._apply_btn = ctk.CTkButton(
            btn_frame, text="✅ Apply & Export",
            command=self._apply, state="disabled",
        )
        self._apply_btn.pack(side="left", padx=4)
        ctk.CTkButton(
            btn_frame, text="Close",
            command=self.destroy, fg_color="gray",
        ).pack(side="right", padx=4)

    # ------------------------------------------------------------------

    def _target_size(self):
        """Return (target_w, target_h) based on selected ratio and source height."""
        ratio_str = self._ratio_var.get()
        rw, rh = self._RATIOS.get(ratio_str, (9, 16))
        # Derive from clip height (assume 1080 if unknown)
        src_h = 1080
        try:
            import cv2
            cap = cv2.VideoCapture(self.clip.get("path", ""))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            cap.release()
        except Exception:
            pass
        target_h = src_h
        target_w = int(target_h * rw / rh)
        return target_w, target_h

    def _draw_first_frame(self):
        """Show the first frame of the clip on the preview canvas."""
        try:
            import cv2
            cap = cv2.VideoCapture(self.clip.get("path", ""))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return
            import numpy as np
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            # Fit into canvas
            pil_img.thumbnail((380, 160), Image.Resampling.BILINEAR)
            self._preview_image = ImageTk.PhotoImage(pil_img)
            self._canvas.create_image(0, 0, anchor="nw", image=self._preview_image)
        except Exception:
            pass

    def _draw_crop_overlay(self, cx: float, cy: float, target_w: int, target_h: int, src_w: int, src_h: int):
        """Overlay the crop rectangle on the preview canvas."""
        # Scale to canvas size
        canvas_w = self._canvas.winfo_width() or 380
        canvas_h = self._canvas.winfo_height() or 160
        scale_x = canvas_w / src_w
        scale_y = canvas_h / src_h

        # Centre of crop in source pixels
        cx_px = cx * src_w
        cy_px = cy * src_h
        x1 = (cx_px - target_w / 2) * scale_x
        y1 = (cy_px - target_h / 2) * scale_y
        x2 = (cx_px + target_w / 2) * scale_x
        y2 = (cy_px + target_h / 2) * scale_y

        self._canvas.delete("crop_rect")
        self._canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="#e74c3c", width=2, tags="crop_rect",
        )

    def _analyse(self):
        self._analyse_btn.configure(state="disabled")
        self._apply_btn.configure(state="disabled")
        self._status_lbl.configure(text="Analysing frames…", text_color="gray")
        self._progress.configure(mode="indeterminate")
        self._progress.start()

        video_path = self.clip.get("path", "")
        target_w, target_h = self._target_size()

        def worker():
            try:
                crop_path = SmartReframer.compute_crop_path(
                    video_path, target_w, target_h, sample_every_n_frames=5
                )
                self.after(0, lambda: self._on_analyse_done(crop_path, target_w, target_h))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_analyse_done(self, crop_path, target_w, target_h):
        self._crop_path = crop_path
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(1.0)
        n = len(crop_path)
        self._status_lbl.configure(
            text=f"Analysis complete ({n} keyframes). Showing first crop position.",
            text_color="#2ecc71",
        )
        self._analyse_btn.configure(state="normal")
        if crop_path:
            self._apply_btn.configure(state="normal")
            # Show crop overlay for first keyframe
            try:
                import cv2
                cap = cv2.VideoCapture(self.clip.get("path", ""))
                src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                cap.release()
            except Exception:
                src_w, src_h = 1920, 1080
            first = crop_path[0]
            self._draw_crop_overlay(first["cx"], first["cy"], target_w, target_h, src_w, src_h)

    def _on_error(self, msg):
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(0)
        self._status_lbl.configure(text=f"Error: {msg}", text_color="#e74c3c")
        self._analyse_btn.configure(state="normal")

    def _apply(self):
        if not self._crop_path:
            return

        from tkinter import filedialog
        out_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            initialfile="reframed.mp4",
            filetypes=[("MP4 Video", "*.mp4")],
            parent=self,
        )
        if not out_path:
            return

        self._apply_btn.configure(state="disabled")
        self._analyse_btn.configure(state="disabled")
        self._status_lbl.configure(text="Rendering reframed video…", text_color="gray")
        self._progress.configure(mode="indeterminate")
        self._progress.start()

        video_path = self.clip.get("path", "")
        target_w, target_h = self._target_size()
        crop_path = list(self._crop_path)

        def worker():
            try:
                SmartReframer.apply_crop_path(
                    video_path, crop_path, target_w, target_h, out_path
                )
                self.after(0, lambda: self._on_apply_done(out_path))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_apply_done(self, out_path):
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._progress.set(1.0)
        self._status_lbl.configure(
            text=f"✔ Saved: {os.path.basename(out_path)}",
            text_color="#2ecc71",
        )
        self._analyse_btn.configure(state="normal")
        self._apply_btn.configure(state="normal")

        # Optionally add the new reframed clip to the media bin
        try:
            import uuid as _uuid
            if out_path not in self.app.media_bin:
                self.app.media_bin.append(out_path)
                self.app.media_listbox.insert("end", os.path.basename(out_path))
        except Exception:
            pass
