# title_generator.py
"""
Animated title-clip generator using Pillow.
Title clips are stored in project_clips with track='T1', path='__title__'.
"""

import uuid
import math
import tkinter as tk
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

ANIMATION_TYPES = ["None", "Fade In", "Fade Out", "Fade In/Out", "Slide Up", "Slide Down"]
ALIGNMENT_TYPES = ["left", "center", "right"]


# ---------------------------------------------------------------------------
# TitleClip
# ---------------------------------------------------------------------------

class TitleClip:
    """
    Generates animated title frames using Pillow.
    Stored in project_clips with track='T1', path='__title__'.
    """

    def __init__(
        self,
        text: str,
        duration: float,
        fps: float,
        width: int,
        height: int,
        font_size: int = 64,
        font_color: tuple = (255, 255, 255),
        bg_color: tuple = (0, 0, 0, 0),    # RGBA – transparent by default
        alignment: str = "center",
        animation: str = "Fade In",
        font_path: str = None,
    ):
        self.text       = text
        self.duration   = float(duration)
        self.fps        = float(fps)
        self.width      = int(width)
        self.height     = int(height)
        self.font_size  = int(font_size)
        self.font_color = tuple(font_color)
        self.bg_color   = tuple(bg_color)
        self.alignment  = alignment
        self.animation  = animation
        self.font_path  = font_path

        self._font = self._load_font()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_font(self) -> ImageFont.FreeTypeFont:
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, self.font_size)
            except Exception:
                pass
        # Try some common system fonts as fallback
        for fallback in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                          "LiberationSans-Regular.ttf"]:
            try:
                return ImageFont.truetype(fallback, self.font_size)
            except Exception:
                pass
        # Ultimate fallback: Pillow default bitmap font (no size param)
        return ImageFont.load_default()

    def _alpha_for_time(self, t: float) -> float:
        """Return opacity [0.0, 1.0] at time *t* based on the animation type."""
        anim = self.animation
        d = self.duration
        fade_dur = min(0.5, d * 0.25)   # fade over 0.25 s or 25% of clip, whichever shorter

        if anim == "None":
            return 1.0
        if anim == "Fade In":
            return min(1.0, t / fade_dur) if fade_dur > 0 else 1.0
        if anim == "Fade Out":
            return min(1.0, (d - t) / fade_dur) if fade_dur > 0 else 1.0
        if anim == "Fade In/Out":
            alpha_in  = min(1.0, t / fade_dur) if fade_dur > 0 else 1.0
            alpha_out = min(1.0, (d - t) / fade_dur) if fade_dur > 0 else 1.0
            return min(alpha_in, alpha_out)
        # Slide Up / Down: full opacity, position handled in _render_frame
        return 1.0

    def _slide_offset(self, t: float) -> int:
        """Return the Y pixel offset for slide animations."""
        anim = self.animation
        d = self.duration
        slide_dur = min(0.5, d * 0.25)
        H = self.height

        if anim == "Slide Up":
            progress = min(1.0, t / slide_dur) if slide_dur > 0 else 1.0
            # ease out: sqrt
            progress = math.sqrt(progress)
            return int((1.0 - progress) * H // 3)
        if anim == "Slide Down":
            progress = min(1.0, t / slide_dur) if slide_dur > 0 else 1.0
            progress = math.sqrt(progress)
            return -int((1.0 - progress) * H // 3)
        return 0

    def _render_frame(self, t: float) -> Image.Image:
        """Render a single RGBA Pillow image at time *t*."""
        # Ensure bg_color has 4 channels
        bg = self.bg_color if len(self.bg_color) == 4 else (*self.bg_color, 255)
        img = Image.new("RGBA", (self.width, self.height), bg)
        draw = ImageDraw.Draw(img)

        alpha_frac = self._alpha_for_time(t)
        alpha_byte = int(alpha_frac * 255)
        fc = self.font_color
        color_with_alpha = (fc[0], fc[1], fc[2], alpha_byte) if len(fc) == 3 else \
                           (fc[0], fc[1], fc[2], int(fc[3] * alpha_frac))

        # Text bounding box for centering
        bbox = draw.textbbox((0, 0), self.text, font=self._font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        align_map = {
            "left":   10,
            "center": (self.width - text_w) // 2,
            "right":  self.width - text_w - 10,
        }
        x = align_map.get(self.alignment, (self.width - text_w) // 2)
        y = (self.height - text_h) // 2 + self._slide_offset(t)

        draw.text((x, y), self.text, font=self._font, fill=color_with_alpha)
        return img

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self, t: float) -> np.ndarray:
        """Return RGB uint8 ndarray (H, W, 3) at time *t*."""
        t = max(0.0, min(float(t), self.duration))
        img = self._render_frame(t)
        # Composite over black if background is transparent
        bg_alpha = self.bg_color[3] if len(self.bg_color) == 4 else 255
        if bg_alpha < 255:
            base = Image.new("RGBA", img.size, (0, 0, 0, 255))
            base = Image.alpha_composite(base, img)
            img = base
        return np.array(img.convert("RGB"), dtype=np.uint8)

    def get_all_frames(self) -> list:
        """Pre-render every frame. Returns a list of RGB uint8 ndarrays."""
        n_frames = max(1, round(self.duration * self.fps))
        frames = []
        for i in range(n_frames):
            t = i / self.fps
            frames.append(self.get_frame(t))
        return frames

    def to_moviepy_clip(self):
        """Convert TitleClip to a MoviePy VideoClip with alpha mask."""
        try:
            from moviepy.editor import VideoClip
        except ImportError:
            from moviepy.video.VideoClip import VideoClip

        def make_frame(t):
            t_clamped = max(0.0, min(float(t), self.duration))
            img = self._render_frame(t_clamped)
            return np.array(img.convert("RGB"), dtype=np.uint8)

        def make_mask(t):
            t_clamped = max(0.0, min(float(t), self.duration))
            img = self._render_frame(t_clamped)
            alpha = np.array(img.split()[-1], dtype=np.float32) / 255.0
            return alpha

        clip = VideoClip(make_frame, duration=self.duration)
        mask = VideoClip(make_mask, duration=self.duration, ismask=True)
        clip = clip.set_mask(mask)
        clip.fps = self.fps
        return clip

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_config(self) -> dict:
        return {
            "text":       self.text,
            "duration":   self.duration,
            "fps":        self.fps,
            "width":      self.width,
            "height":     self.height,
            "font_size":  self.font_size,
            "font_color": list(self.font_color),
            "bg_color":   list(self.bg_color),
            "alignment":  self.alignment,
            "animation":  self.animation,
            "font_path":  self.font_path,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "TitleClip":
        return cls(
            text=cfg["text"],
            duration=cfg["duration"],
            fps=cfg["fps"],
            width=cfg["width"],
            height=cfg["height"],
            font_size=cfg.get("font_size", 64),
            font_color=tuple(cfg.get("font_color", [255, 255, 255])),
            bg_color=tuple(cfg.get("bg_color", [0, 0, 0, 0])),
            alignment=cfg.get("alignment", "center"),
            animation=cfg.get("animation", "Fade In"),
            font_path=cfg.get("font_path"),
        )


def make_title_clip_dict(
    title_clip: TitleClip,
    start: float = 0.0,
    clip_id: str = None,
) -> dict:
    """Create the project_clips dict entry for a title clip."""
    return {
        "id":           clip_id or str(uuid.uuid4()),
        "path":         "__title__",
        "track":        "T1",
        "start":        start,
        "trim_in":      0.0,
        "trim_out":     title_clip.duration,
        "title_config": title_clip.to_config(),
    }


# ---------------------------------------------------------------------------
# TitleDialog
# ---------------------------------------------------------------------------

class TitleDialog(ctk.CTkToplevel):
    """
    Dialog for creating or editing a title clip.
    Opens from the main app (e.g. from a menu or toolbar button).

    On save, appends or updates a clip in ``app.project_clips`` with track='T1'.
    """

    _PREVIEW_W = 400
    _PREVIEW_H = 225

    def __init__(self, app, existing_clip: dict = None):
        super().__init__(app)
        self.title("Title Clip Editor")
        self.resizable(False, False)
        self.grab_set()

        self._app          = app
        self._existing     = existing_clip
        self._preview_img  = None   # hold reference to avoid GC

        # Detect project dimensions from the app if available
        self._proj_w = getattr(app, "proj_w", 1920)
        self._proj_h = getattr(app, "proj_h", 1080)
        self._fps    = getattr(app, "fps",    30)

        self._build_ui()
        if existing_clip:
            self._load_existing(existing_clip)
        self._update_preview()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # Left column: controls
        left = ctk.CTkFrame(self)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Text
        ctk.CTkLabel(left, text="Text", anchor="w").grid(row=0, column=0, **pad, sticky="w")
        self._txt_entry = ctk.CTkEntry(left, width=260)
        self._txt_entry.insert(0, "Title Text")
        self._txt_entry.grid(row=0, column=1, **pad)
        self._txt_entry.bind("<KeyRelease>", lambda _e: self._update_preview())

        # Duration
        ctk.CTkLabel(left, text="Duration (s)", anchor="w").grid(row=1, column=0, **pad, sticky="w")
        self._dur_entry = ctk.CTkEntry(left, width=80)
        self._dur_entry.insert(0, "3.0")
        self._dur_entry.grid(row=1, column=1, **pad, sticky="w")
        self._dur_entry.bind("<KeyRelease>", lambda _e: self._update_preview())

        # Start time
        ctk.CTkLabel(left, text="Start (s)", anchor="w").grid(row=2, column=0, **pad, sticky="w")
        self._start_entry = ctk.CTkEntry(left, width=80)
        self._start_entry.insert(0, "0.0")
        self._start_entry.grid(row=2, column=1, **pad, sticky="w")

        # Font size
        ctk.CTkLabel(left, text="Font Size", anchor="w").grid(row=3, column=0, **pad, sticky="w")
        self._font_size_var = tk.IntVar(value=64)
        ctk.CTkSlider(left, from_=12, to=200, number_of_steps=188,
                      variable=self._font_size_var,
                      command=lambda _v: self._update_preview()).grid(
            row=3, column=1, **pad)

        # Font color (hex entry)
        ctk.CTkLabel(left, text="Font Color (#RRGGBB)", anchor="w").grid(row=4, column=0, **pad, sticky="w")
        self._font_color_entry = ctk.CTkEntry(left, width=100)
        self._font_color_entry.insert(0, "#ffffff")
        self._font_color_entry.grid(row=4, column=1, **pad, sticky="w")
        self._font_color_entry.bind("<KeyRelease>", lambda _e: self._update_preview())

        # Background color
        ctk.CTkLabel(left, text="BG Color (#RRGGBBAA)", anchor="w").grid(row=5, column=0, **pad, sticky="w")
        self._bg_color_entry = ctk.CTkEntry(left, width=100)
        self._bg_color_entry.insert(0, "#00000000")
        self._bg_color_entry.grid(row=5, column=1, **pad, sticky="w")
        self._bg_color_entry.bind("<KeyRelease>", lambda _e: self._update_preview())

        # Alignment
        ctk.CTkLabel(left, text="Alignment", anchor="w").grid(row=6, column=0, **pad, sticky="w")
        self._align_combo = ctk.CTkComboBox(left, values=ALIGNMENT_TYPES, width=120,
                                             command=lambda _v: self._update_preview())
        self._align_combo.set("center")
        self._align_combo.grid(row=6, column=1, **pad, sticky="w")

        # Animation
        ctk.CTkLabel(left, text="Animation", anchor="w").grid(row=7, column=0, **pad, sticky="w")
        self._anim_combo = ctk.CTkComboBox(left, values=ANIMATION_TYPES, width=140,
                                            command=lambda _v: self._update_preview())
        self._anim_combo.set("Fade In")
        self._anim_combo.grid(row=7, column=1, **pad, sticky="w")

        # Font path (optional)
        ctk.CTkLabel(left, text="Font Path (optional)", anchor="w").grid(row=8, column=0, **pad, sticky="w")
        self._font_path_entry = ctk.CTkEntry(left, width=200)
        self._font_path_entry.grid(row=8, column=1, **pad, sticky="w")
        self._font_path_entry.bind("<KeyRelease>", lambda _e: self._update_preview())

        # Right column: preview + buttons
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        ctk.CTkLabel(right, text="Preview", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(4, 2))
        self._preview_canvas = tk.Canvas(
            right, width=self._PREVIEW_W, height=self._PREVIEW_H,
            bg="#000000", highlightthickness=1, highlightbackground="#555"
        )
        self._preview_canvas.pack(padx=8, pady=4)

        # Preview time scrubber
        self._prev_time_var = tk.DoubleVar(value=0.0)
        ctk.CTkSlider(right, from_=0.0, to=1.0, number_of_steps=100,
                      variable=self._prev_time_var,
                      command=lambda _v: self._update_preview()).pack(
            fill="x", padx=8, pady=2)

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.pack(pady=6)
        ctk.CTkButton(btn_row, text="Save", command=self._save, width=100).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Cancel", command=self.destroy,
                      fg_color="gray40", width=100).pack(side="left", padx=4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_hex_color(self, hex_str: str, default) -> tuple:
        hex_str = hex_str.strip().lstrip("#")
        try:
            if len(hex_str) == 6:
                r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
                return (r, g, b)
            if len(hex_str) == 8:
                r, g, b, a = (int(hex_str[i*2:(i+1)*2], 16) for i in range(4))
                return (r, g, b, a)
        except ValueError:
            pass
        return default

    def _build_title_clip(self) -> TitleClip:
        text      = self._txt_entry.get() or "Title"
        try:
            duration = float(self._dur_entry.get())
        except ValueError:
            duration = 3.0
        font_color = self._parse_hex_color(self._font_color_entry.get(), (255, 255, 255))
        bg_color   = self._parse_hex_color(self._bg_color_entry.get(), (0, 0, 0, 0))
        font_path  = self._font_path_entry.get().strip() or None
        return TitleClip(
            text=text,
            duration=duration,
            fps=self._fps,
            width=self._proj_w,
            height=self._proj_h,
            font_size=self._font_size_var.get(),
            font_color=font_color,
            bg_color=bg_color,
            alignment=self._align_combo.get(),
            animation=self._anim_combo.get(),
            font_path=font_path,
        )

    def _update_preview(self, *_args):
        try:
            tc = self._build_title_clip()
            frac = self._prev_time_var.get()
            t = frac * tc.duration
            frame = tc.get_frame(t)                        # RGB uint8 (H, W, 3)
            from PIL import Image as _Img
            pil = _Img.fromarray(frame).resize(
                (self._PREVIEW_W, self._PREVIEW_H), _Img.LANCZOS
            )
            self._preview_img = ImageTk.PhotoImage(pil)
            self._preview_canvas.create_image(0, 0, anchor="nw", image=self._preview_img)
        except Exception:
            pass

    def _load_existing(self, clip: dict):
        cfg = clip.get("title_config", {})
        self._txt_entry.delete(0, "end")
        self._txt_entry.insert(0, cfg.get("text", "Title"))
        self._dur_entry.delete(0, "end")
        self._dur_entry.insert(0, str(cfg.get("duration", 3.0)))
        self._start_entry.delete(0, "end")
        self._start_entry.insert(0, str(clip.get("start", 0.0)))
        self._font_size_var.set(cfg.get("font_size", 64))
        fc = cfg.get("font_color", [255, 255, 255])
        self._font_color_entry.delete(0, "end")
        self._font_color_entry.insert(0, "#{:02x}{:02x}{:02x}".format(*fc[:3]))
        bc = cfg.get("bg_color", [0, 0, 0, 0])
        self._bg_color_entry.delete(0, "end")
        if len(bc) == 4:
            self._bg_color_entry.insert(0, "#{:02x}{:02x}{:02x}{:02x}".format(*bc))
        else:
            self._bg_color_entry.insert(0, "#{:02x}{:02x}{:02x}".format(*bc[:3]))
        self._align_combo.set(cfg.get("alignment", "center"))
        self._anim_combo.set(cfg.get("animation", "Fade In"))
        fp = cfg.get("font_path", "")
        if fp:
            self._font_path_entry.insert(0, fp)

    def _save(self):
        try:
            start = float(self._start_entry.get())
        except ValueError:
            start = 0.0
        tc = self._build_title_clip()

        if self._existing:
            # Update in-place
            self._existing.update(make_title_clip_dict(tc, start=start,
                                                        clip_id=self._existing.get("id")))
        else:
            # Append new clip
            new_clip = make_title_clip_dict(tc, start=start)
            if hasattr(self._app, "project_clips"):
                self._app.project_clips.append(new_clip)
            # Notify app of change if it has a refresh method
            for method_name in ("_refresh_timeline", "refresh_timeline", "_update_timeline"):
                if hasattr(self._app, method_name):
                    getattr(self._app, method_name)()
                    break

        self.destroy()
