# video_scopes.py
"""
Real-time Video Scopes for PyDev NLE & Colorist Suite.
Provides:
  - RGB Parade (Individual R, G, B channel column waveforms)
  - Luma Waveform (0-100 IRE exposure density)
  - Vectorscope (Cb/Cr color-wheel polar diagram with skin-tone line)
  - Histogram (Multi-channel RGB luminance distribution)
"""

import math
import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk


def compute_luma_waveform(rgb_frame: np.ndarray, out_w: int = 300, out_h: int = 200) -> np.ndarray:
    """
    Computes a Luma Waveform monitor frame (0-100 IRE).
    Input: RGB uint8 frame
    Output: RGB uint8 waveform image
    """
    if rgb_frame is None or rgb_frame.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Downsample for ultra-fast real-time rendering
    small = cv2.resize(rgb_frame, (out_w, 120), interpolation=cv2.INTER_AREA)
    # ITU-R BT.709 Luma: Y = 0.2126 R + 0.7152 G + 0.0722 B
    y_luma = (0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2]).astype(np.float32)
    # Normalize to 0 .. out_h - 1 (inverted: 255 at top, 0 at bottom)
    y_pos = np.clip((1.0 - (y_luma / 255.0)) * (out_h - 1), 0, out_h - 1).astype(np.int32)

    scope = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    # Draw graticule lines at 0%, 25%, 50%, 75%, 100% IRE
    for ire in (0.0, 0.25, 0.50, 0.75, 1.0):
        y_line = int((1.0 - ire) * (out_h - 1))
        scope[y_line, :, :] = (45, 55, 65)

    # Accumulate waveform points (phosphor green tint)
    for col in range(out_w):
        y_pts = y_pos[:, col]
        for y in y_pts:
            curr = scope[y, col]
            # Add phosphor glow
            scope[y, col, 0] = min(255, int(curr[0]) + 25)
            scope[y, col, 1] = min(255, int(curr[1]) + 75)
            scope[y, col, 2] = min(255, int(curr[2]) + 45)

    return scope


def compute_rgb_parade(rgb_frame: np.ndarray, out_w: int = 360, out_h: int = 200) -> np.ndarray:
    """
    Computes an RGB Parade monitor frame with separate R, G, B channels side-by-side.
    """
    if rgb_frame is None or rgb_frame.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    ch_w = out_w // 3
    small = cv2.resize(rgb_frame, (ch_w, 100), interpolation=cv2.INTER_AREA)

    scope = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    colors = [(255, 60, 60), (60, 255, 60), (80, 140, 255)]  # R, G, B

    for ch in range(3):
        offset_x = ch * ch_w
        data = small[:, :, ch].astype(np.float32)
        y_pos = np.clip((1.0 - (data / 255.0)) * (out_h - 1), 0, out_h - 1).astype(np.int32)
        c_rgb = colors[ch]

        # Channel divider and graticule
        scope[:, offset_x, :] = (60, 60, 60)
        for ire in (0.0, 0.5, 1.0):
            y_line = int((1.0 - ire) * (out_h - 1))
            scope[y_line, offset_x:offset_x + ch_w, :] = (35, 40, 45)

        for col in range(ch_w):
            for y in y_pos[:, col]:
                curr = scope[y, offset_x + col]
                scope[y, offset_x + col, 0] = min(255, int(curr[0]) + int(c_rgb[0] * 0.35))
                scope[y, offset_x + col, 1] = min(255, int(curr[1]) + int(c_rgb[1] * 0.35))
                scope[y, offset_x + col, 2] = min(255, int(curr[2]) + int(c_rgb[2] * 0.35))

    return scope


def compute_vectorscope(rgb_frame: np.ndarray, out_size: int = 220) -> np.ndarray:
    """
    Computes a Vectorscope (Cb/Cr color-wheel polar diagram with skin-tone line).
    """
    if rgb_frame is None or rgb_frame.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    scope = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    center = out_size // 2
    r_max = (out_size // 2) - 10

    # Draw outer circular target & reference boxes
    cv2.circle(scope, (center, center), r_max, (50, 55, 65), 1)
    cv2.circle(scope, (center, center), r_max // 2, (35, 40, 45), 1)
    cv2.line(scope, (center, 5), (center, out_size - 5), (35, 40, 45), 1)
    cv2.line(scope, (5, center), (out_size - 5, center), (35, 40, 45), 1)

    # Skin-Tone Line (I-bar at ~123 degrees in Cb/Cr space)
    skin_rad = math.radians(123)
    x_skin = int(center - r_max * math.cos(skin_rad))
    y_skin = int(center - r_max * math.sin(skin_rad))
    cv2.line(scope, (center, center), (x_skin, y_skin), (180, 140, 80), 1)

    # Convert RGB to YCbCr
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    # Downsample points for real-time responsiveness
    cb = ycrcb[::4, ::4, 2].astype(np.float32) - 128.0
    cr = ycrcb[::4, ::4, 1].astype(np.float32) - 128.0

    scale = r_max / 128.0
    xs = np.clip((center + cb * scale).astype(np.int32), 0, out_size - 1)
    ys = np.clip((center - cr * scale).astype(np.int32), 0, out_size - 1)

    # Accumulate chrominance trace
    for x, y in zip(xs.ravel(), ys.ravel()):
        curr = scope[y, x]
        scope[y, x, 0] = min(255, curr[0] + 40)
        scope[y, x, 1] = min(255, curr[1] + 80)
        scope[y, x, 2] = min(255, curr[2] + 40)

    return scope


def compute_histogram_scope(rgb_frame: np.ndarray, out_w: int = 300, out_h: int = 200) -> np.ndarray:
    """
    Computes an RGB channel histogram overlay.
    """
    if rgb_frame is None or rgb_frame.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    scope = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    colors = [(255, 70, 70), (70, 255, 70), (80, 150, 255)]

    for ch in range(3):
        hist = cv2.calcHist([rgb_frame], [ch], None, [out_w], [0, 256])
        max_val = max(1.0, float(hist.max()))
        hist_norm = (hist / max_val) * (out_h - 10)
        c_rgb = colors[ch]

        pts = []
        for x in range(out_w):
            y = out_h - 1 - int(hist_norm[x][0])
            pts.append((x, y))

        for i in range(len(pts) - 1):
            cv2.line(scope, pts[i], pts[i + 1], c_rgb, 1, cv2.LINE_AA)

    return scope


class VideoScopesWindow(ctk.CTkToplevel):
    """
    Floating real-time Video Scopes analysis window.
    Features: RGB Parade, Luma Waveform, Vectorscope, and Histogram.
    """

    def __init__(self, app):
        super().__init__(app)
        self.title("Studio Video Scopes")
        self.geometry("740x500")
        self.resizable(True, True)
        self.configure(fg_color="#14161d")

        self.app = app
        self._scope_type = ctk.StringVar(value="RGB Parade")
        self._build_ui()
        self.update_scope()

    def _build_ui(self):
        tb = ctk.CTkFrame(self, fg_color="#1a1c24", height=48)
        tb.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(tb, text="Scope Mode:", font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold")).pack(side="left", padx=(12, 8))

        modes = ["RGB Parade", "Luma Waveform", "Vectorscope", "Histogram", "All Quad-View"]
        seg = ctk.CTkSegmentedButton(tb, values=modes, variable=self._scope_type, font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), height=32, command=lambda v: self.update_scope())
        seg.pack(side="left", padx=6, pady=6)

        self._canvas_lbl = ctk.CTkLabel(self, text="", fg_color="#0d0e12", corner_radius=6)
        self._canvas_lbl.pack(fill="both", expand=True, padx=8, pady=8)

    def update_scope(self, frame_rgb: np.ndarray = None):
        """Update scopes from current master preview frame."""
        if frame_rgb is None:
            if hasattr(self.app, "master_proxy_clip") and self.app.master_proxy_clip:
                try:
                    t = getattr(self.app, "playhead_t", 0.0)
                    frame_rgb = self.app.master_proxy_clip.get_frame(t)
                except Exception:
                    frame_rgb = None

        if frame_rgb is None:
            self._canvas_lbl.configure(image=None, text="No Frame Data")
            return

        mode = self._scope_type.get()
        w = max(300, self._canvas_lbl.winfo_width() - 20)
        h = max(200, self._canvas_lbl.winfo_height() - 20)

        if mode == "RGB Parade":
            scope_img = compute_rgb_parade(frame_rgb, out_w=w, out_h=h)
        elif mode == "Luma Waveform":
            scope_img = compute_luma_waveform(frame_rgb, out_w=w, out_h=h)
        elif mode == "Vectorscope":
            scope_img = compute_vectorscope(frame_rgb, out_size=min(w, h))
        elif mode == "Histogram":
            scope_img = compute_histogram_scope(frame_rgb, out_w=w, out_h=h)
        else:  # All Quad-View
            half_w, half_h = max(150, w // 2), max(100, h // 2)
            p1 = compute_rgb_parade(frame_rgb, out_w=half_w, out_h=half_h)
            p2 = compute_luma_waveform(frame_rgb, out_w=half_w, out_h=half_h)
            p3 = compute_vectorscope(frame_rgb, out_size=min(half_w, half_h))
            if p3.shape[:2] != (half_h, half_w):
                p3 = cv2.resize(p3, (half_w, half_h))
            p4 = compute_histogram_scope(frame_rgb, out_w=half_w, out_h=half_h)
            top_row = np.hstack((p1, p2))
            bot_row = np.hstack((p3, p4))
            scope_img = np.vstack((top_row, bot_row))

        pim = Image.fromarray(scope_img)
        cim = ctk.CTkImage(light_image=pim, dark_image=pim, size=pim.size)
        self._canvas_lbl.configure(image=cim, text="")
        self._canvas_lbl._image = cim
