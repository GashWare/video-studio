# color_grading.py
"""
Per-clip colour grading panel for the Video Studio Inspector.
Embedded as a CTkFrame sub-panel.
"""

import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# LUT preset definitions  (256-entry per-channel curves, shape (256, 3) uint8)
# ---------------------------------------------------------------------------

def _identity_lut() -> np.ndarray:
    lut = np.arange(256, dtype=np.uint8)
    return np.stack([lut, lut, lut], axis=1)  # (256, 3)  B G R


def _build_lut(r_fn, g_fn, b_fn) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    r = np.clip(r_fn(x), 0, 255).astype(np.uint8)
    g = np.clip(g_fn(x), 0, 255).astype(np.uint8)
    b = np.clip(b_fn(x), 0, 255).astype(np.uint8)
    return np.stack([b, g, r], axis=1)  # OpenCV BGR order


def _cinematic_lut():
    # Lift blacks, roll-off highlights, slight teal shadows
    return _build_lut(
        r_fn=lambda x: np.interp(x, [0, 64, 192, 255], [10, 70, 200, 235]),
        g_fn=lambda x: np.interp(x, [0, 64, 192, 255], [8, 68, 195, 230]),
        b_fn=lambda x: np.interp(x, [0, 64, 192, 255], [20, 80, 190, 225]),
    )


def _vintage_lut():
    return _build_lut(
        r_fn=lambda x: np.interp(x, [0, 128, 255], [30, 148, 240]),
        g_fn=lambda x: np.interp(x, [0, 128, 255], [20, 120, 210]),
        b_fn=lambda x: np.interp(x, [0, 128, 255], [10, 100, 180]),
    )


def _cool_tone_lut():
    return _build_lut(
        r_fn=lambda x: np.interp(x, [0, 255], [0, 220]),
        g_fn=lambda x: np.interp(x, [0, 255], [5, 240]),
        b_fn=lambda x: np.interp(x, [0, 255], [20, 255]),
    )


def _warm_tone_lut():
    return _build_lut(
        r_fn=lambda x: np.interp(x, [0, 255], [20, 255]),
        g_fn=lambda x: np.interp(x, [0, 255], [10, 240]),
        b_fn=lambda x: np.interp(x, [0, 255], [0, 210]),
    )


def _high_contrast_lut():
    # S-curve
    return _build_lut(
        r_fn=lambda x: np.interp(x, [0, 64, 128, 192, 255], [0, 40, 128, 215, 255]),
        g_fn=lambda x: np.interp(x, [0, 64, 128, 192, 255], [0, 40, 128, 215, 255]),
        b_fn=lambda x: np.interp(x, [0, 64, 128, 192, 255], [0, 40, 128, 215, 255]),
    )


def _faded_lut():
    return _build_lut(
        r_fn=lambda x: np.interp(x, [0, 255], [30, 225]),
        g_fn=lambda x: np.interp(x, [0, 255], [30, 225]),
        b_fn=lambda x: np.interp(x, [0, 255], [30, 225]),
    )


LUT_PRESETS: dict[str, np.ndarray] = {
    "None":          _identity_lut(),
    "Cinematic":     _cinematic_lut(),
    "Vintage":       _vintage_lut(),
    "Cool Tone":     _cool_tone_lut(),
    "Warm Tone":     _warm_tone_lut(),
    "High Contrast": _high_contrast_lut(),
    "Faded":         _faded_lut(),
}

DEFAULT_COLOR: dict = {
    "brightness": 0, "contrast": 0, "saturation": 0, "hue": 0,
    "lift": [1.0, 1.0, 1.0],
    "gamma": [1.0, 1.0, 1.0],
    "gain": [1.0, 1.0, 1.0],
    "lut_preset": "None",
}

LUT_PRESET_NAMES = ["None", "Cinematic", "Vintage", "Cool Tone", "Warm Tone", "High Contrast", "Faded"]


def parse_cube_lut(cube_content_or_path: str) -> np.ndarray:
    """
    Parses a .cube 3D LUT file and returns an (N, N, N, 3) float32 array in [0, 1].
    """
    import os
    if os.path.isfile(cube_content_or_path):
        with open(cube_content_or_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    else:
        lines = cube_content_or_path.strip().splitlines()

    size = 0
    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('TITLE'):
            continue
        if line.startswith('LUT_3D_SIZE'):
            size = int(line.split()[1])
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                data.append([r, g, b])
            except ValueError:
                continue

    if size == 0:
        size = int(round(len(data) ** (1/3))) if data else 17

    lut = np.array(data, dtype=np.float32).reshape((size, size, size, 3))
    return lut


def apply_3d_lut(rgb_frame: np.ndarray, lut_3d: np.ndarray) -> np.ndarray:
    """
    Applies a 3D LUT (size N, N, N, 3) to an RGB uint8 image using trilinear interpolation.
    """
    if rgb_frame is None or lut_3d is None:
        return rgb_frame

    size = lut_3d.shape[0]
    norm_rgb = (rgb_frame.astype(np.float32) / 255.0) * (size - 1)
    r_coords = norm_rgb[:, :, 0]
    g_coords = norm_rgb[:, :, 1]
    b_coords = norm_rgb[:, :, 2]

    try:
        from scipy.ndimage import map_coordinates
        out_r = map_coordinates(lut_3d[:, :, :, 0], [b_coords, g_coords, r_coords], order=1, mode='nearest')
        out_g = map_coordinates(lut_3d[:, :, :, 1], [b_coords, g_coords, r_coords], order=1, mode='nearest')
        out_b = map_coordinates(lut_3d[:, :, :, 2], [b_coords, g_coords, r_coords], order=1, mode='nearest')
        res = np.stack([out_r, out_g, out_b], axis=-1)
        return np.clip(res * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        r0 = np.floor(r_coords).astype(int).clip(0, size - 2)
        r1 = r0 + 1
        g0 = np.floor(g_coords).astype(int).clip(0, size - 2)
        g1 = g0 + 1
        b0 = np.floor(b_coords).astype(int).clip(0, size - 2)
        b1 = b0 + 1

        rd = (r_coords - r0)[:, :, np.newaxis]
        gd = (g_coords - g0)[:, :, np.newaxis]
        bd = (b_coords - b0)[:, :, np.newaxis]

        c000 = lut_3d[b0, g0, r0]
        c001 = lut_3d[b0, g0, r1]
        c010 = lut_3d[b0, g1, r0]
        c011 = lut_3d[b0, g1, r1]
        c100 = lut_3d[b1, g0, r0]
        c101 = lut_3d[b1, g0, r1]
        c110 = lut_3d[b1, g1, r0]
        c111 = lut_3d[b1, g1, r1]

        c00 = c000 * (1 - rd) + c001 * rd
        c01 = c010 * (1 - rd) + c011 * rd
        c10 = c100 * (1 - rd) + c101 * rd
        c11 = c110 * (1 - rd) + c111 * rd

        c0 = c00 * (1 - gd) + c01 * gd
        c1 = c10 * (1 - gd) + c11 * gd

        out = c0 * (1 - bd) + c1 * bd
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helper: labelled slider row
# ---------------------------------------------------------------------------

class _LabeledSlider(ctk.CTkFrame):
    """A single row: label | slider | value-label."""

    def __init__(self, parent, label: str, from_: float, to: float,
                 default: float, number_of_steps: int = 200, fmt: str = "{:.0f}"):
        super().__init__(parent, fg_color="transparent")
        self.fmt = fmt
        self._var = tk.DoubleVar(value=default)

        ctk.CTkLabel(self, text=label, width=95, anchor="w", font=("Segoe UI", 10, "bold")).pack(side="left")
        self._slider = ctk.CTkSlider(
            self, from_=from_, to=to, number_of_steps=number_of_steps,
            variable=self._var, width=130, height=16,
            command=self._on_change,
        )
        self._slider.pack(side="left", padx=4)
        self._val_lbl = ctk.CTkLabel(self, text=self._fmt_val(default), width=48, font=("Consolas", 10, "bold"))
        self._val_lbl.pack(side="left")

    def _fmt_val(self, v: float) -> str:
        return self.fmt.format(v)

    def _on_change(self, _val=None):
        self._val_lbl.configure(text=self._fmt_val(self._var.get()))
        if self._callback:
            self._callback(self._var.get())

    def set_callback(self, fn):
        self._callback = fn

    def get(self) -> float:
        return self._var.get()

    def set(self, value: float):
        self._var.set(value)
        self._val_lbl.configure(text=self._fmt_val(value))

    _callback = None


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class ColorGradingPanel(ctk.CTkFrame):
    """
    Colour grading sub-panel embedded in the Inspector.
    Properties stored/read via a clip dict.
    """

    def __init__(self, parent, change_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._change_cb = change_callback
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 6, "pady": 2}

        # ── Basic controls ──────────────────────────────────────────────
        basic_lbl = ctk.CTkLabel(self, text="Basic", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"))
        basic_lbl.pack(anchor="w", **pad)

        self._brightness = self._make_slider("Brightness", -100, 100, 0)
        self._contrast   = self._make_slider("Contrast",   -100, 100, 0)
        self._saturation = self._make_slider("Saturation", -100, 100, 0)
        self._hue        = self._make_slider("Hue Shift",  -180, 180, 0)

        # ── Lift (shadows) ──────────────────────────────────────────────
        lift_lbl = ctk.CTkLabel(self, text="Lift (Shadows)", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"))
        lift_lbl.pack(anchor="w", **pad)
        self._lift_r = self._make_slider("Lift R", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")
        self._lift_g = self._make_slider("Lift G", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")
        self._lift_b = self._make_slider("Lift B", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")

        # ── Gamma (midtones) ────────────────────────────────────────────
        gamma_lbl = ctk.CTkLabel(self, text="Gamma (Midtones)", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"))
        gamma_lbl.pack(anchor="w", **pad)
        self._gamma_r = self._make_slider("Gamma R", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")
        self._gamma_g = self._make_slider("Gamma G", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")
        self._gamma_b = self._make_slider("Gamma B", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")

        # ── Gain (highlights) ───────────────────────────────────────────
        gain_lbl = ctk.CTkLabel(self, text="Gain (Highlights)", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"))
        gain_lbl.pack(anchor="w", **pad)
        self._gain_r = self._make_slider("Gain R", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")
        self._gain_g = self._make_slider("Gain G", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")
        self._gain_b = self._make_slider("Gain B", 0.0, 2.0, 1.0, steps=200, fmt="{:.2f}")

        # ── Histogram ───────────────────────────────────────────────────
        hist_lbl = ctk.CTkLabel(self, text="Histogram", font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"))
        hist_lbl.pack(anchor="w", **pad)
        self._hist_canvas = tk.Canvas(self, width=180, height=60,
                                       bg="#1a1a1a", highlightthickness=1,
                                       highlightbackground="#444444")
        self._hist_canvas.pack(padx=6, pady=4)

        # ── LUT Presets ─────────────────────────────────────────────────
        lut_frame = ctk.CTkFrame(self, fg_color="transparent")
        lut_frame.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(lut_frame, text="LUT Preset", width=95, anchor="w", font=("Segoe UI", 10, "bold")).pack(side="left")
        self._lut_combo = ctk.CTkComboBox(
            lut_frame, values=LUT_PRESET_NAMES, width=160, height=30,
            command=self._on_change,
        )
        self._lut_combo.set("None")
        self._lut_combo.pack(side="left", padx=4)

        # ── Reset button ─────────────────────────────────────────────────
        ctk.CTkButton(self, text="Reset All", command=self._reset_all,
                      height=32, font=("Segoe UI", 11, "bold")).pack(padx=6, pady=6)

    def _make_slider(self, label: str, from_: float, to: float, default: float,
                     steps: int = 200, fmt: str = "{:.0f}") -> _LabeledSlider:
        s = _LabeledSlider(self, label, from_, to, default,
                            number_of_steps=steps, fmt=fmt)
        s.set_callback(self._on_change)
        s.pack(fill="x", padx=6, pady=1)
        return s

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_change(self, _val=None):
        if self._change_cb:
            self._change_cb()

    def _reset_all(self):
        self._brightness.set(0)
        self._contrast.set(0)
        self._saturation.set(0)
        self._hue.set(0)
        for sl in (self._lift_r, self._lift_g, self._lift_b,
                   self._gamma_r, self._gamma_g, self._gamma_b,
                   self._gain_r, self._gain_g, self._gain_b):
            sl.set(1.0)
        self._lut_combo.set("None")
        if self._change_cb:
            self._change_cb()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_from_clip(self, clip: dict):
        """Populate all sliders from clip's ``color`` sub-dict."""
        c = clip.get("color", {})
        self._brightness.set(c.get("brightness", 0))
        self._contrast.set(c.get("contrast", 0))
        self._saturation.set(c.get("saturation", 0))
        self._hue.set(c.get("hue", 0))
        lift  = c.get("lift",  [1.0, 1.0, 1.0])
        gamma = c.get("gamma", [1.0, 1.0, 1.0])
        gain  = c.get("gain",  [1.0, 1.0, 1.0])
        self._lift_r.set(lift[0]);  self._lift_g.set(lift[1]);  self._lift_b.set(lift[2])
        self._gamma_r.set(gamma[0]); self._gamma_g.set(gamma[1]); self._gamma_b.set(gamma[2])
        self._gain_r.set(gain[0]);  self._gain_g.set(gain[1]);  self._gain_b.set(gain[2])
        self._lut_combo.set(c.get("lut_preset", "None"))

    def save_to_clip(self, clip: dict):
        """Write current slider values back to clip's ``color`` sub-dict."""
        clip["color"] = {
            "brightness": self._brightness.get(),
            "contrast":   self._contrast.get(),
            "saturation": self._saturation.get(),
            "hue":        self._hue.get(),
            "lift":  [self._lift_r.get(),  self._lift_g.get(),  self._lift_b.get()],
            "gamma": [self._gamma_r.get(), self._gamma_g.get(), self._gamma_b.get()],
            "gain":  [self._gain_r.get(),  self._gain_g.get(),  self._gain_b.get()],
            "lut_preset": self._lut_combo.get(),
        }

    def update_histogram(self, frame: np.ndarray):
        """Render an RGB histogram from *frame* (H×W×3 uint8) onto the canvas."""
        if frame is None or frame.size == 0:
            return
        canvas = self._hist_canvas
        W, H = 180, 60
        canvas.delete("all")
        colors = ("#ef4444", "#22c55e", "#3b82f6")  # R G B
        for ch_idx, color in enumerate(colors):
            hist = cv2.calcHist([frame], [ch_idx], None, [64], [0, 256])
            hist = hist.flatten()
            max_val = hist.max() if hist.max() > 0 else 1
            for i, v in enumerate(hist):
                x0 = int(i * W / 64)
                x1 = int((i + 1) * W / 64)
                bar_h = int((v / max_val) * H)
                if bar_h > 0:
                    canvas.create_rectangle(x0, H - bar_h, x1, H,
                                            fill=color, outline="", stipple="gray50")

    def apply_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply all colour grading to *frame* (H×W×3 RGB uint8). Returns RGB uint8."""
        if frame is None or frame.size == 0:
            return frame
        # Work in BGR (OpenCV convention)
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).astype(np.float32)

        c = {}
        self.save_to_clip(c)
        color = c["color"]

        # ── Brightness / Contrast ───────────────────────────────────────
        brightness = float(color["brightness"])
        contrast   = float(color["contrast"])
        alpha = 1.0 + contrast / 100.0          # scale factor
        beta  = brightness                       # additive offset
        img = img * alpha + beta
        img = np.clip(img, 0, 255)

        # ── LUT preset ──────────────────────────────────────────────────
        preset_name = color.get("lut_preset", "None")
        lut = LUT_PRESETS.get(preset_name, LUT_PRESETS["None"])  # (256, 3) BGR
        img_u8 = img.astype(np.uint8)
        img_u8 = lut[img_u8[:, :, 0], 0:1].squeeze(2)[:, :, np.newaxis]  # not quite right
        # Proper LUT application per channel
        img_u8_bgr = np.clip(img, 0, 255).astype(np.uint8)
        b_ch = lut[img_u8_bgr[:, :, 0], 0]
        g_ch = lut[img_u8_bgr[:, :, 1], 1]
        r_ch = lut[img_u8_bgr[:, :, 2], 2]
        img  = np.stack([b_ch, g_ch, r_ch], axis=2).astype(np.float32)

        # ── Saturation / Hue ────────────────────────────────────────────
        saturation = float(color["saturation"])
        hue_shift  = float(color["hue"])
        if saturation != 0 or hue_shift != 0:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
            if hue_shift != 0:
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 2.0) % 180.0
            if saturation != 0:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] + saturation * 2.55, 0, 255)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8),
                               cv2.COLOR_HSV2BGR).astype(np.float32)

        # ── Lift / Gamma / Gain per channel ─────────────────────────────
        lift  = color["lift"]   # [R, G, B]
        gamma = color["gamma"]
        gain  = color["gain"]

        # Convert to 0-1 float, apply per-channel:  out = gain * (lift * (in ^ gamma))
        # OpenCV BGR order → index mapping: 0=B, 1=G, 2=R
        channel_map = [(2, 0), (1, 1), (0, 2)]  # (bgr_idx, rgb_idx)
        img_norm = img / 255.0
        for bgr_i, rgb_i in channel_map:
            g = max(gamma[rgb_i], 1e-6)
            img_norm[:, :, bgr_i] = (
                gain[rgb_i] * np.power(
                    np.clip(img_norm[:, :, bgr_i] * lift[rgb_i], 0, None),
                    1.0 / g
                )
            )
        img = np.clip(img_norm * 255.0, 0, 255).astype(np.uint8)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
