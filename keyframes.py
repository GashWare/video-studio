# keyframes.py
"""
Keyframe animation system for animating clip properties over time.
"""

import tkinter as tk
import customtkinter as ctk
import numpy as np

ANIMATABLE_PROPS = ["scale", "x", "y", "volume", "opacity", "rot_deg"]

EASING_FUNCTIONS = ["Linear", "Ease In", "Ease Out", "Ease In/Out", "Exponential", "Bezier (S-Curve)"]

# Default values for each animatable property
PROP_DEFAULTS: dict = {
    "scale":   1.0,
    "x":       0.0,
    "y":       0.0,
    "volume":  1.0,
    "opacity": 1.0,
    "rot_deg": 0.0,
}

# Display ranges for each property (used by the editor UI)
PROP_RANGES: dict = {
    "scale":   (0.0,  5.0),
    "x":       (-2000.0, 2000.0),
    "y":       (-2000.0, 2000.0),
    "volume":  (0.0,  2.0),
    "opacity": (0.0,  1.0),
    "rot_deg": (-360.0, 360.0),
}


# ---------------------------------------------------------------------------
# Easing helpers
# ---------------------------------------------------------------------------

def _ease_in(t: float) -> float:
    """Cubic ease-in: slow start, fast end."""
    return t * t * t


def _ease_out(t: float) -> float:
    """Cubic ease-out: fast start, slow end."""
    t = 1.0 - t
    return 1.0 - t * t * t


def _ease_in_out(t: float) -> float:
    """Smoothstep: slow start and end."""
    return t * t * (3.0 - 2.0 * t)


def _exponential(t: float) -> float:
    """Exponential power easing."""
    return t * t * t * t


def _bezier_s(t: float) -> float:
    """Steeper 5th-order smootherstep Bezier curve."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _apply_easing(t_norm: float, easing: str) -> float:
    """Apply *easing* to a normalised time value in [0, 1]."""
    t_norm = float(np.clip(t_norm, 0.0, 1.0))
    if easing == "Ease In":
        return _ease_in(t_norm)
    if easing == "Ease Out":
        return _ease_out(t_norm)
    if easing == "Ease In/Out":
        return _ease_in_out(t_norm)
    if easing == "Exponential":
        return _exponential(t_norm)
    if easing in ("Bezier (S-Curve)", "Bezier"):
        return _bezier_s(t_norm)
    return t_norm   # Linear


# ---------------------------------------------------------------------------
# KeyframeTrack
# ---------------------------------------------------------------------------

class KeyframeTrack:
    """
    Manages keyframes for a single animatable property.

    Each keyframe: ``{'t': float, 'value': float, 'easing': str}``
    The *easing* field controls how the segment **from this keyframe to the next**
    is interpolated.
    """

    def __init__(self, prop_name: str, default_value: float = None):
        self.prop_name     = prop_name
        self.default_value = (
            default_value if default_value is not None
            else PROP_DEFAULTS.get(prop_name, 0.0)
        )
        self._keyframes: list = []   # sorted by 't'

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_keyframe(self, t: float, value: float, easing: str = "Linear"):
        """Insert or replace a keyframe at time *t*."""
        t = float(t)
        # Remove any existing keyframe at the same time (within tolerance)
        self._keyframes = [kf for kf in self._keyframes if abs(kf["t"] - t) > 1e-6]
        self._keyframes.append({"t": t, "value": float(value), "easing": easing})
        self._keyframes.sort(key=lambda kf: kf["t"])

    def remove_keyframe(self, t: float):
        """Remove the nearest keyframe within 0.05 s of *t*."""
        t = float(t)
        candidates = [(abs(kf["t"] - t), i) for i, kf in enumerate(self._keyframes)]
        if not candidates:
            return
        dist, idx = min(candidates, key=lambda x: x[0])
        if dist <= 0.05:
            del self._keyframes[idx]

    def clear(self):
        """Remove all keyframes."""
        self._keyframes.clear()

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def get_value(self, t: float) -> float:
        """
        Interpolate the property value at time *t*.

        - No keyframes → returns ``default_value``.
        - Before first keyframe / after last → clamps to boundary value.
        - Between two keyframes → applies the segment easing of the earlier one.
        """
        if not self._keyframes:
            return self.default_value

        kfs = self._keyframes
        t = float(t)

        if t <= kfs[0]["t"]:
            return kfs[0]["value"]
        if t >= kfs[-1]["t"]:
            return kfs[-1]["value"]

        # Find surrounding keyframes
        for i in range(len(kfs) - 1):
            k0, k1 = kfs[i], kfs[i + 1]
            if k0["t"] <= t <= k1["t"]:
                seg_dur = k1["t"] - k0["t"]
                if seg_dur < 1e-9:
                    return k1["value"]
                t_norm = (t - k0["t"]) / seg_dur
                t_eased = _apply_easing(t_norm, k0.get("easing", "Linear"))
                return k0["value"] + (k1["value"] - k0["value"]) * t_eased

        return kfs[-1]["value"]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "prop_name":     self.prop_name,
            "default_value": self.default_value,
            "keyframes":     [dict(kf) for kf in self._keyframes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KeyframeTrack":
        track = cls(d["prop_name"], d.get("default_value"))
        for kf in d.get("keyframes", []):
            track._keyframes.append(dict(kf))
        track._keyframes.sort(key=lambda k: k["t"])
        return track

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def keyframes(self) -> list:
        return list(self._keyframes)

    def __repr__(self) -> str:
        return f"<KeyframeTrack prop={self.prop_name!r} kfs={len(self._keyframes)}>"


# ---------------------------------------------------------------------------
# KeyframeEditor (UI)
# ---------------------------------------------------------------------------

_KF_DOT_R   = 5     # dot radius px
_ROW_H      = 32    # px per property row
_LABEL_W    = 80    # px for label column
_TIMELINE_W = 500   # px for timeline area
_CANVAS_W   = _LABEL_W + _TIMELINE_W
_CANVAS_H   = len(ANIMATABLE_PROPS) * _ROW_H + 20


class KeyframeEditor(ctk.CTkToplevel):
    """
    Compact keyframe curve editor for a single clip.

    Layout
    ------
    - Each row = one animatable property.
    - Coloured dots mark keyframe positions along the time axis.
    - Left-click on the timeline → add a keyframe at the playhead time with
      the value from the value entry.
    - Right-click on a dot → delete that keyframe.
    - A value entry allows setting an exact value before adding.
    """

    def __init__(self, app, clip: dict):
        super().__init__(app)
        self.title(f"Keyframe Editor — {clip.get('path', 'Clip')}")
        self.resizable(True, True)
        self.grab_set()

        self._app      = app
        self._clip     = clip
        self._duration = clip.get("trim_out", 10.0) - clip.get("trim_in", 0.0)

        # Build or restore tracks from clip
        self._tracks: dict[str, KeyframeTrack] = {}
        kf_data = clip.get("keyframes", {})
        for prop in ANIMATABLE_PROPS:
            if prop in kf_data:
                self._tracks[prop] = KeyframeTrack.from_dict(kf_data[prop])
            else:
                self._tracks[prop] = KeyframeTrack(prop)

        # Playhead position (seconds)
        self._playhead = 0.0

        self._build_ui()
        self._redraw()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", **pad)

        # Playhead scrubber
        ctk.CTkLabel(top, text="Playhead (s):").pack(side="left")
        self._ph_var = tk.DoubleVar(value=0.0)
        self._ph_slider = ctk.CTkSlider(
            top, from_=0.0, to=max(self._duration, 1.0),
            number_of_steps=max(int(self._duration * 30), 30),
            variable=self._ph_var, width=300,
            command=self._on_playhead_move,
        )
        self._ph_slider.pack(side="left", padx=8)
        self._ph_lbl = ctk.CTkLabel(top, text="0.00 s", width=60)
        self._ph_lbl.pack(side="left")

        # Value entry
        ctk.CTkLabel(top, text="Value:").pack(side="left", padx=(12, 2))
        self._val_entry = ctk.CTkEntry(top, width=80)
        self._val_entry.insert(0, "1.0")
        self._val_entry.pack(side="left")

        # Easing selector
        ctk.CTkLabel(top, text="Easing:").pack(side="left", padx=(8, 2))
        self._easing_combo = ctk.CTkComboBox(top, values=EASING_FUNCTIONS, width=120)
        self._easing_combo.set("Linear")
        self._easing_combo.pack(side="left")

        # Canvas
        canvas_frame = ctk.CTkFrame(self)
        canvas_frame.pack(fill="both", expand=True, **pad)
        self._canvas = tk.Canvas(
            canvas_frame, width=_CANVAS_W, height=_CANVAS_H,
            bg="#1e1e1e", highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>",   self._on_canvas_click)
        self._canvas.bind("<Button-3>",   self._on_canvas_right_click)

        # Bottom buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", **pad)
        ctk.CTkButton(btn_row, text="Save to Clip", command=self._save,
                      width=120).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Clear All", command=self._clear_all,
                      fg_color="gray40", width=100).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Close", command=self.destroy,
                      fg_color="gray40", width=80).pack(side="right", padx=4)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    _ROW_COLORS = {
        "scale":   "#f59e0b",
        "x":       "#3b82f6",
        "y":       "#22c55e",
        "volume":  "#a855f7",
        "opacity": "#ec4899",
        "rot_deg": "#ef4444",
    }

    def _redraw(self):
        c = self._canvas
        c.delete("all")
        dur = max(self._duration, 1.0)

        for row_idx, prop in enumerate(ANIMATABLE_PROPS):
            y_center = row_idx * _ROW_H + _ROW_H // 2 + 10
            color = self._ROW_COLORS.get(prop, "#ffffff")

            # Label
            c.create_text(
                _LABEL_W // 2, y_center,
                text=prop, fill=color, font=("Segoe UI", 9), anchor="center"
            )

            # Row background
            c.create_line(_LABEL_W, y_center, _LABEL_W + _TIMELINE_W, y_center,
                          fill="#333333", width=1)

            # Keyframe dots
            track = self._tracks[prop]
            for kf in track.keyframes:
                x = _LABEL_W + int(kf["t"] / dur * _TIMELINE_W)
                c.create_oval(x - _KF_DOT_R, y_center - _KF_DOT_R,
                               x + _KF_DOT_R, y_center + _KF_DOT_R,
                               fill=color, outline="#ffffff", width=1,
                               tags=(f"dot_{prop}_{kf['t']}",))
                # Value tooltip
                c.create_text(x, y_center - _KF_DOT_R - 4,
                               text=f"{kf['value']:.2f}", fill="#cccccc",
                               font=("Segoe UI", 7), anchor="s")

        # Playhead line
        ph_x = _LABEL_W + int(self._playhead / dur * _TIMELINE_W)
        c.create_line(ph_x, 5, ph_x, _CANVAS_H - 5,
                      fill="#ffffff", width=2, dash=(4, 3))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_playhead_move(self, _val=None):
        self._playhead = self._ph_var.get()
        self._ph_lbl.configure(text=f"{self._playhead:.2f} s")
        # Update value entry to show interpolated value of the selected row
        self._redraw()

    def _row_at_y(self, y: int) -> str | None:
        idx = max(0, (y - 10) // _ROW_H)
        if 0 <= idx < len(ANIMATABLE_PROPS):
            return ANIMATABLE_PROPS[idx]
        return None

    def _time_at_x(self, x: int) -> float:
        dur = max(self._duration, 1.0)
        tl_x = x - _LABEL_W
        return max(0.0, min(dur, tl_x / _TIMELINE_W * dur))

    def _on_canvas_click(self, event: tk.Event):
        if event.x < _LABEL_W:
            return
        prop = self._row_at_y(event.y)
        if prop is None:
            return
        t = self._time_at_x(event.x)
        try:
            value = float(self._val_entry.get())
        except ValueError:
            value = self._tracks[prop].default_value
        easing = self._easing_combo.get()
        self._tracks[prop].add_keyframe(t, value, easing)
        self._redraw()

    def _on_canvas_right_click(self, event: tk.Event):
        if event.x < _LABEL_W:
            return
        prop = self._row_at_y(event.y)
        if prop is None:
            return
        t = self._time_at_x(event.x)
        self._tracks[prop].remove_keyframe(t)
        self._redraw()

    # ------------------------------------------------------------------
    # Save / Clear
    # ------------------------------------------------------------------

    def _save(self):
        """Serialise all tracks back into the clip dict."""
        self._clip["keyframes"] = {
            prop: track.to_dict()
            for prop, track in self._tracks.items()
            if track.keyframes   # only write non-empty tracks
        }
        # Notify app
        for method_name in ("_refresh_timeline", "refresh_timeline", "_update_ui"):
            if hasattr(self._app, method_name):
                getattr(self._app, method_name)()
                break
        self.destroy()

    def _clear_all(self):
        for track in self._tracks.values():
            track.clear()
        self._redraw()
