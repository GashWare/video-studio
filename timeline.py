import tkinter as tk
import os
import math
import threading
import collections

try:
    import cv2
    from PIL import Image, ImageTk
    _HAS_CV2_PIL = True
except ImportError:
    _HAS_CV2_PIL = False

from waveform import waveform_cache

try:
    from commands import TrimClipCommand, MoveClipCommand
    _HAS_COMMANDS = True
except ImportError:
    _HAS_COMMANDS = False

# Zoom presets: pixels per second (supports zooming out to fit long sequences)
_ZOOM_LEVELS = [0.2, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0, 18.0, 25.0, 40.0, 60.0, 90.0, 140.0, 200.0, 300.0, 500.0]
_ZOOM_DEFAULT_IDX = 9          # 25.0 px/s
_TRACK_LABEL_W = 240           # width of the track-label gutter (fits large FL Studio knobs)

# Resize handle hit zone (pixels from clip edge)
_RESIZE_HIT = 14


# ---------------------------------------------------------------------------
# Periodic Timeline Video Thumbnail Cache
# ---------------------------------------------------------------------------
class TimelineThumbnailCache:
    """
    Extracts and caches periodic thumbnail frames for video clips in the timeline.
    Uses background worker thread to extract frames via OpenCV without blocking the UI.
    """
    def __init__(self, max_cache: int = 500):
        self._cache = collections.OrderedDict()      # (path, t_key, w, h) -> ImageTk.PhotoImage
        self._pil_cache = collections.OrderedDict()  # (path, t_key) -> PIL.Image
        self._pending = set()
        self._lock = threading.Lock()
        self._max_cache = max_cache
        self._queue = collections.deque()
        self._worker_running = False

    def get(self, path: str, t: float, width_px: int, height_px: int, callback=None):
        if not _HAS_CV2_PIL or not path or not os.path.isfile(path):
            return None
        if width_px <= 4 or height_px <= 4:
            return None

        t_key = round(max(0.0, float(t)), 1)
        w_key = max(10, int(width_px))
        h_key = max(10, int(height_px))
        key = (path, t_key, w_key, h_key)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            # Check if base PIL image is already cached
            base_key = (path, t_key)
            if base_key in self._pil_cache:
                pil_im = self._pil_cache[base_key]
                try:
                    resized = pil_im.resize((w_key, h_key), Image.Resampling.BILINEAR)
                    photo = ImageTk.PhotoImage(resized)
                    self._cache[key] = photo
                    if len(self._cache) > self._max_cache:
                        self._cache.popitem(last=False)
                    return photo
                except Exception:
                    pass

            # Queue background extraction
            if base_key not in self._pending:
                self._pending.add(base_key)
                self._queue.append((path, t_key, w_key, h_key, callback))
                if not self._worker_running:
                    self._worker_running = True
                    threading.Thread(target=self._worker, daemon=True).start()

        return None

    def _worker(self):
        while True:
            with self._lock:
                if not self._queue:
                    self._worker_running = False
                    break
                path, t_key, w_key, h_key, callback = self._queue.popleft()

            pil_im = None
            try:
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_idx = int(t_key * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_MSEC, t_key * 1000.0)
                        ret, frame = cap.read()
                    cap.release()

                    if ret and frame is not None and frame.size > 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_im = Image.fromarray(rgb)
            except Exception:
                pil_im = None

            with self._lock:
                self._pending.discard((path, t_key))
                if pil_im is not None:
                    self._pil_cache[(path, t_key)] = pil_im
                    if len(self._pil_cache) > self._max_cache:
                        self._pil_cache.popitem(last=False)

            if callback and pil_im is not None:
                try:
                    callback(path)
                except Exception:
                    pass

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._pil_cache.clear()
            self._pending.clear()
            self._queue.clear()


timeline_thumb_cache = TimelineThumbnailCache()


# Visual track themes for clear track-by-track distinction
_TRACK_THEMES = {
    'V2': {
        'name':     'V2 (Overlay)',
        'accent':   '#9b59b6',     # Purple/Violet
        'badge_bg': '#4a235a',
        'badge_fg': '#f5eef8',
        'lane_bg':  '#1a1524',
        'clip_base':'#6c3483',
        'clip_hi':  '#a569bd',
        'wf_color': '#bb8fce',
    },
    'V1': {
        'name':     'V1 (Main)',
        'accent':   '#3498db',     # Sky / Royal Blue
        'badge_bg': '#1b4f72',
        'badge_fg': '#ebf5fb',
        'lane_bg':  '#131c28',
        'clip_base':'#1f618d',
        'clip_hi':  '#5dade2',
        'wf_color': '#5dade2',
    },
    'A1': {
        'name':     'A1 (Audio 1)',
        'accent':   '#2ecc71',     # Emerald Green
        'badge_bg': '#145a32',
        'badge_fg': '#eafaf1',
        'lane_bg':  '#112218',
        'clip_base':'#1e8449',
        'clip_hi':  '#58d68d',
        'wf_color': '#2ecc71',
    },
    'A2': {
        'name':     'A2 (Audio 2)',
        'accent':   '#e67e22',     # Warm Amber / Orange
        'badge_bg': '#6e2c00',
        'badge_fg': '#fef5e7',
        'lane_bg':  '#241a12',
        'clip_base':'#b9770e',
        'clip_hi':  '#f39c12',
        'wf_color': '#f1c40f',
    },
}


def get_track_theme(tk_key: str) -> dict:
    """Return visual theme for any track key (built-in or dynamic)."""
    if tk_key in _TRACK_THEMES:
        return _TRACK_THEMES[tk_key]
    if tk_key.startswith('V'):
        idx = int(tk_key[1:]) if tk_key[1:].isdigit() else 1
        is_even = (idx % 2 == 0)
        return {
            'name':     f"{tk_key} (Video {idx})",
            'accent':   '#9b59b6' if is_even else '#3498db',
            'badge_bg': '#4a235a' if is_even else '#1b4f72',
            'badge_fg': '#f5eef8' if is_even else '#ebf5fb',
            'lane_bg':  '#1a1524' if is_even else '#131c28',
            'clip_base':'#6c3483' if is_even else '#1f618d',
            'clip_hi':  '#a569bd' if is_even else '#5dade2',
            'wf_color': '#bb8fce' if is_even else '#5dade2',
        }
    elif tk_key.startswith(('S', 'T')):
        idx = int(tk_key[1:]) if tk_key[1:].isdigit() else 1
        is_even = (idx % 2 == 0)
        return {
            'name':     f"{tk_key} (Subtitle {idx})" if tk_key.startswith('S') else f"{tk_key} (Title {idx})",
            'accent':   '#f39c12' if is_even else '#f1c40f',
            'badge_bg': '#7d6608' if is_even else '#5a4904',
            'badge_fg': '#fef9e7',
            'lane_bg':  '#201c10' if is_even else '#18150c',
            'clip_base':'#9a7d0a' if is_even else '#b7950b',
            'clip_hi':  '#f1c40f' if is_even else '#f7dc6f',
            'wf_color': '#f1c40f',
        }
    elif tk_key.startswith('I'):
        idx = int(tk_key[1:]) if tk_key[1:].isdigit() else 1
        is_even = (idx % 2 == 0)
        return {
            'name':     f"{tk_key} (Image {idx})",
            'accent':   '#e91e63' if is_even else '#d81b60',
            'badge_bg': '#880e4f' if is_even else '#6a0036',
            'badge_fg': '#fce4ec',
            'lane_bg':  '#23101b' if is_even else '#190a12',
            'clip_base':'#ad1457' if is_even else '#c2185b',
            'clip_hi':  '#ec407a' if is_even else '#f48fb1',
            'wf_color': '#f48fb1',
        }
    else:
        idx = int(tk_key[1:]) if tk_key[1:].isdigit() else 1
        is_even = (idx % 2 == 0)
        return {
            'name':     f"{tk_key} (Audio {idx})",
            'accent':   '#e67e22' if is_even else '#2ecc71',
            'badge_bg': '#6e2c00' if is_even else '#145a32',
            'badge_fg': '#fef5e7' if is_even else '#eafaf1',
            'lane_bg':  '#241a12' if is_even else '#112218',
            'clip_base':'#b9770e' if is_even else '#1e8449',
            'clip_hi':  '#f39c12' if is_even else '#58d68d',
            'wf_color': '#f1c40f' if is_even else '#2ecc71',
        }


# Clip colour-label palette
_LABEL_COLORS = {
    "red":    "#c0392b",
    "orange": "#e67e22",
    "yellow": "#f1c40f",
    "green":  "#27ae60",
    "blue":   "#2980b9",
    "purple": "#8e44ad",
    "pink":   "#e91e8c",
}

_SNAP_THRESH_PX = 8   # pixels within which to snap to another clip's edge


def _draw_fl_knob(
    canvas: tk.Canvas,
    cx: float,
    cy: float,
    radius: float,
    value: float,
    min_val: float,
    max_val: float,
    default_val: float,
    is_bipolar: bool,
    label: str,
    accent_color: str = "#00d2be",
):
    """
    Renders an authentic FL Studio vector rotary knob on a Tkinter canvas.
    - Outer track groove background arc
    - Colored active indicator arc
    - Dark metallic dial body with circular bevel
    - Radiating pointer notch
    - Label and formatted value readout
    """
    # 1. Background arc ring (from 225 deg down clockwise 270 deg to -45 deg)
    r_track = radius + 3.0
    canvas.create_arc(
        cx - r_track, cy - r_track, cx + r_track, cy + r_track,
        start=225, extent=-270, style='arc',
        outline="#282c37", width=3.5, tags="static"
    )

    # 2. Active arc
    val_clamped = max(min_val, min(max_val, float(value)))
    if is_bipolar:
        # Bipolar: Center is 90 deg (12 o'clock). Range is [-1.0, 1.0].
        p = val_clamped
        if abs(p) > 0.02:
            extent = -p * 135.0
            canvas.create_arc(
                cx - r_track, cy - r_track, cx + r_track, cy + r_track,
                start=90, extent=extent, style='arc',
                outline=accent_color, width=3.5, tags="static"
            )
        # Center detent mark at 12 o'clock
        canvas.create_line(cx, cy - r_track - 2, cx, cy - r_track + 2, fill="#555a68", width=1.5, tags="static")
        angle_deg = 90.0 - p * 135.0
    else:
        # Unipolar: 0.0 to 2.0 (default 1.0).
        frac = (val_clamped - min_val) / max(0.001, max_val - min_val)
        if frac > 0.01:
            extent = -frac * 270.0
            canvas.create_arc(
                cx - r_track, cy - r_track, cx + r_track, cy + r_track,
                start=225, extent=extent, style='arc',
                outline=accent_color, width=3.5, tags="static"
            )
        # Default marker tick (100% / 12 o'clock)
        def_frac = (default_val - min_val) / max(0.001, max_val - min_val)
        def_deg = 225.0 - def_frac * 270.0
        def_rad = math.radians(def_deg)
        canvas.create_line(
            cx + (r_track - 1) * math.cos(def_rad), cy - (r_track - 1) * math.sin(def_rad),
            cx + (r_track + 3) * math.cos(def_rad), cy - (r_track + 3) * math.sin(def_rad),
            fill="#555a68", width=1.5, tags="static"
        )
        angle_deg = 225.0 - frac * 270.0

    # 3. Dial Body (dark circular cap with bevel)
    r_body = radius - 0.5
    canvas.create_oval(
        cx - r_body, cy - r_body, cx + r_body, cy + r_body,
        fill="#1a1c24", outline="#3d4456", width=1.5, tags="static"
    )

    # 4. Pointer Notch (radiating outward)
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    canvas.create_line(
        cx + 2.0 * cos_a, cy - 2.0 * sin_a,
        cx + (r_body - 1.0) * cos_a, cy - (r_body - 1.0) * sin_a,
        fill="#ffffff", width=2.5, tags="static"
    )

    # 5. Label and value readout below/beside knob
    if is_bipolar:
        pan_str = "C" if abs(val_clamped) < 0.05 else (f"L{int(abs(val_clamped)*100)}" if val_clamped < 0 else f"R{int(val_clamped*100)}")
        readout_txt = pan_str
        readout_col = "#ffb347" if abs(val_clamped) >= 0.05 else "#9ca3af"
    elif label == "SPD":
        readout_txt = f"{val_clamped:.2f}x"
        readout_col = "#c084fc" if abs(val_clamped - 1.0) > 0.02 else "#9ca3af"
    else:
        readout_txt = f"{int(val_clamped * 100)}%"
        readout_col = "#00d2be" if abs(val_clamped - 1.0) > 0.02 else "#9ca3af"

    # Top readout text
    canvas.create_text(
        cx, cy - radius - 6, text=readout_txt,
        fill=readout_col, font=("Segoe UI", 8, "bold"), tags="static"
    )
    # Bottom Label (VOL, PAN, or SPD)
    canvas.create_text(
        cx, cy + radius + 7, text=label,
        fill="#8a92a6", font=("Segoe UI", 8, "bold"), tags="static"
    )


class MultiTrackTimeline(tk.Frame):
    """
    NLE Interactive Multi-Track Timeline.

    Controls
    --------
    Ctrl + Mouse-Wheel   : zoom in / out (centred on cursor)
    Mouse-Wheel          : horizontal scroll
    Shift + Mouse-Wheel  : horizontal scroll (alternative)
    Click ruler/empty    : move playhead + scrub
    Click clip           : select clip (Shift+click to multi-select)
    B1-Drag clip edge    : resize clip (trim in / trim out)
    B1-Drag clip body    : move clip (snaps to 0.1 s grid)
    B1-Release           : commit move/trim → rebuild proxy
    Right-click clip     : context menu (Split, Duplicate, Separate Audio, Link, Label, Delete)
    Right-click gutter   : track mixer settings (Volume, Pan, Speed, Mute, Solo, Delete Track)
    +  /  -  buttons     : zoom via toolbar
    + Video / + Audio    : add timeline tracks
    Horizontal scrollbar : pan timeline
    """

    def __init__(self, master, app, **kwargs):
        super().__init__(master, bg="#1a1a1a", **kwargs)

        self.app = app
        self.tracks     = ['V2 (Overlay)', 'V1 (Main)', 'A1 (Audio 1)', 'A2 (Audio 2)']
        self.track_keys = ['V2', 'V1', 'A1', 'A2']
        self.track_h    = 68
        self.ruler_h    = 32

        # Track mixer controls: volume (0.0 to 2.0, default 1.0), stereo pan (-1.0 to 1.0, default 0.0), speed (0.25 to 4.0, default 1.0)
        self.track_volumes = {k: 1.0 for k in self.track_keys}
        self.track_pans    = {k: 0.0 for k in self.track_keys}
        self.track_speeds  = {k: 1.0 for k in self.track_keys}

        # Zoom state
        self._zoom_idx  = _ZOOM_DEFAULT_IDX
        self.px_per_sec = _ZOOM_LEVELS[self._zoom_idx]

        # Scroll offset (pixels from the left edge of the content area)
        self._scroll_x  = 0

        # Interaction state — move drag
        self.active_clip_id  = None
        self.selected_clip_ids: set = set()
        self.playhead_time   = 0.0
        self._drag = {"id": None, "start_x": 0, "orig_start": 0.0, "moved": False}

        # Interaction state — resize drag
        self._resize_mode  = None          # "trim_in" | "trim_out" | None
        self._resize_drag  = {
            "id": None,
            "start_x": 0,
            "orig_trim_in": 0.0,
            "orig_trim_out": 0.0,
            "orig_start": 0.0,
            "active": False,
        }

        # Interaction state — FL Studio rotary knob drag
        self._knob_drag = None

        # Dirty-flag for full vs. playhead-only redraws
        self._needs_full_redraw = True

        # Solo / mute state per track
        self.track_muted  = {k: False for k in self.track_keys}   # {track_key: bool}
        self.track_solo   = {k: False for k in self.track_keys}    # {track_key: bool}

        # Timeline markers
        self.markers      = []   # list of {'t': float, 'label': str, 'color': str}

        # Work area
        self.work_in      = 0.0  # render work area start
        self.work_out     = None # render work area end (None = full sequence)

        # Video thumbnail cache and image references
        self.thumb_cache  = timeline_thumb_cache
        self._thumb_refs  = []
        self._thumb_redraw_pending = False

        self._mute_btns   = {}
        self._solo_btns   = {}
        self._sm_frame    = None

        self._build_ui()

    def _on_thumb_ready(self, path):
        """Thread-safe notification that background thumbnail extraction completed."""
        try:
            self.after_idle(self._flush_thumb_redraw)
        except Exception:
            pass

    def _flush_thumb_redraw(self):
        if not getattr(self, '_thumb_redraw_pending', False):
            self._thumb_redraw_pending = True
            def _do():
                self._thumb_redraw_pending = False
                self._needs_full_redraw = True
                self.draw()
            try:
                self.after(60, _do)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── Toolbar ──────────────────────────────────────────────────
        tb = tk.Frame(self, bg="#111111", height=36)
        tb.pack(side="top", fill="x")
        tb.pack_propagate(False)

        tk.Label(tb, text="Timeline", bg="#111111", fg="#cccccc",
                 font=("Segoe UI", 11, "bold")).pack(side="left", padx=(10, 8))

        tk.Button(tb, text="−", bg="#252830", fg="white", relief="flat",
                  width=3, font=("Segoe UI", 10, "bold"), command=self._zoom_out,
                  activebackground="#3b4252", activeforeground="white"
                  ).pack(side="left", padx=(2, 0), pady=4)
        self._zoom_lbl = tk.Label(tb, text=self._zoom_label(), bg="#111111",
                                  fg="#cccccc", width=8, font=("Segoe UI", 9, "bold"))
        self._zoom_lbl.pack(side="left", pady=4)
        tk.Button(tb, text="+", bg="#252830", fg="white", relief="flat",
                  width=3, font=("Segoe UI", 10, "bold"), command=self._zoom_in,
                  activebackground="#3b4252", activeforeground="white"
                  ).pack(side="left", padx=(0, 6), pady=4)

        tk.Button(tb, text="⟨⟨ Reset Zoom", bg="#252830", fg="#dddddd",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  command=self._zoom_reset,
                  activebackground="#3b4252", activeforeground="white"
                  ).pack(side="left", padx=4, pady=4)

        # ── Track Add Buttons ─────────────────────────────────────────
        tk.Button(tb, text="+ Video", bg="#1b4f72", fg="#ebf5fb",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  command=self.add_video_track,
                  activebackground="#2874a6", activeforeground="white"
                  ).pack(side="left", padx=(8, 2), pady=4)

        tk.Button(tb, text="+ Audio", bg="#145a32", fg="#eafaf1",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  command=self.add_audio_track,
                  activebackground="#1e8449", activeforeground="white"
                  ).pack(side="left", padx=(2, 2), pady=4)

        tk.Button(tb, text="+ Subtitle", bg="#7d6608", fg="#fef9e7",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  command=self.add_subtitle_track,
                  activebackground="#9a7d0a", activeforeground="white"
                  ).pack(side="left", padx=(2, 2), pady=4)

        tk.Button(tb, text="+ Image", bg="#880e4f", fg="#fce4ec",
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  command=self.add_image_track,
                  activebackground="#ad1457", activeforeground="white"
                  ).pack(side="left", padx=(2, 8), pady=4)

        # ── Solo/Mute buttons per track (dynamic) ──────────────────────
        self._sm_frame = tk.Frame(tb, bg="#111111")
        self._sm_frame.pack(side="left", padx=(6, 4))
        self._rebuild_solo_mute_toolbar()

        # ── Canvas ───────────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg="#1a1a1a")
        canvas_frame.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#1a1a1a",
                                highlightthickness=0, cursor="arrow")
        self.canvas.pack(side="top", fill="both", expand=True)

        # Horizontal scrollbar
        self._hbar = tk.Scrollbar(canvas_frame, orient="horizontal",
                                   command=self._on_hscroll,
                                   bg="#222222", troughcolor="#1a1a1a")
        self._hbar.pack(side="bottom", fill="x")

        # Canvas bindings
        self.canvas.bind("<Configure>",       self._on_configure)
        self.canvas.bind("<Button-1>",        self._on_press)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>",        self._on_right_click)
        self.canvas.bind("<Motion>",          self._on_hover)

        # Mouse-wheel (platform-aware)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(seq, self._on_wheel)

    def _rebuild_solo_mute_toolbar(self):
        """Reconstruct the M/S track toggles inside toolbar."""
        if not self._sm_frame:
            return
        for child in self._sm_frame.winfo_children():
            child.destroy()
        self._mute_btns.clear()
        self._solo_btns.clear()

        for tk_key in self.track_keys:
            col_frame = tk.Frame(self._sm_frame, bg="#111111")
            col_frame.pack(side="left", padx=2)

            lbl = tk.Label(col_frame, text=tk_key, bg="#111111", fg="#8892a0",
                           font=("Segoe UI", 8, "bold"), width=3)
            lbl.pack(side="top")

            btn_row = tk.Frame(col_frame, bg="#111111")
            btn_row.pack(side="top")

            m_btn = tk.Button(btn_row, text="M", bg="#252830", fg="#aaaaaa",
                              relief="flat", width=2, font=("Segoe UI", 8, "bold"),
                              activebackground="#3b4252", activeforeground="white",
                              command=lambda k=tk_key: self.toggle_mute(k))
            m_btn.pack(side="left", padx=1)

            s_btn = tk.Button(btn_row, text="S", bg="#252830", fg="#aaaaaa",
                              relief="flat", width=2, font=("Segoe UI", 8, "bold"),
                              activebackground="#3b4252", activeforeground="white",
                              command=lambda k=tk_key: self.toggle_solo(k))
            s_btn.pack(side="left", padx=1)

            self._mute_btns[tk_key] = m_btn
            self._solo_btns[tk_key] = s_btn

        self._update_solo_mute_buttons()

    def _update_solo_mute_buttons(self):
        """Refresh solo/mute button colours to reflect current state."""
        for k in self.track_keys:
            mb = self._mute_btns.get(k)
            sb = self._solo_btns.get(k)
            if mb:
                mb.configure(bg="#c0392b" if self.track_muted.get(k) else "#222222",
                             fg="white" if self.track_muted.get(k) else "#aaaaaa")
            if sb:
                sb.configure(bg="#f1c40f" if self.track_solo.get(k) else "#222222",
                             fg="#111111" if self.track_solo.get(k) else "#aaaaaa")

    # ------------------------------------------------------------------
    # Zoom helpers
    # ------------------------------------------------------------------
    def _zoom_label(self):
        if self.px_per_sec < 1.0:
            return f"{self.px_per_sec:.1f}px/s"
        return f"{int(self.px_per_sec) if self.px_per_sec == int(self.px_per_sec) else self.px_per_sec}px/s"

    def _zoom_in(self):
        if self._zoom_idx < len(_ZOOM_LEVELS) - 1:
            self._zoom_idx += 1
            self.px_per_sec = _ZOOM_LEVELS[self._zoom_idx]
            self._zoom_lbl.configure(text=self._zoom_label())
            self._needs_full_redraw = True
            self.draw()

    def _zoom_out(self):
        if self._zoom_idx > 0:
            self._zoom_idx -= 1
            self.px_per_sec = _ZOOM_LEVELS[self._zoom_idx]
            self._zoom_lbl.configure(text=self._zoom_label())
            self._needs_full_redraw = True
            self.draw()

    def _zoom_reset(self):
        self._zoom_idx  = _ZOOM_DEFAULT_IDX
        self.px_per_sec = _ZOOM_LEVELS[self._zoom_idx]
        self._scroll_x  = 0
        self._zoom_lbl.configure(text=self._zoom_label())
        self._needs_full_redraw = True
        self.draw()

    def _zoom_at(self, cursor_x, direction):
        """Zoom centred on *cursor_x* (canvas pixel)."""
        t_under = max(0.0, (cursor_x + self._scroll_x - _TRACK_LABEL_W) / self.px_per_sec)
        if direction > 0:
            self._zoom_in()
        else:
            self._zoom_out()
        new_x = t_under * self.px_per_sec + _TRACK_LABEL_W - cursor_x
        self._scroll_x = max(0, new_x)
        self._zoom_lbl.configure(text=self._zoom_label())
        self._needs_full_redraw = True
        self.draw()

    # ------------------------------------------------------------------
    # Scroll helpers
    # ------------------------------------------------------------------
    def get_effective_clip_duration(self, clip: dict) -> float:
        """Return the rendered timeline duration of a clip accounting for clip speed and track speed."""
        trk = clip.get('track', 'V1')
        trk_spd = self.track_speeds.get(trk, 1.0)
        clip_spd = float(clip.get('speed', 1.0) or 1.0)
        eff_spd = trk_spd * clip_spd
        if eff_spd <= 0:
            eff_spd = 1.0
        trim_in = float(clip.get('trim_in', 0.0))
        trim_out = float(clip.get('trim_out', 3.0))
        return max(0.01, (trim_out - trim_in) / eff_spd)

    def _max_scroll(self):
        if not self.app.project_clips:
            return 0
        max_t = max(c['start'] + self.get_effective_clip_duration(c)
                    for c in self.app.project_clips)
        content_w = _TRACK_LABEL_W + max_t * self.px_per_sec + 200
        visible_w = self.canvas.winfo_width()
        return max(0, content_w - visible_w)

    def _clamp_scroll(self):
        self._scroll_x = max(0, min(self._scroll_x, self._max_scroll()))

    def _on_hscroll(self, *args):
        if args[0] == "moveto":
            frac = float(args[1])
            self._scroll_x = frac * max(1, self._max_scroll() + self.canvas.winfo_width())
        elif args[0] == "scroll":
            self._scroll_x += int(args[1]) * 40
        self._clamp_scroll()
        self._needs_full_redraw = True
        self.draw()

    def _update_scrollbar(self):
        total = max(1, self._max_scroll() + self.canvas.winfo_width())
        vis   = self.canvas.winfo_width()
        lo    = self._scroll_x / total
        hi    = (self._scroll_x + vis) / total
        self._hbar.set(lo, hi)

    # ------------------------------------------------------------------
    # Mouse-wheel dispatch
    # ------------------------------------------------------------------
    def _on_wheel(self, event):
        if event.num == 4:       delta = 1
        elif event.num == 5:     delta = -1
        else:                    delta = 1 if event.delta > 0 else -1

        hit = self._hit_test_knob(event.x, event.y)
        if hit:
            tk_key, knob_type, _, _ = hit
            if knob_type == 'vol':
                curr = self.track_volumes.get(tk_key, 1.0)
                self.set_track_volume(tk_key, round(curr + 0.05 * delta, 2))
            elif knob_type == 'pan':
                curr = self.track_pans.get(tk_key, 0.0)
                self.set_track_pan(tk_key, round(curr + 0.05 * delta, 2))
            elif knob_type == 'spd':
                curr = self.track_speeds.get(tk_key, 1.0)
                self.set_track_speed(tk_key, round(curr + 0.05 * delta, 2))
            return

        if event.state & 0x0004:   # Ctrl held → zoom
            self._zoom_at(event.x, delta)
        else:                       # horizontal scroll
            self._scroll_x -= delta * 60
            self._clamp_scroll()
            self._needs_full_redraw = True
            self.draw()

    def _on_double_click(self, event):
        """Reset knob to default on double-click (FL Studio standard)."""
        hit = self._hit_test_knob(event.x, event.y)
        if hit:
            tk_key, knob_type, _, _ = hit
            if knob_type == 'vol':
                self.set_track_volume(tk_key, 1.0)
            elif knob_type == 'pan':
                self.set_track_pan(tk_key, 0.0)
            elif knob_type == 'spd':
                self.set_track_speed(tk_key, 1.0)

    # ------------------------------------------------------------------
    # Configure (resize)
    # ------------------------------------------------------------------
    def _on_configure(self, event=None):
        self._needs_full_redraw = True
        self.draw(event)

    # ------------------------------------------------------------------
    # Time/pixel conversion (accounts for scroll offset)
    # ------------------------------------------------------------------
    def _x_to_time(self, canvas_x):
        return max(0.0, (canvas_x + self._scroll_x - _TRACK_LABEL_W) / self.px_per_sec)

    def _time_to_x(self, t):
        return _TRACK_LABEL_W + t * self.px_per_sec - self._scroll_x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_playhead(self, t):
        self.playhead_time = max(0.0, t)
        if self._needs_full_redraw:
            self.draw()
        else:
            self.update_playhead_only()

    def update_playhead_only(self):
        """Erase and redraw just the playhead items (no full canvas clear)."""
        c = self.canvas
        c.delete("playhead")
        w = c.winfo_width()
        h = c.winfo_height()
        tracks_h = len(self.tracks) * self.track_h
        ruler_y  = tracks_h
        px = self._time_to_x(self.playhead_time)
        if _TRACK_LABEL_W <= px <= w:
            c.create_line(px, 0, px, ruler_y + self.ruler_h,
                          fill="#ff4444", width=2, tags="playhead")
            c.create_polygon(
                px - 6, ruler_y,
                px + 6, ruler_y,
                px + 6, ruler_y + 8,
                px,     ruler_y + self.ruler_h - 2,
                px - 6, ruler_y + 8,
                fill="#ff4444", outline="#ff0000",
                tags="playhead",
            )

    def add_marker(self, t: float, label: str = '', color: str = '#f39c12'):
        """Add a named marker at time t."""
        self.markers.append({'t': t, 'label': label, 'color': color})
        self.markers.sort(key=lambda m: m['t'])
        self.draw()

    def remove_marker_at(self, t: float, tolerance: float = 0.2):
        """Remove the nearest marker within tolerance seconds of t."""
        if not self.markers:
            return
        nearest = min(self.markers, key=lambda m: abs(m['t'] - t))
        if abs(nearest['t'] - t) <= tolerance:
            self.markers.remove(nearest)
            self.draw()

    def set_work_area(self, in_t: float, out_t):
        self.work_in = max(0.0, in_t)
        self.work_out = out_t
        self.draw()

    # ------------------------------------------------------------------
    # Dynamic Track Management & Mixer Controls
    # ------------------------------------------------------------------
    def add_video_track(self):
        """Add a new video track at the top of the video stack."""
        v_indices = [int(k[1:]) for k in self.track_keys if k.startswith('V') and k[1:].isdigit()]
        next_v = (max(v_indices) + 1) if v_indices else 1
        tk_key = f"V{next_v}"
        tk_name = f"V{next_v} (Video {next_v})"

        # Insert at the top of video tracks
        self.track_keys.insert(0, tk_key)
        self.tracks.insert(0, tk_name)
        self.track_volumes[tk_key] = 1.0
        self.track_pans[tk_key]    = 0.0
        self.track_speeds[tk_key]  = 1.0
        self.track_muted[tk_key]   = False
        self.track_solo[tk_key]    = False

        self._rebuild_solo_mute_toolbar()
        if hasattr(self.app, 'refresh_track_selectors'):
            self.app.refresh_track_selectors()
        self._needs_full_redraw = True
        self.draw()
        self.app.rebuild_proxy_cache()

    def add_audio_track(self):
        """Add a new audio track at the bottom of the audio stack."""
        a_indices = [int(k[1:]) for k in self.track_keys if k.startswith('A') and k[1:].isdigit()]
        next_a = (max(a_indices) + 1) if a_indices else 1
        tk_key = f"A{next_a}"
        tk_name = f"A{next_a} (Audio {next_a})"

        self.track_keys.append(tk_key)
        self.tracks.append(tk_name)
        self.track_volumes[tk_key] = 1.0
        self.track_pans[tk_key]    = 0.0
        self.track_speeds[tk_key]  = 1.0
        self.track_muted[tk_key]   = False
        self.track_solo[tk_key]    = False

        self._rebuild_solo_mute_toolbar()
        if hasattr(self.app, 'refresh_track_selectors'):
            self.app.refresh_track_selectors()
        self._needs_full_redraw = True
        self.draw()
        self.app.rebuild_proxy_cache()

    def add_subtitle_track(self):
        """Add a new subtitle track at the top of the stack."""
        s_indices = [int(k[1:]) for k in self.track_keys if (k.startswith('S') or k.startswith('T')) and k[1:].isdigit()]
        next_s = (max(s_indices) + 1) if s_indices else 1
        tk_key = f"S{next_s}"
        tk_name = f"S{next_s} (Subtitle {next_s})"

        # Insert at the top of the timeline
        self.track_keys.insert(0, tk_key)
        self.tracks.insert(0, tk_name)
        self.track_volumes[tk_key] = 1.0
        self.track_pans[tk_key]    = 0.0
        self.track_speeds[tk_key]  = 1.0
        self.track_muted[tk_key]   = False
        self.track_solo[tk_key]    = False

        self._rebuild_solo_mute_toolbar()
        if hasattr(self.app, 'refresh_track_selectors'):
            self.app.refresh_track_selectors()
        self._needs_full_redraw = True
        self.draw()
        self.app.rebuild_proxy_cache()

    def add_image_track(self):
        """Add a new image track above video tracks."""
        i_indices = [int(k[1:]) for k in self.track_keys if k.startswith('I') and k[1:].isdigit()]
        next_i = (max(i_indices) + 1) if i_indices else 1
        tk_key = f"I{next_i}"
        tk_name = f"I{next_i} (Image {next_i})"

        # Insert above first video or audio track, or at beginning
        ins_idx = 0
        for idx, k in enumerate(self.track_keys):
            if k.startswith('V') or k.startswith('A'):
                ins_idx = idx
                break

        self.track_keys.insert(ins_idx, tk_key)
        self.tracks.insert(ins_idx, tk_name)
        self.track_volumes[tk_key] = 1.0
        self.track_pans[tk_key]    = 0.0
        self.track_speeds[tk_key]  = 1.0
        self.track_muted[tk_key]   = False
        self.track_solo[tk_key]    = False

        self._rebuild_solo_mute_toolbar()
        if hasattr(self.app, 'refresh_track_selectors'):
            self.app.refresh_track_selectors()
        self._needs_full_redraw = True
        self.draw()
        self.app.rebuild_proxy_cache()

    def remove_track(self, track_key: str):
        """Remove a track and its clips (keeping at least 1 video and 1 audio track)."""
        if track_key not in self.track_keys:
            return
        v_count = sum(1 for k in self.track_keys if k.startswith('V'))
        a_count = sum(1 for k in self.track_keys if k.startswith('A'))
        if track_key.startswith('V') and v_count <= 1:
            return
        if track_key.startswith('A') and a_count <= 1:
            return

        idx = self.track_keys.index(track_key)
        self.track_keys.pop(idx)
        self.tracks.pop(idx)
        self.track_volumes.pop(track_key, None)
        self.track_pans.pop(track_key, None)
        self.track_speeds.pop(track_key, None)
        self.track_muted.pop(track_key, None)
        self.track_solo.pop(track_key, None)

        # Remove clips associated with this track
        self.app.project_clips = [c for c in self.app.project_clips if c.get('track') != track_key]

        self._rebuild_solo_mute_toolbar()
        if hasattr(self.app, 'refresh_track_selectors'):
            self.app.refresh_track_selectors()
        self._needs_full_redraw = True
        self.draw()
        self.app.rebuild_proxy_cache()

    def set_track_volume(self, track_key: str, vol: float):
        """Set track volume multiplier (0.0 to 2.0)."""
        self.track_volumes[track_key] = max(0.0, min(float(vol), 2.0))
        self._needs_full_redraw = True
        self.draw()
        self.app.rebuild_proxy_cache()

    def set_track_pan(self, track_key: str, pan: float):
        """Set track stereo panning (-1.0 left to +1.0 right)."""
        self.track_pans[track_key] = max(-1.0, min(float(pan), 1.0))
        self._needs_full_redraw = True
        self.draw()
        self.app.rebuild_proxy_cache()

    def set_track_speed(self, track_key: str, spd: float):
        """Set track speed multiplier (0.25 to 4.0)."""
        self.track_speeds[track_key] = max(0.25, min(float(spd), 4.0))
        self._needs_full_redraw = True
        self.draw()
        self.app.rebuild_proxy_cache()

    def toggle_mute(self, track_key: str):
        self.track_muted[track_key] = not self.track_muted.get(track_key, False)
        self._update_solo_mute_buttons()
        self.draw()
        self.app.rebuild_proxy_cache()

    def toggle_solo(self, track_key: str):
        self.track_solo[track_key] = not self.track_solo.get(track_key, False)
        self._update_solo_mute_buttons()
        self.draw()
        self.app.rebuild_proxy_cache()

    def get_active_tracks(self) -> list:
        """Return track keys that are active (not muted, or soloed)."""
        if any(self.track_solo.values()):
            return [k for k, v in self.track_solo.items() if v]
        return [k for k, v in self.track_muted.items() if not v]

    def _hit_test_knob(self, x: float, y: float):
        """Hit test for FL Studio volume, pan, and speed rotary knobs in the gutter."""
        if x > _TRACK_LABEL_W:
            return None
        trk_idx = int(y // self.track_h)
        if 0 <= trk_idx < len(self.track_keys):
            tk_key = self.track_keys[trk_idx]
            y0 = trk_idx * self.track_h
            cy = y0 + 44
            # Vol knob center cx=46, cy=y0+44, hit radius 16
            if (x - 46) ** 2 + (y - cy) ** 2 <= 16 ** 2:
                return (tk_key, 'vol', 46, cy)
            # Pan knob center cx=116, cy=y0+44, hit radius 16
            if (x - 116) ** 2 + (y - cy) ** 2 <= 16 ** 2:
                return (tk_key, 'pan', 116, cy)
            # Speed knob center cx=186, cy=y0+44, hit radius 16
            if (x - 186) ** 2 + (y - cy) ** 2 <= 16 ** 2:
                return (tk_key, 'spd', 186, cy)
        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, event=None):
        c = self.canvas
        c.delete("all")
        self._thumb_refs = []
        w = c.winfo_width()
        h = c.winfo_height()
        if w <= 1 or h <= 1:
            return

        tracks_h = len(self.tracks) * self.track_h
        any_solo = any(self.track_solo.values())

        # ── Track backgrounds & Gutters ───────────────────────────
        for i, trk_name in enumerate(self.tracks):
            y        = i * self.track_h
            tk_key   = self.track_keys[i]
            theme    = get_track_theme(tk_key)
            is_muted = self.track_muted.get(tk_key, False)
            is_solo  = self.track_solo.get(tk_key, False)

            if any_solo and not is_solo:
                bg = "#0c0c0e"
            elif is_muted and not any_solo:
                bg = "#111114"
            else:
                bg = theme['lane_bg']

            # Track lane background
            c.create_rectangle(0, y, w, y + self.track_h,
                                fill=bg, outline="", tags="static")

            # Gutter background
            c.create_rectangle(0, y, _TRACK_LABEL_W, y + self.track_h,
                                fill="#15161a", outline="", tags="static")

            # Left vertical signature accent bar
            c.create_rectangle(0, y, 5, y + self.track_h,
                                fill=theme['accent'], outline="", tags="static")

            # Gutter divider
            c.create_line(_TRACK_LABEL_W, y, _TRACK_LABEL_W, y + self.track_h,
                          fill="#38383e", tags="static")

            # Track badge [V1], [A1], etc.
            c.create_rectangle(8, y + 7, 44, y + 27,
                                fill=theme['badge_bg'], outline=theme['accent'],
                                width=1, tags="static")
            c.create_text(26, y + 17, text=tk_key,
                          fill=theme['badge_fg'], font=("Segoe UI", 8, "bold"),
                          tags="static")

            # Track label text
            c.create_text(52, y + 17,
                          text=trk_name, anchor="w",
                          fill="#e4e4e7", font=("Segoe UI", 9, "bold"),
                          tags="static")

            # FL Studio style Volume, Panning, and Speed Rotary Knobs
            vol_val = self.track_volumes.get(tk_key, 1.0)
            pan_val = self.track_pans.get(tk_key, 0.0)
            spd_val = self.track_speeds.get(tk_key, 1.0)
            _draw_fl_knob(
                canvas=c, cx=46, cy=y + 44, radius=10.0,
                value=vol_val, min_val=0.0, max_val=2.0, default_val=1.0,
                is_bipolar=False, label="VOL", accent_color="#00d2be"
            )
            _draw_fl_knob(
                canvas=c, cx=116, cy=y + 44, radius=10.0,
                value=pan_val, min_val=-1.0, max_val=1.0, default_val=0.0,
                is_bipolar=True, label="PAN", accent_color="#ff9900"
            )
            _draw_fl_knob(
                canvas=c, cx=186, cy=y + 44, radius=10.0,
                value=spd_val, min_val=0.25, max_val=4.0, default_val=1.0,
                is_bipolar=False, label="SPD", accent_color="#a855f7"
            )

            # Delete track button [×] on far right of gutter
            c.create_text(_TRACK_LABEL_W - 14, y + 17, text="✕",
                          fill="#777777", activefill="#e74c3c", font=("Segoe UI", 10, "bold"),
                          tags="static")

            # Mute / Solo badges in gutter
            if is_muted and not any_solo:
                c.create_text(_TRACK_LABEL_W - 32, y + 17,
                              text="M", anchor="e",
                              fill="#e74c3c", font=("Segoe UI", 8, "bold"),
                              tags="static")
            if is_solo:
                c.create_rectangle(_TRACK_LABEL_W, y, _TRACK_LABEL_W + 3,
                                   y + self.track_h,
                                   fill="#f1c40f", outline="", tags="static")

            # Track horizontal separator
            c.create_line(0, y + self.track_h, w, y + self.track_h,
                          fill="#27272a", tags="static")

        # ── Clips ──────────────────────────────────────────────────
        for clip in self.app.project_clips:
            try:
                trk_idx = self.track_keys.index(clip['track'])
            except ValueError:
                continue
            duration = self.get_effective_clip_duration(clip)
            x1 = self._time_to_x(clip['start'])
            x2 = self._time_to_x(clip['start'] + duration)
            if x2 < _TRACK_LABEL_W or x1 > w:
                continue
            x1 = max(x1, _TRACK_LABEL_W)

            y1 = trk_idx * self.track_h + 4
            y2 = y1 + self.track_h - 8

            clip_id  = clip['id']
            is_sel   = clip_id in self.selected_clip_ids
            trk_key  = clip.get('track', 'V1')
            theme    = get_track_theme(trk_key)

            # ── Colour label ──
            label_name = clip.get('label')
            if label_name and label_name in _LABEL_COLORS:
                base_col = _LABEL_COLORS[label_name]
                r = int(base_col[1:3], 16)
                g = int(base_col[3:5], 16)
                b = int(base_col[5:7], 16)
                hi_col = "#{:02x}{:02x}{:02x}".format(
                    min(255, r + 50), min(255, g + 50), min(255, b + 50))
            else:
                base_col = theme['clip_base']
                hi_col   = theme['clip_hi']

            fill_col    = hi_col if is_sel else base_col
            outline_col = "#ffffff" if is_sel else theme['accent']
            outline_w   = 2 if is_sel else 1

            # Clip body base
            c.create_rectangle(x1, y1, x2, y2,
                                fill=fill_col,
                                outline=outline_col,
                                width=outline_w,
                                tags="static")

            clip_pix_w = x2 - x1
            clip_pix_h = y2 - y1

            # ── Periodic Video Thumbnails Filmstrip ──
            is_video = (trk_key.startswith('V') or clip.get('path', '').lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv'))) and clip.get('path') and not clip.get('path') == '__title__' and os.path.isfile(clip.get('path', ''))
            if is_video and _HAS_CV2_PIL and clip_pix_w > 10:
                tile_w = max(55, min(100, int(clip_pix_h * 1.5)))
                num_tiles = max(1, int(clip_pix_w / tile_w))
                tin  = float(clip.get('trim_in', 0.0))
                tout = float(clip.get('trim_out', tin + 1.0))

                for t_idx in range(num_tiles):
                    tx1 = x1 + t_idx * tile_w
                    tx2 = min(x2, tx1 + tile_w)
                    if tx2 <= _TRACK_LABEL_W or tx1 >= w:
                        continue
                    actual_x1 = max(tx1, _TRACK_LABEL_W)
                    actual_w  = int(tx2 - actual_x1)
                    actual_h  = int(clip_pix_h)
                    if actual_w <= 4 or actual_h <= 4:
                        continue

                    t_norm  = (t_idx + 0.5) / num_tiles
                    t_media = tin + t_norm * max(0.01, tout - tin)
                    photo   = self.thumb_cache.get(clip['path'], t_media, actual_w, actual_h, callback=self._on_thumb_ready)
                    if photo:
                        c.create_image(actual_x1, y1, image=photo, anchor="nw", tags="static")
                        self._thumb_refs.append(photo)

                    # Filmstrip vertical frame divider
                    if t_idx > 0 and tx1 > _TRACK_LABEL_W:
                        c.create_line(tx1, y1, tx1, y2, fill="#18181b", width=1.5, tags="static")

                # Semi-transparent dark header banner across top of video clip for readable title & timecode
                header_h = min(22, max(16, clip_pix_h // 2))
                c.create_rectangle(x1, y1, x2, y1 + header_h, fill="#0d1117", outline="", tags="static")
                c.create_line(x1, y1 + header_h, x2, y1 + header_h, fill="#21262d", width=1, tags="static")
            else:
                # Top highlight stripe
                c.create_rectangle(x1, y1, x2, y1 + 3,
                                    fill="#555555" if not is_sel else "#888888",
                                    outline="", tags="static")

            # ── Audio waveform ──
            if 'A' in clip.get('track', ''):
                width_px = max(1, int(x2 - x1))
                wf = waveform_cache.get(
                    clip['path'],
                    width_px,
                    callback=lambda p: self.after_idle(self.draw),
                )
                if wf is not None:
                    waveform_cache.draw_waveform(
                        c, wf, x1, y1, x2, y2,
                        color=theme['wf_color'], tag="waveform",
                    )

            # ── Resize / Trim Grab Handles ──
            handle_w = 6
            c.create_rectangle(x1, y1, x1 + handle_w, y2,
                                fill="#ffffff" if is_sel else theme['accent'],
                                outline="", tags="static")
            c.create_line(x1 + 3, y1 + 10, x1 + 3, y2 - 10, fill="#000000" if is_sel else "#ffffff", width=1.5, tags="static")

            c.create_rectangle(x2 - handle_w, y1, x2, y2,
                                fill="#ffffff" if is_sel else theme['accent'],
                                outline="", tags="static")
            c.create_line(x2 - 3, y1 + 10, x2 - 3, y2 - 10, fill="#000000" if is_sel else "#ffffff", width=1.5, tags="static")

            # Clip label (truncated to fit)
            trk_key = clip.get('track', 'V1')
            if trk_key.startswith(('S', 'T')) or clip.get('path') == '__title__':
                txt = clip.get('title_text') or (clip.get('title_config', {}).get('text')) or 'Subtitle'
                prefix = '💬 ' if trk_key.startswith('S') else '🔤 '
                disp_name = prefix + txt
            elif trk_key.startswith('I'):
                disp_name = '🖼️ ' + os.path.basename(clip.get('path', 'Image'))
            elif trk_key.startswith('A'):
                disp_name = '🎵 ' + os.path.basename(clip.get('path', 'Audio'))
            else:
                disp_name = '🎬 ' + os.path.basename(clip.get('path', 'Video'))

            pix_w   = x2 - x1
            max_ch  = max(0, int(pix_w / 7) - 3)
            disp    = (disp_name[:max_ch - 1] + "…") if len(disp_name) > max_ch and max_ch > 3 else disp_name
            label_y = y1 + 10 if is_video else y1 + (self.track_h - 8) // 2
            if pix_w > 24:
                c.create_text(x1 + 10, label_y,
                              text=disp, anchor="w",
                              fill="white", font=("Segoe UI", 9, "bold"),
                              tags="static")

            # Linked clip 🔗 badge
            if clip.get('linked_clip_id'):
                if pix_w > 40:
                    badge_x = x2 - 18
                    badge_y = y1 + 10
                    c.create_text(badge_x, badge_y, text="🔗",
                                  fill="#f1c40f", font=("Segoe UI", 9, "bold"),
                                  tags="static")

            # Duration badge
            trk_spd = self.track_speeds.get(trk_key, 1.0)
            clip_spd = float(clip.get('speed', 1.0) or 1.0)
            eff_spd = trk_spd * clip_spd
            dur_txt = f"{duration:.1f}s ({eff_spd:.2f}x)" if abs(eff_spd - 1.0) > 0.02 else f"{duration:.1f}s"
            if pix_w > 75:
                dur_y = y1 + 10 if is_video else y1 + (self.track_h - 8) // 2
                c.create_text(x2 - (32 if clip.get('linked_clip_id') else 10),
                              dur_y,
                              text=dur_txt, anchor="e",
                              fill="#dddddd", font=("Segoe UI", 8, "bold"),
                              tags="static")

            # Mute/solo state line at bottom of clip
            is_muted = self.track_muted.get(trk_key, False)
            is_solo  = self.track_solo.get(trk_key, False)
            if is_muted and not any_solo:
                c.create_line(x1, y2, x2, y2, fill="#c0392b", width=2, tags="static")
            elif any_solo and is_solo:
                c.create_line(x1, y2, x2, y2, fill="#f1c40f", width=2, tags="static")

        # ── Timeline markers ───────────────────────────────────────
        for marker in self.markers:
            mx = self._time_to_x(marker['t'])
            if _TRACK_LABEL_W <= mx <= w:
                mc = marker.get('color', '#f39c12')
                c.create_polygon(mx, 2, mx+10, 6, mx, 10, fill=mc, outline='')
                c.create_line(mx, 0, mx, tracks_h, fill=mc, dash=(2, 3), width=1)
                if marker.get('label'):
                    c.create_text(mx+12, 6, text=marker['label'][:12],
                                  anchor='w', fill=mc, font=('Segoe UI', 8, 'bold'))

        # ── Ruler ──────────────────────────────────────────────────
        ruler_y = tracks_h
        c.create_rectangle(0, ruler_y, w, ruler_y + self.ruler_h,
                            fill="#0e0e0e", outline="", tags="static")
        c.create_line(0, ruler_y, w, ruler_y, fill="#444444", tags="static")

        work_end = self.work_out if self.work_out else (self._x_to_time(w) + 1)
        wx1 = max(_TRACK_LABEL_W, self._time_to_x(self.work_in))
        wx2 = min(w, self._time_to_x(work_end))
        if wx2 > wx1:
            c.create_rectangle(wx1, ruler_y, wx2, ruler_y + self.ruler_h,
                                fill='#252525', outline='')
            c.create_line(wx1, ruler_y, wx1, ruler_y + self.ruler_h, fill='#f1c40f', width=2)
            c.create_line(wx2, ruler_y, wx2, ruler_y + self.ruler_h, fill='#f1c40f', width=2)

        if self.px_per_sec >= 90:    major_s, minor_s = 1, 0.5
        elif self.px_per_sec >= 40:  major_s, minor_s = 5, 1
        elif self.px_per_sec >= 18:  major_s, minor_s = 10, 5
        elif self.px_per_sec >= 8:   major_s, minor_s = 30, 10
        elif self.px_per_sec >= 3:   major_s, minor_s = 60, 30
        elif self.px_per_sec >= 1.0: major_s, minor_s = 300, 60
        elif self.px_per_sec >= 0.4: major_s, minor_s = 600, 120
        else:                        major_s, minor_s = 1800, 300

        t = 0.0
        while True:
            x = self._time_to_x(t)
            if x > w:
                break
            if x >= _TRACK_LABEL_W:
                c.create_line(x, ruler_y + self.ruler_h - 6,
                              x, ruler_y + self.ruler_h,
                              fill="#555555", tags="static")
            t += minor_s

        t = 0.0
        while True:
            x = self._time_to_x(t)
            if x > w:
                break
            if x >= _TRACK_LABEL_W:
                c.create_line(x, ruler_y, x, ruler_y + self.ruler_h,
                              fill="#777777", tags="static")
                mins, secs = divmod(t, 60)
                hours, mins = divmod(mins, 60)
                if hours >= 1:
                    lbl = f"{int(hours)}:{int(mins):02d}:{secs:02.0f}"
                elif mins >= 1:
                    lbl = f"{int(mins)}:{secs:04.1f}" if minor_s < 1 else f"{int(mins)}:{int(secs):02d}"
                else:
                    lbl = f"{secs:.1f}s"
                c.create_text(x + 4, ruler_y + 11,
                              text=lbl, anchor="w",
                              fill="#aaaaaa", font=("Segoe UI", 8, "bold"),
                              tags="static")
            t += major_s

        # ── Playhead ───────────────────────────────────────────────
        px = self._time_to_x(self.playhead_time)
        if _TRACK_LABEL_W <= px <= w:
            c.create_line(px, 0, px, ruler_y + self.ruler_h,
                          fill="#ff4444", width=2, tags="playhead")
            c.create_polygon(
                px - 6, ruler_y,
                px + 6, ruler_y,
                px + 6, ruler_y + 8,
                px,     ruler_y + self.ruler_h - 2,
                px - 6, ruler_y + 8,
                fill="#ff4444", outline="#ff0000",
                tags="playhead",
            )

        self._needs_full_redraw = False
        self._update_scrollbar()

    # ------------------------------------------------------------------
    # Hover — resize cursor feedback
    # ------------------------------------------------------------------
    def _on_hover(self, event):
        tracks_h = len(self.tracks) * self.track_h
        if event.y >= tracks_h or event.x < _TRACK_LABEL_W:
            if event.x < _TRACK_LABEL_W and event.y < tracks_h:
                if self._hit_test_knob(event.x, event.y):
                    self.canvas.configure(cursor="sb_v_double_arrow")
                    self._resize_mode = None
                    return
            self.canvas.configure(cursor="arrow")
            self._resize_mode = None
            return

        for clip in self.app.project_clips:
            try:
                trk_idx = self.track_keys.index(clip['track'])
            except ValueError:
                continue
            duration = self.get_effective_clip_duration(clip)
            x1  = self._time_to_x(clip['start'])
            x2  = self._time_to_x(clip['start'] + duration)
            y1  = trk_idx * self.track_h + 4
            y2  = y1 + self.track_h - 8
            if not (y1 <= event.y <= y2 and x1 <= event.x <= x2):
                continue
            if abs(event.x - x1) <= _RESIZE_HIT:
                self.canvas.configure(cursor="sb_h_double_arrow")
                self._resize_mode = "trim_in"
                return
            if abs(event.x - x2) <= _RESIZE_HIT:
                self.canvas.configure(cursor="sb_h_double_arrow")
                self._resize_mode = "trim_out"
                return
            self.canvas.configure(cursor="arrow")
            self._resize_mode = None
            return

        self.canvas.configure(cursor="arrow")
        self._resize_mode = None

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------
    def _on_press(self, event):
        tracks_h = len(self.tracks) * self.track_h

        # Click on gutter (track header)
        if event.x < _TRACK_LABEL_W and event.y < tracks_h:
            # Check rotary knob hit first
            hit = self._hit_test_knob(event.x, event.y)
            if hit:
                tk_key, knob_type, _, _ = hit
                if knob_type == 'vol':
                    cur_val = self.track_volumes.get(tk_key, 1.0)
                elif knob_type == 'pan':
                    cur_val = self.track_pans.get(tk_key, 0.0)
                else:
                    cur_val = self.track_speeds.get(tk_key, 1.0)

                self._knob_drag = {
                    'key': tk_key,
                    'type': knob_type,
                    'start_y': event.y,
                    'start_val': cur_val,
                    'moved': False,
                }
                return

            trk_idx = int(event.y // self.track_h)
            if 0 <= trk_idx < len(self.track_keys):
                tk_key = self.track_keys[trk_idx]
                # If clicked delete button [✕]
                if event.x >= _TRACK_LABEL_W - 20:
                    self.remove_track(tk_key)
                else:
                    self._show_track_menu(event, tk_key)
            return

        # Click on ruler or below tracks → move playhead
        if event.y > tracks_h or event.x < _TRACK_LABEL_W:
            t = self._x_to_time(event.x)
            self.app.scrub_to(t)
            self.active_clip_id    = None
            self.selected_clip_ids = set()
            self.app.update_inspector()
            self._needs_full_redraw = True
            self.draw()
            return

        # Hit-test clips
        clicked_clip = None
        for clip in self.app.project_clips:
            try:
                trk_idx = self.track_keys.index(clip['track'])
            except ValueError:
                continue
            dur  = self.get_effective_clip_duration(clip)
            x1   = self._time_to_x(clip['start'])
            x2   = self._time_to_x(clip['start'] + dur)
            y1   = trk_idx * self.track_h + 4
            y2   = y1 + self.track_h - 8
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                clicked_clip = clip
                break

        if clicked_clip is None:
            self.active_clip_id    = None
            self.selected_clip_ids = set()
            self.app.update_inspector()
            self._needs_full_redraw = True
            self.draw()
            return

        clip_id = clicked_clip['id']

        # Click on 🔗 link badge
        dur = self.get_effective_clip_duration(clicked_clip)
        clip_x2 = self._time_to_x(clicked_clip['start'] + dur)
        clip_y1 = self.track_keys.index(clicked_clip['track']) * self.track_h + 4
        if clicked_clip.get('linked_clip_id') and (clip_x2 - 22 <= event.x <= clip_x2) and (clip_y1 <= event.y <= clip_y1 + 18):
            if hasattr(self.app, 'toggle_clip_link'):
                self.app.toggle_clip_link(clip_id)
            return

        # ── Resize mode ───────────────────────────────────────────
        if self._resize_mode in ("trim_in", "trim_out"):
            self._resize_drag.update({
                "id":           clip_id,
                "start_x":      event.x,
                "orig_trim_in":  clicked_clip['trim_in'],
                "orig_trim_out": clicked_clip['trim_out'],
                "orig_start":    clicked_clip['start'],
                "active":        True,
            })
            self.active_clip_id    = clip_id
            self.selected_clip_ids = {clip_id}
            self.app.update_inspector()
            self._needs_full_redraw = True
            self.draw()
            return

        # ── Move mode ─────────────────────────────────────────────
        shift_held = bool(event.state & 0x0001)
        if shift_held:
            if clip_id in self.selected_clip_ids:
                self.selected_clip_ids.discard(clip_id)
            else:
                self.selected_clip_ids.add(clip_id)
        else:
            self.selected_clip_ids = {clip_id}

        self.active_clip_id  = clip_id
        self._drag["id"]     = clip_id
        self._drag["start_x"] = event.x
        self._drag["orig_start"] = clicked_clip['start']
        self._drag["moved"]  = False

        self.app.update_inspector()
        self._needs_full_redraw = True
        self.draw()

    def _on_drag(self, event):
        # ── Knob drag ─────────────────────────────────────────────
        if self._knob_drag and self._knob_drag.get('key'):
            dy = self._knob_drag['start_y'] - event.y  # Drag up increases value
            tk_key = self._knob_drag['key']
            knob_type = self._knob_drag['type']
            start_val = self._knob_drag['start_val']
            self._knob_drag['moved'] = True

            if knob_type == 'vol':
                # Range 0.0 to 2.0 (sensitivity: 0.01 per pixel)
                new_val = max(0.0, min(2.0, start_val + dy * 0.01))
                self.track_volumes[tk_key] = round(new_val, 2)
            elif knob_type == 'pan':
                # Range -1.0 to 1.0 (sensitivity: 0.01 per pixel)
                new_val = max(-1.0, min(1.0, start_val + dy * 0.01))
                if abs(new_val) < 0.03:  # Subtle center snap detent
                    new_val = 0.0
                self.track_pans[tk_key] = round(new_val, 2)
            elif knob_type == 'spd':
                # Range 0.25 to 4.0 (sensitivity: 0.015 per pixel)
                new_val = max(0.25, min(4.0, start_val + dy * 0.015))
                if abs(new_val - 1.0) < 0.04:  # Subtle 1.00x snap detent
                    new_val = 1.0
                self.track_speeds[tk_key] = round(new_val, 2)

            self._needs_full_redraw = True
            self.draw()
            return

        # ── Resize drag ───────────────────────────────────────────
        if self._resize_drag["active"] and self._resize_drag["id"]:
            dx = event.x - self._resize_drag["start_x"]

            for clip in self.app.project_clips:
                if clip['id'] != self._resize_drag["id"]:
                    continue

                eff_spd = float(clip.get('speed', 1.0) or 1.0) * self.track_speeds.get(clip.get('track', 'V1'), 1.0)
                if eff_spd <= 0:
                    eff_spd = 1.0
                dt = (dx / self.px_per_sec) * eff_spd

                orig_in  = self._resize_drag["orig_trim_in"]
                orig_out = self._resize_drag["orig_trim_out"]
                orig_st  = self._resize_drag["orig_start"]

                if self._resize_mode == "trim_in":
                    raw_in  = orig_in + dt
                    new_in  = round(max(0.0, min(raw_in, orig_out - 0.1)) * 10) / 10
                    delta   = (new_in - orig_in) / eff_spd
                    clip['trim_in'] = new_in
                    clip['start']   = round(max(0.0, orig_st + delta) * 10) / 10

                    # Synchronize linked companion clip
                    linked_id = clip.get('linked_clip_id')
                    if linked_id:
                        for comp in self.app.project_clips:
                            if comp['id'] == linked_id:
                                comp['trim_in'] = new_in
                                comp['start']   = clip['start']
                                break
                else:  # trim_out
                    raw_out = orig_out + dt
                    new_out = round(max(orig_in + 0.1, raw_out) * 10) / 10
                    clip['trim_out'] = new_out

                    # Synchronize linked companion clip
                    linked_id = clip.get('linked_clip_id')
                    if linked_id:
                        for comp in self.app.project_clips:
                            if comp['id'] == linked_id:
                                comp['trim_out'] = new_out
                                break
                break

            self._needs_full_redraw = True
            self.draw()
            return

        # ── Move drag ─────────────────────────────────────────────
        if not self._drag["id"]:
            t = self._x_to_time(event.x)
            self.app.scrub_to(t)
            return

        # Magnetic snap
        if self._drag.get("id"):
            snap_times = []
            for clip in self.app.project_clips:
                if clip['id'] != self._drag["id"]:
                    snap_times.append(clip['start'])
                    snap_times.append(clip['start'] + self.get_effective_clip_duration(clip))

            snap_thresh_t = _SNAP_THRESH_PX / self.px_per_sec
            raw = self._drag["orig_start"] + (event.x - self._drag["start_x"]) / self.px_per_sec
            raw = max(0.0, raw)

            snapped_to = None
            for st in snap_times:
                if abs(raw - st) < snap_thresh_t:
                    snapped_to = st
                    break

            snapped = snapped_to if snapped_to is not None else round(raw * 10) / 10
        else:
            dx = event.x - self._drag["start_x"]
            dt = dx / self.px_per_sec
            snapped = round(max(0.0, self._drag["orig_start"] + dt) * 10) / 10

        for clip in self.app.project_clips:
            if clip['id'] == self._drag["id"]:
                delta_t = snapped - clip['start']
                clip['start']       = snapped
                self._drag["moved"] = True

                # Synchronize linked companion clip
                linked_id = clip.get('linked_clip_id')
                if linked_id:
                    for comp in self.app.project_clips:
                        if comp['id'] == linked_id:
                            comp['start'] = max(0.0, comp['start'] + delta_t)
                            break
                break

            self._needs_full_redraw = True
            self.draw()

    def _on_release(self, event):
        # ── Knob release ──────────────────────────────────────────
        if self._knob_drag and self._knob_drag.get('key'):
            moved = self._knob_drag.get('moved', False)
            self._knob_drag = None
            if moved:
                self.app.rebuild_proxy_cache()
            return

        # ── Resize release ────────────────────────────────────────
        if self._resize_drag["active"] and self._resize_drag["id"]:
            clip_id = self._resize_drag["id"]
            for clip in self.app.project_clips:
                if clip['id'] == clip_id:
                    new_in  = clip['trim_in']
                    new_out = clip['trim_out']
                    break
            else:
                new_in = new_out = 0.0

            if _HAS_COMMANDS and hasattr(self.app, 'cmd_stack') and self.app.cmd_stack:
                cmd = TrimClipCommand(
                    self.app, clip_id,
                    self._resize_drag["orig_trim_in"],
                    self._resize_drag["orig_trim_out"],
                    new_in, new_out,
                )
                self.app.cmd_stack.push(cmd)

            self.app.rebuild_proxy_cache()
            self._resize_drag["active"] = False
            self._resize_drag["id"]     = None
            return

        # ── Move release ──────────────────────────────────────────
        if self._drag["id"] and self._drag["moved"]:
            clip_id = self._drag["id"]
            for clip in self.app.project_clips:
                if clip['id'] == clip_id:
                    new_start = clip['start']
                    break
            else:
                new_start = self._drag["orig_start"]

            if _HAS_COMMANDS and hasattr(self.app, 'cmd_stack') and self.app.cmd_stack:
                cmd = MoveClipCommand(
                    self.app, clip_id,
                    self._drag["orig_start"],
                    new_start,
                )
                self.app.cmd_stack.push(cmd)

            self.app.rebuild_proxy_cache()
        self._drag["moved"] = False
        self._drag["id"]    = None

    def _on_right_click(self, event):
        tracks_h = len(self.tracks) * self.track_h
        if event.y >= tracks_h:
            return

        # If right clicked on gutter -> show Track Mixer Menu
        if event.x < _TRACK_LABEL_W:
            trk_idx = event.y // self.track_h
            if 0 <= trk_idx < len(self.track_keys):
                self._show_track_menu(event, self.track_keys[trk_idx])
            return

        # Hit-test clips
        target_clip = None
        for clip in self.app.project_clips:
            try:
                trk_idx = self.track_keys.index(clip['track'])
            except ValueError:
                continue
            dur = self.get_effective_clip_duration(clip)
            x1  = self._time_to_x(clip['start'])
            x2  = self._time_to_x(clip['start'] + dur)
            y1  = trk_idx * self.track_h + 4
            y2  = y1 + self.track_h - 8
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                target_clip = clip
                break

        if not target_clip:
            # Context menu for empty timeline canvas / ruler
            menu = tk.Menu(self, tearoff=0, bg="#2b2b2b", fg="white",
                           activebackground="#3a7ebf", activeforeground="white",
                           bd=0, relief="flat")
            menu.add_command(label="➕ Add Video Track", command=self.add_video_track)
            menu.add_command(label="➕ Add Audio Track", command=self.add_audio_track)
            menu.add_command(label="➕ Add Subtitle Track", command=self.add_subtitle_track)
            menu.add_command(label="➕ Add Image Track", command=self.add_image_track)
            menu.add_separator()
            if hasattr(self.app, "open_audio_rack"):
                menu.add_command(label="🎛 Studio Audio FX Rack & VSTs…",
                                 command=self.app.open_audio_rack)
            if hasattr(self.app, "_ai_subtitle"):
                menu.add_command(label="💬 Auto-Generate Subtitles (Whisper)…",
                                 command=self.app._ai_subtitle)
            if hasattr(self.app, "_new_title_clip"):
                menu.add_command(label="🔤 New Title Clip…",
                                 command=self.app._new_title_clip)
            if hasattr(self.app, "import_media"):
                menu.add_command(label="➕ Import Media…",
                                 command=self.app.import_media)
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
            return

        target_id = target_clip['id']
        self.active_clip_id    = target_id
        self.selected_clip_ids = {target_id}
        self.app.update_inspector()
        self._needs_full_redraw = True
        self.draw()

        menu = tk.Menu(self, tearoff=0, bg="#2b2b2b", fg="white",
                       activebackground="#3a7ebf", activeforeground="white",
                       bd=0, relief="flat")
        menu.add_command(label="✂ Split at Playhead",
                         command=lambda: self.app.split_clip_at_playhead(target_id))
        menu.add_command(label="⧉ Duplicate",
                         command=lambda: self.app.duplicate_clip(target_id))
        if hasattr(self.app, "open_crop_tool") and not target_clip.get('track', '').startswith('A'):
            menu.add_command(label="📐 Crop & Aspect Ratio…",
                             command=self.app.open_crop_tool)

        # Audio FX Rack
        if hasattr(self.app, "open_audio_rack"):
            menu.add_command(label="🎛 Audio FX Rack & VST3…",
                             command=lambda: self.app.open_audio_rack(target_clip=target_clip))

        # Audio separation & Linking options
        if target_clip.get('track', '').startswith('V'):
            menu.add_command(label="🎵 Separate Audio to Track",
                             command=lambda: self.app.separate_audio(target_id))

        if target_clip.get('linked_clip_id'):
            menu.add_command(label="🔗 Unlink Companion Clip",
                             command=lambda: self.app.toggle_clip_link(target_id))
        else:
            other_ids = [cid for cid in self.selected_clip_ids if cid != target_id]
            if other_ids:
                menu.add_command(label="🔗 Link to Selected Clip",
                                 command=lambda: self.app.link_clips(target_id, other_ids[0]))

        # Precision Trimming Tools Submenu
        trim_menu = tk.Menu(menu, tearoff=0, bg="#2b2b2b", fg="white",
                            activebackground="#3a7ebf", activeforeground="white", bd=0, relief="flat")
        fps_val = getattr(self.app, 'fps', 30.0)
        frame_t = 1.0 / fps_val
        trim_menu.add_command(label="Slip Left (1 frame)", command=lambda: self.app.slip_clip(target_id, -frame_t))
        trim_menu.add_command(label="Slip Right (1 frame)", command=lambda: self.app.slip_clip(target_id, frame_t))
        trim_menu.add_command(label="Slip Left (1 second)", command=lambda: self.app.slip_clip(target_id, -1.0))
        trim_menu.add_command(label="Slip Right (1 second)", command=lambda: self.app.slip_clip(target_id, 1.0))
        trim_menu.add_separator()
        trim_menu.add_command(label="Slide Left (1 frame)", command=lambda: self.app.slide_clip(target_id, -frame_t))
        trim_menu.add_command(label="Slide Right (1 frame)", command=lambda: self.app.slide_clip(target_id, frame_t))
        menu.add_cascade(label="↔ Slip & Slide Tools", menu=trim_menu)

        menu.add_separator()

        # ── Label submenu ─────────────────────────────────────────
        label_menu = tk.Menu(menu, tearoff=0, bg="#2b2b2b", fg="white",
                             activebackground="#3a7ebf", activeforeground="white",
                             bd=0, relief="flat")
        label_menu.add_command(
            label="None",
            command=lambda: self._set_label(target_id, None),
        )
        label_menu.add_separator()
        for color_name in ("red", "orange", "yellow", "green", "blue", "purple", "pink"):
            label_menu.add_command(
                label=color_name.capitalize(),
                command=lambda cn=color_name: self._set_label(target_id, cn),
            )
        menu.add_cascade(label="🏷 Label", menu=label_menu)
        menu.add_separator()

        menu.add_command(label="🗑 Delete",
                         command=lambda: self.app.delete_clip(target_id))
        menu.add_command(label="⏩ Ripple Delete (Shift+Del)",
                         command=lambda: self.app.ripple_delete_clip(target_id))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _show_track_menu(self, event, tk_key: str):
        """Display track mixer context menu for adjusting volume, pan, speed, mute, solo, adding, and deleting."""
        menu = tk.Menu(self, tearoff=0, bg="#2b2b2b", fg="white",
                       activebackground="#3a7ebf", activeforeground="white",
                       bd=0, relief="flat")
        curr_vol = self.track_volumes.get(tk_key, 1.0)
        curr_pan = self.track_pans.get(tk_key, 0.0)
        curr_spd = self.track_speeds.get(tk_key, 1.0)
        pan_str = "Center" if abs(curr_pan) < 0.05 else (f"Left {int(abs(curr_pan)*100)}%" if curr_pan < 0 else f"Right {int(curr_pan*100)}%")

        menu.add_command(label=f"Track: {tk_key}", state="disabled")
        menu.add_separator()

        # Add Track Submenu
        add_menu = tk.Menu(menu, tearoff=0, bg="#2b2b2b", fg="white",
                           activebackground="#3a7ebf", activeforeground="white", bd=0, relief="flat")
        add_menu.add_command(label="➕ Add Video Track", command=self.add_video_track)
        add_menu.add_command(label="➕ Add Audio Track", command=self.add_audio_track)
        add_menu.add_command(label="➕ Add Subtitle Track", command=self.add_subtitle_track)
        add_menu.add_command(label="➕ Add Image Track", command=self.add_image_track)
        menu.add_cascade(label="➕ Add Track", menu=add_menu)
        menu.add_separator()

        # Volume submenu
        vol_menu = tk.Menu(menu, tearoff=0, bg="#2b2b2b", fg="white",
                           activebackground="#3a7ebf", activeforeground="white", bd=0, relief="flat")
        for v_pct in (0, 25, 50, 75, 100, 125, 150, 200):
            vol_menu.add_command(
                label=f"{v_pct}%" + ("  ✓" if int(curr_vol * 100) == v_pct else ""),
                command=lambda v=v_pct/100.0: self.set_track_volume(tk_key, v)
            )
        menu.add_cascade(label=f"🔊 Volume ({int(curr_vol*100)}%)", menu=vol_menu)

        # Pan submenu
        pan_menu = tk.Menu(menu, tearoff=0, bg="#2b2b2b", fg="white",
                           activebackground="#3a7ebf", activeforeground="white", bd=0, relief="flat")
        for p_val, p_lbl in [(-1.0, "100% Left"), (-0.5, "50% Left"), (0.0, "Center"), (0.5, "50% Right"), (1.0, "100% Right")]:
            pan_menu.add_command(
                label=p_lbl + ("  ✓" if abs(curr_pan - p_val) < 0.05 else ""),
                command=lambda p=p_val: self.set_track_pan(tk_key, p)
            )
        menu.add_cascade(label=f"🎛 Panning ({pan_str})", menu=pan_menu)

        # Speed submenu
        spd_menu = tk.Menu(menu, tearoff=0, bg="#2b2b2b", fg="white",
                           activebackground="#3a7ebf", activeforeground="white", bd=0, relief="flat")
        for s_val, s_lbl in [
            (0.25, "0.25× (Quarter Speed)"),
            (0.5,  "0.50× (Half Speed)"),
            (0.75, "0.75× (Slow)"),
            (1.0,  "1.00× (Normal Speed)"),
            (1.25, "1.25× (1.25×)"),
            (1.5,  "1.50× (1.5× Fast)"),
            (2.0,  "2.00× (2× Double Speed)"),
            (4.0,  "4.00× (4× Quad Speed)"),
        ]:
            spd_menu.add_command(
                label=s_lbl + ("  ✓" if abs(curr_spd - s_val) < 0.02 else ""),
                command=lambda s=s_val: self.set_track_speed(tk_key, s)
            )
        menu.add_cascade(label=f"⏩ Speed ({curr_spd:.2f}×)", menu=spd_menu)

        menu.add_separator()
        menu.add_command(label="🔇 Toggle Mute" + (" (Muted)" if self.track_muted.get(tk_key) else ""),
                         command=lambda: self.toggle_mute(tk_key))
        menu.add_command(label="⭐ Toggle Solo" + (" (Soloed)" if self.track_solo.get(tk_key) else ""),
                         command=lambda: self.toggle_solo(tk_key))

        menu.add_separator()
        menu.add_command(label="🗑 Delete Track",
                         command=lambda: self.remove_track(tk_key))

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    # ------------------------------------------------------------------
    # Label helper
    # ------------------------------------------------------------------
    def _set_label(self, clip_id, color_name):
        """Set clip label and redraw. Delegates to app.set_clip_label if available."""
        for clip in self.app.project_clips:
            if clip['id'] == clip_id:
                clip['label'] = color_name
                break
        if hasattr(self.app, 'set_clip_label'):
            self.app.set_clip_label(clip_id, color_name)
        self._needs_full_redraw = True
        self.draw()

