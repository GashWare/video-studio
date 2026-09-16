import customtkinter as ctk
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk


class MaskPainter(ctk.CTkToplevel):
    """
    Keyframe Mask Editor — draw / erase the alpha mask for one frame.

    Controls
    --------
    Left-drag    : paint (add to mask)
    Right-drag   : erase (remove from mask)
    Ctrl+Z       : undo last stroke
    F            : flood-fill at cursor (add)
    Shift+F      : flood-fill at cursor (erase)
    Mouse-hover  : shows brush-size preview circle on canvas
    """

    _MAX_UNDO = 20     # undo history depth

    def __init__(self, app, frame_idx, img, mask, clip_ref):
        super().__init__(app)
        self.title(f"Mask Touchup — Frame {frame_idx}")
        self.resizable(True, True)

        self.app       = app
        self.idx       = frame_idx
        self.clip_ref  = clip_ref
        self.img       = img.copy()

        # Normalise mask to uint8 2-D
        m = mask.copy() if mask is not None else np.zeros(img.shape[:2], dtype=np.uint8)
        if m.ndim > 2:
            m = m[:, :, 0]
        self.mask = m

        # Undo stack (list of mask snapshots)
        self._undo_stack = []

        # Cursor overlay item id
        self._cursor_oval = None

        self._build_ui()
        self._redraw()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── Toolbar ──────────────────────────────────────────────────
        tb = ctk.CTkFrame(self, fg_color="#1e1e1e", height=44)
        tb.pack(fill="x", padx=6, pady=(6, 2))

        ctk.CTkLabel(tb, text="Brush:").pack(side="left", padx=(8, 4))
        self.b_size = tk.IntVar(value=30)
        self._size_lbl = ctk.CTkLabel(tb, text="30 px", width=42)
        self._size_lbl.pack(side="left")
        ctk.CTkSlider(tb, variable=self.b_size, from_=3, to=160, width=140,
                       command=self._on_size_change).pack(side="left", padx=(2, 12))

        ctk.CTkLabel(tb, text="Opacity:").pack(side="left", padx=(0, 4))
        self.b_opacity = tk.IntVar(value=255)
        ctk.CTkSlider(tb, variable=self.b_opacity, from_=1, to=255, width=100
                       ).pack(side="left", padx=(0, 12))

        ctk.CTkButton(tb, text="⟲ Undo", width=72, height=28,
                       command=self._undo).pack(side="left", padx=4)
        ctk.CTkButton(tb, text="Fill ✦", width=72, height=28,
                       fg_color="#2d6fbf",
                       command=lambda: self._flood_fill(True)).pack(side="left", padx=2)
        ctk.CTkButton(tb, text="Erase ✦", width=72, height=28,
                       fg_color="#555555",
                       command=lambda: self._flood_fill(False)).pack(side="left", padx=2)
        ctk.CTkButton(tb, text="Clear All", width=76, height=28,
                       fg_color="#7a2020",
                       command=self._clear_mask).pack(side="left", padx=2)
        ctk.CTkButton(tb, text="Fill All", width=76, height=28,
                       fg_color="#1e6e1e",
                       command=self._fill_mask).pack(side="left", padx=2)

        ctk.CTkButton(tb, text="💾 Save Override", width=120, height=28,
                       fg_color="teal", hover_color="#006666",
                       command=self._save).pack(side="right", padx=8)

        # Help text
        ctk.CTkLabel(
            self,
            text="Left-drag: paint  |  Right-drag: erase  |  Ctrl+Z: undo  |  F / Shift+F: flood fill",
            text_color="#666666", font=("Segoe UI", 8)
        ).pack(fill="x", padx=10)

        # ── Canvas in scrollable area ─────────────────────────────
        self._scroll_f = ctk.CTkScrollableFrame(self, width=860, height=620)
        self._scroll_f.pack(fill="both", expand=True, padx=6, pady=6)

        h, w = self.img.shape[:2]
        self.canvas = tk.Canvas(self._scroll_f, width=w, height=h,
                                 cursor="none", bg="#3a3a3a",
                                 highlightthickness=0)
        self.canvas.pack()

        # Canvas bindings
        self.canvas.bind("<B1-Motion>",     lambda e: self._stroke(e, True))
        self.canvas.bind("<Button-1>",      lambda e: self._stroke(e, True))
        self.canvas.bind("<B3-Motion>",     lambda e: self._stroke(e, False))
        self.canvas.bind("<Button-3>",      lambda e: self._stroke(e, False))
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._end_stroke())
        self.canvas.bind("<ButtonRelease-3>", lambda e: self._end_stroke())
        self.canvas.bind("<Motion>",        self._on_mouse_move)
        self.canvas.bind("<Leave>",         self._on_mouse_leave)

        # Keyboard
        self.bind("<Control-z>", lambda e: self._undo())
        self.bind("f",           lambda e: self._flood_fill(True))
        self.bind("F",           lambda e: self._flood_fill(False))

        self._stroke_active = False
        self._last_cursor_pos = (0, 0)

    # ------------------------------------------------------------------
    # Brush cursor
    # ------------------------------------------------------------------
    def _on_size_change(self, val):
        self._size_lbl.configure(text=f"{int(float(val))} px")
        self._update_cursor_oval(self._last_cursor_pos)

    def _on_mouse_move(self, event):
        self._last_cursor_pos = (event.x, event.y)
        self._update_cursor_oval((event.x, event.y))

    def _on_mouse_leave(self, event):
        if self._cursor_oval:
            self.canvas.delete(self._cursor_oval)
            self._cursor_oval = None

    def _update_cursor_oval(self, pos):
        r = self.b_size.get() // 2
        x, y = pos
        if self._cursor_oval:
            self.canvas.delete(self._cursor_oval)
        self._cursor_oval = self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            outline="#ffffff", width=1, dash=(3, 3)
        )

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def _stroke(self, event, add: bool):
        if not self._stroke_active:
            # Save undo snapshot at start of each stroke
            self._push_undo()
            self._stroke_active = True
        val = min(255, self.b_opacity.get()) if add else 0
        cv2.circle(self.mask, (event.x, event.y), self.b_size.get(), val, -1)
        self._redraw()
        self._update_cursor_oval((event.x, event.y))

    def _end_stroke(self):
        self._stroke_active = False

    def _flood_fill(self, add: bool, pos=None):
        if pos is None:
            pos = self._last_cursor_pos
        x, y = int(pos[0]), int(pos[1])
        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        self._push_undo()
        fill_val = 255 if add else 0
        # Use OpenCV floodFill on a copy and detect changed pixels
        tmp   = self.mask.copy()
        flood = tmp.copy()
        cv2.floodFill(flood, None, (x, y), fill_val, loDiff=30, upDiff=30)
        self.mask = flood
        self._redraw()

    def _clear_mask(self):
        self._push_undo()
        self.mask[:] = 0
        self._redraw()

    def _fill_mask(self):
        self._push_undo()
        self.mask[:] = 255
        self._redraw()

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------
    def _push_undo(self):
        self._undo_stack.append(self.mask.copy())
        if len(self._undo_stack) > self._MAX_UNDO:
            self._undo_stack.pop(0)

    def _undo(self):
        if self._undo_stack:
            self.mask = self._undo_stack.pop()
            self._redraw()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _redraw(self):
        # Blend: original image (60%) + green-tinted mask (40%)
        coloured = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB)
        # Tint white mask areas green
        tint = np.zeros_like(coloured)
        tint[:, :, 1] = self.mask            # green channel = mask value
        overlay = cv2.addWeighted(self.img, 0.6, tint, 0.4, 0)

        # Draw a subtle red X over fully-transparent (0) regions
        alpha_vis = cv2.cvtColor((255 - self.mask), cv2.COLOR_GRAY2RGBA)
        alpha_vis[:, :, 3] = (255 - self.mask) // 3

        self._tk_img = ImageTk.PhotoImage(image=Image.fromarray(overlay))
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        # Re-draw cursor on top
        if self._cursor_oval:
            self.canvas.delete(self._cursor_oval)
            self._cursor_oval = None
        self._update_cursor_oval(self._last_cursor_pos)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _save(self):
        if 'custom_masks' not in self.clip_ref:
            self.clip_ref['custom_masks'] = {}
        self.clip_ref['custom_masks'][self.idx] = self.mask.copy()
        self.destroy()
