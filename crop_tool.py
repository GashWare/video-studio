# crop_tool.py
"""
Interactive Visual Crop & Aspect Ratio Dialog for Video Studio.

Allows users to visually drag and resize a crop box over a video frame,
choose standard aspect ratio presets (16:9, 9:16, 1:1, 4:3, 21:9, 4:5, 2:3, Freeform),
center/maximize crop, and apply it directly to the clip with undo support and live preview.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

log = logging.getLogger(__name__)

ASPECT_PRESETS: Dict[str, Optional[float]] = {
    "Freeform": None,
    "Original": -1.0,      # Special sentinel for original frame ratio
    "16:9": 16.0 / 9.0,    # YouTube / Landscape
    "9:16": 9.0 / 16.0,    # TikTok / Shorts / Reels
    "1:1": 1.0,            # Instagram Square
    "4:5": 4.0 / 5.0,      # Instagram Portrait
    "4:3": 4.0 / 3.0,      # Classic TV
    "21:9": 21.0 / 9.0,    # Ultrawide / Cinema
    "2:3": 2.0 / 3.0,      # Classic Photo
}


class CropAspectDialog(ctk.CTkToplevel):
    """
    Interactive visual modal for setting clip aspect ratio and crop boundaries.
    """

    HANDLE_SIZE = 12       # Screen pixels for corner/edge handle detection
    MIN_BOX_PX = 24        # Minimum box width/height in canvas pixels

    def __init__(
        self,
        app: Any,
        clip_data: dict,
        frame_rgb: np.ndarray,
        on_apply: Optional[Callable[[Optional[dict], str], None]] = None,
    ) -> None:
        super().__init__(app)
        self.app = app
        self.clip = clip_data
        self.on_apply = on_apply
        self.frame_rgb = frame_rgb  # Shape (H, W, 3), RGB uint8

        self.img_h, self.img_w = frame_rgb.shape[:2]
        self.orig_aspect = self.img_w / max(1, self.img_h)

        clip_label = clip_data.get("label") or clip_data.get("path", "Clip")
        if isinstance(clip_label, str) and ("/" in clip_label or "\\" in clip_label):
            import os
            clip_label = os.path.basename(clip_label)
        self.title(f"📐 Crop & Aspect Ratio — {clip_label}")
        self.geometry("940x720")
        self.minsize(760, 560)
        self.configure(fg_color="#12151c")

        # Active preset name
        saved_aspect = clip_data.get("aspect", "Original")
        if saved_aspect not in ASPECT_PRESETS:
            saved_aspect = "Original"
        self.current_preset = saved_aspect

        # Normalized crop coordinates [0.0 - 1.0]: (x1, y1, x2, y2)
        saved_crop = clip_data.get("crop")
        if saved_crop and isinstance(saved_crop, dict):
            self.x1 = float(saved_crop.get("x1", 0.0))
            self.y1 = float(saved_crop.get("y1", 0.0))
            self.x2 = float(saved_crop.get("x2", 1.0))
            self.y2 = float(saved_crop.get("y2", 1.0))
            # If saved crop exists but aspect was 'Original', use Freeform
            if saved_aspect == "Original":
                self.current_preset = "Freeform"
        elif saved_aspect != "Original" and saved_aspect in ASPECT_PRESETS and ASPECT_PRESETS[saved_aspect]:
            # Calculate initial centered crop for the preset
            target_r = ASPECT_PRESETS[saved_aspect]
            self.x1, self.y1, self.x2, self.y2 = self._calc_centered_crop(target_r)
        else:
            self.x1, self.y1, self.x2, self.y2 = 0.0, 0.0, 1.0, 1.0

        # State for canvas transform and drag interactions
        self.disp_w = 1
        self.disp_h = 1
        self.disp_x0 = 0
        self.disp_y0 = 0
        self.disp_scale = 1.0
        self.tk_img: Optional[ImageTk.PhotoImage] = None

        self._active_handle: Optional[str] = None
        self._drag_start_mouse: Tuple[int, int] = (0, 0)
        self._drag_start_box: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

        self._build_ui()
        self.after(50, self._initial_render)

        self.transient(app)
        self.grab_set()
        self.focus_force()

    # ------------------------------------------------------------------
    # Geometry Helpers
    # ------------------------------------------------------------------

    def _target_aspect_ratio(self) -> Optional[float]:
        """Return the target numeric width/height ratio for the current preset, or None if freeform."""
        if self.current_preset == "Freeform":
            return None
        if self.current_preset == "Original":
            return self.orig_aspect
        return ASPECT_PRESETS.get(self.current_preset)

    def _calc_centered_crop(self, target_r: float) -> Tuple[float, float, float, float]:
        """Calculate max-sized centered normalized crop rectangle matching target_r."""
        orig_r = self.orig_aspect
        if orig_r > target_r:
            # Source is wider than target -> crop horizontally
            crop_w = target_r / orig_r
            x1 = (1.0 - crop_w) / 2.0
            x2 = x1 + crop_w
            return (max(0.0, x1), 0.0, min(1.0, x2), 1.0)
        else:
            # Source is taller than target -> crop vertically
            crop_h = orig_r / target_r
            y1 = (1.0 - crop_h) / 2.0
            y2 = y1 + crop_h
            return (0.0, max(0.0, y1), 1.0, min(1.0, y2))

    # ------------------------------------------------------------------
    # UI Building
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Top Header Bar
        hdr = ctk.CTkFrame(self, fg_color="#182030", height=48, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(
            hdr,
            text="📐  Interactive Crop & Aspect Ratio",
            font=("Segoe UI", 13, "bold"),
            text_color="#f8fafc",
        ).pack(side="left", padx=16)

        self.lbl_dims = ctk.CTkLabel(
            hdr,
            text=f"Source: {self.img_w}×{self.img_h}  |  Crop: {int((self.x2 - self.x1)*self.img_w)}×{int((self.y2 - self.y1)*self.img_h)}",
            font=("Segoe UI", 11),
            text_color="#94a3b8",
        )
        self.lbl_dims.pack(side="right", padx=16)

        # Main Body (Presets Toolbar + Canvas)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # Presets Bar (Horizontal top strip)
        preset_bar = ctk.CTkFrame(body, fg_color="#1a1e28", height=42, corner_radius=8)
        preset_bar.pack(fill="x", pady=(0, 8))
        preset_bar.pack_propagate(False)

        ctk.CTkLabel(
            preset_bar,
            text="Aspect Presets:",
            font=("Segoe UI", 11, "bold"),
            text_color="#cbd5e1",
        ).pack(side="left", padx=(12, 8))

        self.preset_btns = {}
        preset_labels = [
            ("Original", "Original"),
            ("16:9", "16:9 (Landscape)"),
            ("9:16", "9:16 (Shorts/TikTok)"),
            ("1:1", "1:1 (Square)"),
            ("4:5", "4:5 (Portrait)"),
            ("4:3", "4:3 (SD)"),
            ("21:9", "21:9 (Cinema)"),
            ("Freeform", "Freeform"),
        ]

        for p_key, p_label in preset_labels:
            btn = ctk.CTkButton(
                preset_bar,
                text=p_key,
                width=64,
                height=28,
                corner_radius=6,
                font=("Segoe UI", 10, "bold"),
                fg_color="#2563eb" if p_key == self.current_preset else "#283042",
                hover_color="#1d4ed8" if p_key == self.current_preset else "#374151",
                text_color="#ffffff",
                command=lambda k=p_key: self._on_preset_click(k),
            )
            btn.pack(side="left", padx=3, pady=6)
            self.preset_btns[p_key] = btn

        # Quick Action Buttons on right of preset bar
        ctk.CTkButton(
            preset_bar,
            text="Center",
            width=58,
            height=28,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            font=("Segoe UI", 10),
            command=self._on_center_click,
        ).pack(side="right", padx=(2, 10), pady=6)

        ctk.CTkButton(
            preset_bar,
            text="Maximize",
            width=68,
            height=28,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            font=("Segoe UI", 10),
            command=self._on_maximize_click,
        ).pack(side="right", padx=2, pady=6)

        ctk.CTkButton(
            preset_bar,
            text="Full Frame",
            width=76,
            height=28,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            font=("Segoe UI", 10),
            command=self._on_full_frame_click,
        ).pack(side="right", padx=2, pady=6)

        # Center Canvas Container
        canvas_container = ctk.CTkFrame(body, fg_color="#0b0e14", corner_radius=8)
        canvas_container.pack(fill="both", expand=True)

        self.canvas = ctk.CTkCanvas(
            canvas_container,
            bg="#0b0e14",
            highlightthickness=0,
            bd=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Motion>", self._on_mouse_hover)

        # Bottom Control & Action Bar
        bottom = ctk.CTkFrame(self, fg_color="#182030", height=54, corner_radius=0)
        bottom.pack(fill="x")
        bottom.pack_propagate(False)

        self.lbl_help = ctk.CTkLabel(
            bottom,
            text="💡 Drag the corners or edges to crop. Drag inside the box to reposition. Rule of thirds grid assists framing.",
            font=("Segoe UI", 10),
            text_color="#94a3b8",
        )
        self.lbl_help.pack(side="left", padx=16)

        # Action buttons
        ctk.CTkButton(
            bottom,
            text="Cancel",
            width=90,
            height=34,
            corner_radius=6,
            fg_color="#334155",
            hover_color="#475569",
            font=("Segoe UI", 11),
            command=self.destroy,
        ).pack(side="right", padx=(4, 16))

        ctk.CTkButton(
            bottom,
            text="Reset Crop",
            width=100,
            height=34,
            corner_radius=6,
            fg_color="#854d0e",
            hover_color="#a16207",
            font=("Segoe UI", 11, "bold"),
            command=self._on_reset_and_apply,
        ).pack(side="right", padx=4)

        ctk.CTkButton(
            bottom,
            text="✔  Apply Crop",
            width=130,
            height=34,
            corner_radius=6,
            fg_color="#16a34a",
            hover_color="#15803d",
            font=("Segoe UI", 11, "bold"),
            command=self._on_apply_click,
        ).pack(side="right", padx=4)

    # ------------------------------------------------------------------
    # Canvas Rendering
    # ------------------------------------------------------------------

    def _initial_render(self) -> None:
        self._update_display_transform()
        self._draw_all()

    def _on_canvas_resize(self, event) -> None:
        self._update_display_transform()
        self._draw_all()

    def _update_display_transform(self) -> None:
        """Compute the scale and offset to fit self.frame_rgb on self.canvas."""
        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())

        scale = min(cw / self.img_w, ch / self.img_h)
        self.disp_scale = scale
        self.disp_w = int(round(self.img_w * scale))
        self.disp_h = int(round(self.img_h * scale))
        self.disp_x0 = (cw - self.disp_w) // 2
        self.disp_y0 = (ch - self.disp_h) // 2

        # Create resized display image
        if self.disp_w > 0 and self.disp_h > 0:
            resized = cv2.resize(self.frame_rgb, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
            pil_img = Image.fromarray(resized)
            self.tk_img = ImageTk.PhotoImage(pil_img)

    def _norm_to_canvas(self, nx: float, ny: float) -> Tuple[int, int]:
        cx = int(round(self.disp_x0 + nx * self.disp_w))
        cy = int(round(self.disp_y0 + ny * self.disp_h))
        return (cx, cy)

    def _canvas_to_norm(self, cx: int, cy: int) -> Tuple[float, float]:
        if self.disp_w <= 0 or self.disp_h <= 0:
            return (0.0, 0.0)
        nx = (cx - self.disp_x0) / float(self.disp_w)
        ny = (cy - self.disp_y0) / float(self.disp_h)
        return (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))

    def _draw_all(self) -> None:
        self.canvas.delete("all")
        if self.tk_img is not None:
            self.canvas.create_image(self.disp_x0, self.disp_y0, anchor="nw", image=self.tk_img)

        # Crop box coordinates on canvas
        bx1, by1 = self._norm_to_canvas(self.x1, self.y1)
        bx2, by2 = self._norm_to_canvas(self.x2, self.y2)

        # 1. Darken areas outside the crop box
        # Top outer
        if by1 > self.disp_y0:
            self.canvas.create_rectangle(
                self.disp_x0, self.disp_y0, self.disp_x0 + self.disp_w, by1,
                fill="#000000", stipple="gray50", outline=""
            )
        # Bottom outer
        if by2 < self.disp_y0 + self.disp_h:
            self.canvas.create_rectangle(
                self.disp_x0, by2, self.disp_x0 + self.disp_w, self.disp_y0 + self.disp_h,
                fill="#000000", stipple="gray50", outline=""
            )
        # Left outer
        if bx1 > self.disp_x0:
            self.canvas.create_rectangle(
                self.disp_x0, by1, bx1, by2,
                fill="#000000", stipple="gray50", outline=""
            )
        # Right outer
        if bx2 < self.disp_x0 + self.disp_w:
            self.canvas.create_rectangle(
                bx2, by1, self.disp_x0 + self.disp_w, by2,
                fill="#000000", stipple="gray50", outline=""
            )

        # 2. Rule of thirds grid inside crop box
        w_box = bx2 - bx1
        h_box = by2 - by1
        if w_box > 40 and h_box > 40:
            gx1 = bx1 + w_box / 3.0
            gx2 = bx1 + (2.0 * w_box) / 3.0
            gy1 = by1 + h_box / 3.0
            gy2 = by1 + (2.0 * h_box) / 3.0

            self.canvas.create_line(gx1, by1, gx1, by2, fill="#ffffff", dash=(2, 4), width=1)
            self.canvas.create_line(gx2, by1, gx2, by2, fill="#ffffff", dash=(2, 4), width=1)
            self.canvas.create_line(bx1, gy1, bx2, gy1, fill="#ffffff", dash=(2, 4), width=1)
            self.canvas.create_line(bx1, gy2, bx2, gy2, fill="#ffffff", dash=(2, 4), width=1)

        # 3. Main Crop Box Outline
        self.canvas.create_rectangle(
            bx1, by1, bx2, by2,
            outline="#38bdf8", width=2
        )

        # 4. Handles (4 Corners + 4 Edges)
        hs = self.HANDLE_SIZE
        half = hs // 2

        handles = {
            "nw": (bx1, by1),
            "ne": (bx2, by1),
            "sw": (bx1, by2),
            "se": (bx2, by2),
            "n": (bx1 + w_box // 2, by1),
            "s": (bx1 + w_box // 2, by2),
            "w": (bx1, by1 + h_box // 2),
            "e": (bx2, by1 + h_box // 2),
        }

        for name, (hx, hy) in handles.items():
            self.canvas.create_rectangle(
                hx - half, hy - half, hx + half, hy + half,
                fill="#38bdf8", outline="#0f172a", width=1.5
            )

        # Update dimensions label
        actual_cw = int(round((self.x2 - self.x1) * self.img_w))
        actual_ch = int(round((self.y2 - self.y1) * self.img_h))
        cur_ratio = actual_cw / max(1, actual_ch)
        self.lbl_dims.configure(
            text=f"Preset: {self.current_preset}  |  Crop: {actual_cw}×{actual_ch} ({cur_ratio:.2f}:1)  |  Source: {self.img_w}×{self.img_h}"
        )

    # ------------------------------------------------------------------
    # Mouse Interaction (Drag, Resize, Move)
    # ------------------------------------------------------------------

    def _hit_test(self, mx: int, my: int) -> Optional[str]:
        """Detect if mouse (mx, my) is on a handle or inside the crop box."""
        bx1, by1 = self._norm_to_canvas(self.x1, self.y1)
        bx2, by2 = self._norm_to_canvas(self.x2, self.y2)
        hs = self.HANDLE_SIZE + 4  # hit slop

        # Corners
        if abs(mx - bx1) <= hs and abs(my - by1) <= hs:
            return "nw"
        if abs(mx - bx2) <= hs and abs(my - by1) <= hs:
            return "ne"
        if abs(mx - bx1) <= hs and abs(my - by2) <= hs:
            return "sw"
        if abs(mx - bx2) <= hs and abs(my - by2) <= hs:
            return "se"

        # Edges
        mid_x = (bx1 + bx2) // 2
        mid_y = (by1 + by2) // 2
        if abs(mx - mid_x) <= hs and abs(my - by1) <= hs:
            return "n"
        if abs(mx - mid_x) <= hs and abs(my - by2) <= hs:
            return "s"
        if abs(mx - bx1) <= hs and abs(my - mid_y) <= hs:
            return "w"
        if abs(mx - bx2) <= hs and abs(my - mid_y) <= hs:
            return "e"

        # Inside body
        if bx1 <= mx <= bx2 and by1 <= my <= by2:
            return "move"

        return None

    def _on_mouse_hover(self, event) -> None:
        if self._active_handle is not None:
            return
        hit = self._hit_test(event.x, event.y)
        cursor_map = {
            "nw": "size_nw_se",
            "se": "size_nw_se",
            "ne": "size_ne_sw",
            "sw": "size_ne_sw",
            "n": "size_ns",
            "s": "size_ns",
            "w": "size_we",
            "e": "size_we",
            "move": "fleur",
        }
        cursor = cursor_map.get(hit, "crosshair")
        try:
            self.canvas.configure(cursor=cursor)
        except Exception:
            pass

    def _on_mouse_down(self, event) -> None:
        hit = self._hit_test(event.x, event.y)
        if hit is None:
            return
        self._active_handle = hit
        self._drag_start_mouse = (event.x, event.y)
        self._drag_start_box = (self.x1, self.y1, self.x2, self.y2)

    def _on_mouse_drag(self, event) -> None:
        if self._active_handle is None or self.disp_w <= 0 or self.disp_h <= 0:
            return

        dx_norm = (event.x - self._drag_start_mouse[0]) / float(self.disp_w)
        dy_norm = (event.y - self._drag_start_mouse[1]) / float(self.disp_h)

        ox1, oy1, ox2, oy2 = self._drag_start_box
        handle = self._active_handle
        target_r = self._target_aspect_ratio()

        min_w_norm = self.MIN_BOX_PX / float(self.disp_w)
        min_h_norm = self.MIN_BOX_PX / float(self.disp_h)

        if handle == "move":
            # Translate the box, clamping within [0, 1]
            bw = ox2 - ox1
            bh = oy2 - oy1
            nx1 = max(0.0, min(1.0 - bw, ox1 + dx_norm))
            ny1 = max(0.0, min(1.0 - bh, oy1 + dy_norm))
            self.x1 = nx1
            self.y1 = ny1
            self.x2 = nx1 + bw
            self.y2 = ny1 + bh
            self._draw_all()
            return

        # Resizing
        nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2

        if target_r is None:
            # Freeform resizing
            if "w" in handle:
                nx1 = min(ox2 - min_w_norm, max(0.0, ox1 + dx_norm))
            if "e" in handle:
                nx2 = max(ox1 + min_w_norm, min(1.0, ox2 + dx_norm))
            if "n" in handle:
                ny1 = min(oy2 - min_h_norm, max(0.0, oy1 + dy_norm))
            if "s" in handle:
                ny2 = max(oy1 + min_h_norm, min(1.0, oy2 + dy_norm))
        else:
            # Aspect-constrained resizing: (W_actual / H_actual) = target_r
            # W_norm * img_w / (H_norm * img_h) = target_r
            # => W_norm = H_norm * target_r * (img_h / img_w)
            norm_ratio = target_r * (self.img_h / float(self.img_w))

            if handle in ("se", "e", "s"):
                # Anchor at (ox1, oy1)
                new_w = max(min_w_norm, min(1.0 - ox1, (ox2 - ox1) + dx_norm))
                new_h = new_w / norm_ratio
                if oy1 + new_h > 1.0:
                    new_h = 1.0 - oy1
                    new_w = new_h * norm_ratio
                nx2 = ox1 + new_w
                ny2 = oy1 + new_h

            elif handle in ("nw", "w", "n"):
                # Anchor at (ox2, oy2)
                new_w = max(min_w_norm, min(ox2, (ox2 - ox1) - dx_norm))
                new_h = new_w / norm_ratio
                if oy2 - new_h < 0.0:
                    new_h = oy2
                    new_w = new_h * norm_ratio
                nx1 = ox2 - new_w
                ny1 = oy2 - new_h

            elif handle == "ne":
                # Anchor at (ox1, oy2)
                new_w = max(min_w_norm, min(1.0 - ox1, (ox2 - ox1) + dx_norm))
                new_h = new_w / norm_ratio
                if oy2 - new_h < 0.0:
                    new_h = oy2
                    new_w = new_h * norm_ratio
                nx2 = ox1 + new_w
                ny1 = oy2 - new_h

            elif handle == "sw":
                # Anchor at (ox2, oy1)
                new_w = max(min_w_norm, min(ox2, (ox2 - ox1) - dx_norm))
                new_h = new_w / norm_ratio
                if oy1 + new_h > 1.0:
                    new_h = 1.0 - oy1
                    new_w = new_h * norm_ratio
                nx1 = ox2 - new_w
                ny2 = oy1 + new_h

        self.x1 = max(0.0, min(1.0, nx1))
        self.y1 = max(0.0, min(1.0, ny1))
        self.x2 = max(self.x1 + 0.01, min(1.0, nx2))
        self.y2 = max(self.y1 + 0.01, min(1.0, ny2))

        self._draw_all()

    def _on_mouse_up(self, event) -> None:
        self._active_handle = None

    # ------------------------------------------------------------------
    # Presets & Toolbar Actions
    # ------------------------------------------------------------------

    def _on_preset_click(self, preset_key: str) -> None:
        self.current_preset = preset_key

        # Update button highlights
        for key, btn in self.preset_btns.items():
            if key == preset_key:
                btn.configure(fg_color="#2563eb", hover_color="#1d4ed8")
            else:
                btn.configure(fg_color="#283042", hover_color="#374151")

        target_r = self._target_aspect_ratio()
        if target_r is not None:
            # Recompute max centered box matching preset
            self.x1, self.y1, self.x2, self.y2 = self._calc_centered_crop(target_r)
        self._draw_all()

    def _on_center_click(self) -> None:
        """Center current crop box within frame."""
        bw = self.x2 - self.x1
        bh = self.y2 - self.y1
        self.x1 = max(0.0, (1.0 - bw) / 2.0)
        self.y1 = max(0.0, (1.0 - bh) / 2.0)
        self.x2 = self.x1 + bw
        self.y2 = self.y1 + bh
        self._draw_all()

    def _on_maximize_click(self) -> None:
        """Expand crop box to maximal bounds while preserving current ratio."""
        target_r = self._target_aspect_ratio()
        if target_r is None:
            cur_w = (self.x2 - self.x1) * self.img_w
            cur_h = (self.y2 - self.y1) * self.img_h
            target_r = cur_w / max(1.0, cur_h)

        self.x1, self.y1, self.x2, self.y2 = self._calc_centered_crop(target_r)
        self._draw_all()

    def _on_full_frame_click(self) -> None:
        """Reset to full frame."""
        self.x1, self.y1, self.x2, self.y2 = 0.0, 0.0, 1.0, 1.0
        self.current_preset = "Original"
        for key, btn in self.preset_btns.items():
            if key == "Original":
                btn.configure(fg_color="#2563eb", hover_color="#1d4ed8")
            else:
                btn.configure(fg_color="#283042", hover_color="#374151")
        self._draw_all()

    def _on_reset_and_apply(self) -> None:
        """Clear crop entirely and apply."""
        if self.on_apply:
            self.on_apply(None, "Original")
        self.destroy()

    def _on_apply_click(self) -> None:
        """Apply active crop box and aspect ratio."""
        # Check if full frame (within tiny epsilon)
        is_full = (
            abs(self.x1) < 0.002
            and abs(self.y1) < 0.002
            and abs(self.x2 - 1.0) < 0.002
            and abs(self.y2 - 1.0) < 0.002
        )

        crop_dict = None if (is_full and self.current_preset in ("Original", "Freeform")) else {
            "x1": round(self.x1, 5),
            "y1": round(self.y1, 5),
            "x2": round(self.x2, 5),
            "y2": round(self.y2, 5),
        }

        aspect_str = self.current_preset
        if self.on_apply:
            self.on_apply(crop_dict, aspect_str)
        self.destroy()
