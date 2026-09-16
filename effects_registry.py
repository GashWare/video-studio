# effects_registry.py
"""
Plugin-style video effects registry.

Effects are registered with a name, parameter schema, and an apply() function.
engine.py calls apply_registered_effects(clip, clip_data) to chain all enabled effects.
The Inspector auto-builds a UI panel for each registered effect from its param_schema.
New effects can be added by dropping a .py file into the /effects/ folder.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
import importlib.util
import os
import sys
import numpy as np
import cv2


@dataclass
class EffectParam:
    """Describes one parameter of an effect for UI auto-generation."""
    name:         str
    label:        str
    type:         str          # "float" | "int" | "bool" | "choice"
    default:      Any
    min_val:      float = 0.0
    max_val:      float = 1.0
    choices:      list  = field(default_factory=list)  # for type="choice"
    step:         float = 0.01


@dataclass
class EffectDefinition:
    """A registered video effect."""
    name:         str          # unique key stored in clip dict
    label:        str          # human-readable name shown in Inspector
    description:  str
    params:       list[EffectParam]
    apply_frame:  Callable     # fn(frame: np.ndarray, params: dict) -> np.ndarray
    category:     str = "General"  # "General" | "Colour" | "Blur" | "Stylize" | "Distort"


class EffectsRegistry:
    def __init__(self):
        self._effects: dict[str, EffectDefinition] = {}

    def register(self, effect: EffectDefinition):
        """Register an effect. Raises ValueError if name already taken."""
        if effect.name in self._effects:
            raise ValueError(f"Effect '{effect.name}' is already registered.")
        self._effects[effect.name] = effect

    def get(self, name: str) -> EffectDefinition | None:
        """Return effect by name, or None."""
        return self._effects.get(name)

    def all_effects(self) -> list[EffectDefinition]:
        """Return all registered effects sorted by category then label."""
        return sorted(self._effects.values(), key=lambda e: (e.category, e.label))

    def by_category(self) -> dict[str, list[EffectDefinition]]:
        """Return effects grouped by category."""
        result: dict[str, list[EffectDefinition]] = {}
        for effect in self.all_effects():
            result.setdefault(effect.category, []).append(effect)
        return result

    def apply_to_frame(self, frame: np.ndarray, effect_name: str, params: dict) -> np.ndarray:
        """Apply one named effect to a frame. Returns frame unchanged if effect not found."""
        effect = self._effects.get(effect_name)
        if effect is None:
            return frame
        return effect.apply_frame(frame, params)

    def apply_all_enabled(self, frame: np.ndarray, clip_effects: dict) -> np.ndarray:
        """
        clip_effects: dict of {effect_name: {enabled: bool, param1: val, ...}}
        Apply all enabled effects in registration order.
        """
        for effect_name, effect_data in clip_effects.items():
            if not effect_data.get("enabled", False):
                continue
            effect = self._effects.get(effect_name)
            if effect is None:
                continue
            params = {k: v for k, v in effect_data.items() if k != "enabled"}
            frame = effect.apply_frame(frame, params)
        return frame

    def load_plugin_folder(self, folder: str):
        """
        Scan folder/*.py files, import each as a module, call module.register(registry)
        if it has that function. Silently skip files that fail to import.
        """
        if not os.path.isdir(folder):
            return
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(folder, filename)
            module_name = f"_effect_plugin_{filename[:-3]}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                if hasattr(module, "register") and callable(module.register):
                    module.register(self)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------
registry = EffectsRegistry()


# ---------------------------------------------------------------------------
# Helper: ensure frame is uint8 BGR
# ---------------------------------------------------------------------------
def _to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1. Gaussian Blur
# ---------------------------------------------------------------------------
def _apply_blur_gaussian(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    # kernel size must be odd and >= 1
    ksize = max(1, int(strength) * 2 + 1)
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


registry.register(EffectDefinition(
    name="blur_gaussian",
    label="Gaussian Blur",
    description="Softens the image using a Gaussian kernel.",
    category="Blur",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=30.0, step=0.5),
    ],
    apply_frame=_apply_blur_gaussian,
))


# ---------------------------------------------------------------------------
# 2. Sharpen (unsharp mask)
# ---------------------------------------------------------------------------
def _apply_sharpen(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(frame, 1.0 + strength, blurred, -strength, 0)
    return _to_uint8(sharpened)


registry.register(EffectDefinition(
    name="sharpen",
    label="Sharpen",
    description="Sharpens the image using an unsharp mask.",
    category="Blur",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=3.0, step=0.05),
    ],
    apply_frame=_apply_sharpen,
))


# ---------------------------------------------------------------------------
# 3. Vignette
# ---------------------------------------------------------------------------
def _apply_vignette(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    radius   = float(params.get("radius", 0.7))
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    # Normalise distance; vignette starts at `radius` fraction of max_dist
    norm = np.clip((dist / (max_dist * radius) - 1.0) / (1.0 / radius), 0, 1)
    mask = 1.0 - strength * norm
    mask = mask[:, :, np.newaxis]
    return _to_uint8(frame.astype(np.float32) * mask)


registry.register(EffectDefinition(
    name="vignette",
    label="Vignette",
    description="Darkens the edges of the frame with a radial gradient.",
    category="Stylize",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=1.0, step=0.01),
        EffectParam(name="radius",   label="Radius",   type="float",
                    default=0.7, min_val=0.3, max_val=1.0, step=0.01),
    ],
    apply_frame=_apply_vignette,
))


# ---------------------------------------------------------------------------
# 4. Chromatic Aberration
# ---------------------------------------------------------------------------
def _apply_chromatic_aberration(frame: np.ndarray, params: dict) -> np.ndarray:
    amount = int(params.get("amount", 0))
    if amount <= 0:
        return frame
    h, w = frame.shape[:2]
    b, g, r = cv2.split(frame)
    # Shift R channel left by `amount`, B channel right by `amount`
    M_r = np.float32([[1, 0, -amount], [0, 1, 0]])
    M_b = np.float32([[1, 0,  amount], [0, 1, 0]])
    r_shifted = cv2.warpAffine(r, M_r, (w, h))
    b_shifted = cv2.warpAffine(b, M_b, (w, h))
    return cv2.merge([b_shifted, g, r_shifted])


registry.register(EffectDefinition(
    name="chromatic_aberration",
    label="Chromatic Aberration",
    description="Shifts R and B channels in opposite directions for an RGB fringe effect.",
    category="Distort",
    params=[
        EffectParam(name="amount", label="Amount", type="int",
                    default=0, min_val=0, max_val=20, step=1),
    ],
    apply_frame=_apply_chromatic_aberration,
))


# ---------------------------------------------------------------------------
# 5. Film Grain
# ---------------------------------------------------------------------------
def _apply_film_grain(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    noise = np.random.normal(0, strength, frame.shape)
    noisy = frame.astype(np.float32) + noise
    return _to_uint8(noisy)


registry.register(EffectDefinition(
    name="film_grain",
    label="Film Grain",
    description="Adds analogue-style random noise to the frame.",
    category="Stylize",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=50.0, step=0.5),
    ],
    apply_frame=_apply_film_grain,
))


# ---------------------------------------------------------------------------
# 6. Flip Horizontal
# ---------------------------------------------------------------------------
def _apply_flip_horizontal(frame: np.ndarray, params: dict) -> np.ndarray:
    if not params.get("enabled", False):
        return frame
    return cv2.flip(frame, 1)


registry.register(EffectDefinition(
    name="flip_horizontal",
    label="Flip Horizontal",
    description="Mirrors the frame left-to-right.",
    category="General",
    params=[
        EffectParam(name="enabled", label="Enabled", type="bool", default=False),
    ],
    apply_frame=_apply_flip_horizontal,
))


# ---------------------------------------------------------------------------
# 7. Flip Vertical
# ---------------------------------------------------------------------------
def _apply_flip_vertical(frame: np.ndarray, params: dict) -> np.ndarray:
    if not params.get("enabled", False):
        return frame
    return cv2.flip(frame, 0)


registry.register(EffectDefinition(
    name="flip_vertical",
    label="Flip Vertical",
    description="Flips the frame upside-down.",
    category="General",
    params=[
        EffectParam(name="enabled", label="Enabled", type="bool", default=False),
    ],
    apply_frame=_apply_flip_vertical,
))


# ---------------------------------------------------------------------------
# 8. Pixelate
# ---------------------------------------------------------------------------
def _apply_pixelate(frame: np.ndarray, params: dict) -> np.ndarray:
    block_size = max(1, int(params.get("block_size", 1)))
    if block_size <= 1:
        return frame
    h, w = frame.shape[:2]
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


registry.register(EffectDefinition(
    name="pixelate",
    label="Pixelate",
    description="Reduces resolution into large square blocks.",
    category="Stylize",
    params=[
        EffectParam(name="block_size", label="Block Size", type="int",
                    default=1, min_val=1, max_val=50, step=1),
    ],
    apply_frame=_apply_pixelate,
))


# ---------------------------------------------------------------------------
# 9. Emboss
# ---------------------------------------------------------------------------
_EMBOSS_KERNEL = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2],
], dtype=np.float32)


def _apply_emboss(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    embossed_gray = cv2.filter2D(gray, -1, _EMBOSS_KERNEL) + 128
    embossed_bgr = cv2.cvtColor(_to_uint8(embossed_gray), cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(frame, 1.0 - strength, embossed_bgr, strength, 0)
    return _to_uint8(blended)


registry.register(EffectDefinition(
    name="emboss",
    label="Emboss",
    description="Gives the frame a sculpted, raised-relief look.",
    category="Stylize",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=1.0, step=0.01),
    ],
    apply_frame=_apply_emboss,
))


# ---------------------------------------------------------------------------
# 10. Edge Glow (neon outline)
# ---------------------------------------------------------------------------
def _apply_edge_glow(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    color_r = int(params.get("color_r", 0))
    color_g = int(params.get("color_g", 255))
    color_b = int(params.get("color_b", 255))

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Slightly blur the edges for a glow look
    edges_blurred = cv2.GaussianBlur(edges, (5, 5), 0)

    # Build a coloured edge layer (BGR order)
    edge_layer = np.zeros_like(frame, dtype=np.uint8)
    edge_layer[:, :, 0] = (edges_blurred.astype(np.float32) * (color_b / 255.0)).astype(np.uint8)
    edge_layer[:, :, 1] = (edges_blurred.astype(np.float32) * (color_g / 255.0)).astype(np.uint8)
    edge_layer[:, :, 2] = (edges_blurred.astype(np.float32) * (color_r / 255.0)).astype(np.uint8)

    blended = cv2.addWeighted(frame, 1.0, edge_layer, strength, 0)
    return _to_uint8(blended)


registry.register(EffectDefinition(
    name="edge_glow",
    label="Edge Glow",
    description="Detects edges with Canny and blends a neon-coloured glow back onto the frame.",
    category="Stylize",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0,   min_val=0.0, max_val=1.0, step=0.01),
        EffectParam(name="color_r", label="Glow R",   type="int",
                    default=0,   min_val=0,   max_val=255, step=1),
        EffectParam(name="color_g", label="Glow G",   type="int",
                    default=255, min_val=0,   max_val=255, step=1),
        EffectParam(name="color_b", label="Glow B",   type="int",
                    default=255, min_val=0,   max_val=255, step=1),
    ],
    apply_frame=_apply_edge_glow,
))


# ---------------------------------------------------------------------------
# 11. Sepia
# ---------------------------------------------------------------------------
def _apply_sepia(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    # BGR sepia kernel
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ], dtype=np.float32)
    sepia = cv2.transform(frame.astype(np.float32), kernel)
    sepia = np.clip(sepia, 0, 255)
    blended = cv2.addWeighted(frame.astype(np.float32), 1.0 - strength,
                               sepia, strength, 0)
    return _to_uint8(blended)


registry.register(EffectDefinition(
    name="sepia",
    label="Sepia Tone",
    description="Classic warm-brown sepia colour grade.",
    category="Colour",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=1.0, step=0.01),
    ],
    apply_frame=_apply_sepia,
))


# ---------------------------------------------------------------------------
# 12. Colour Temperature  (cool ↔ warm)
# ---------------------------------------------------------------------------
def _apply_color_temperature(frame: np.ndarray, params: dict) -> np.ndarray:
    temp = float(params.get("temperature", 0))   # −100 (cool) … +100 (warm)
    if temp == 0:
        return frame
    f = frame.astype(np.float32)
    shift = temp * 0.8
    # Warm → boost R, cut B.  Cool → boost B, cut R.
    f[:, :, 2] = np.clip(f[:, :, 2] + shift, 0, 255)  # R  (BGR index 2)
    f[:, :, 0] = np.clip(f[:, :, 0] - shift, 0, 255)  # B  (BGR index 0)
    return _to_uint8(f)


registry.register(EffectDefinition(
    name="color_temperature",
    label="Colour Temperature",
    description="Shift the colour balance warm (positive) or cool (negative).",
    category="Colour",
    params=[
        EffectParam(name="temperature", label="Temperature", type="float",
                    default=0, min_val=-100.0, max_val=100.0, step=1.0),
    ],
    apply_frame=_apply_color_temperature,
))


# ---------------------------------------------------------------------------
# 13. Bilateral Denoise  (edge-preserving noise reduction)
# ---------------------------------------------------------------------------
def _apply_bilateral_denoise(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    d = max(3, int(strength * 15))
    sigma = strength * 75
    denoised = cv2.bilateralFilter(frame, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    return denoised


registry.register(EffectDefinition(
    name="bilateral_denoise",
    label="Bilateral Denoise",
    description="Edge-preserving noise reduction using bilateral filtering.",
    category="Blur",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=1.0, step=0.01),
    ],
    apply_frame=_apply_bilateral_denoise,
))


# ---------------------------------------------------------------------------
# 14. Motion Blur  (directional)
# ---------------------------------------------------------------------------
def _apply_motion_blur(frame: np.ndarray, params: dict) -> np.ndarray:
    size = int(params.get("size", 0))
    angle = float(params.get("angle", 0))   # degrees
    if size <= 1:
        return frame
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    rad = np.deg2rad(angle)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    rotated = cv2.warpAffine(kernel, M, (size, size))
    s = rotated.sum()
    if s > 0:
        rotated /= s
    return cv2.filter2D(frame, -1, rotated)


registry.register(EffectDefinition(
    name="motion_blur",
    label="Motion Blur",
    description="Directional blur simulating camera or subject motion.",
    category="Blur",
    params=[
        EffectParam(name="size",  label="Size",  type="int",
                    default=0, min_val=0, max_val=60, step=2),
        EffectParam(name="angle", label="Angle°", type="float",
                    default=0, min_val=0.0, max_val=360.0, step=5.0),
    ],
    apply_frame=_apply_motion_blur,
))


# ---------------------------------------------------------------------------
# 15. Lens Distortion  (barrel / pincushion)
# ---------------------------------------------------------------------------
def _apply_lens_distortion(frame: np.ndarray, params: dict) -> np.ndarray:
    k1 = float(params.get("k1", 0))   # barrel < 0, pincushion > 0
    if abs(k1) < 0.001:
        return frame
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # Camera matrix (unity focal length scaled to image size)
    K = np.array([[w, 0, cx],
                  [0, w, cy],
                  [0, 0,  1]], dtype=np.float32)
    dist = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float32)
    new_K = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))[0]
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


registry.register(EffectDefinition(
    name="lens_distortion",
    label="Lens Distortion",
    description="Barrel (negative) or pincushion (positive) lens warp.",
    category="Distort",
    params=[
        EffectParam(name="k1", label="Distortion", type="float",
                    default=0, min_val=-0.5, max_val=0.5, step=0.01),
    ],
    apply_frame=_apply_lens_distortion,
))


# ---------------------------------------------------------------------------
# 16. Glitch  (RGB channel shift + horizontal scan-line tears)
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(seed=0)   # deterministic seed for preview


def _apply_glitch(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    out = frame.copy()
    # Horizontal channel shift
    max_shift = max(1, int(strength * w * 0.04))
    shift_r = int(_rng.integers(-max_shift, max_shift))
    shift_b = int(_rng.integers(-max_shift, max_shift))
    out[:, :, 2] = np.roll(out[:, :, 2], shift_r, axis=1)   # R
    out[:, :, 0] = np.roll(out[:, :, 0], shift_b, axis=1)   # B
    # Random scan-line tears
    n_tears = max(0, int(strength * 6))
    for _ in range(n_tears):
        y = int(_rng.integers(0, h))
        tear_h = int(_rng.integers(1, max(2, int(h * 0.03))))
        x_shift = int(_rng.integers(-max_shift * 3, max_shift * 3))
        y2 = min(h, y + tear_h)
        out[y:y2, :] = np.roll(out[y:y2, :], x_shift, axis=1)
    return out


registry.register(EffectDefinition(
    name="glitch",
    label="Glitch",
    description="Digital glitch artefacts: channel shift and scan-line tears.",
    category="Stylize",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=1.0, step=0.01),
    ],
    apply_frame=_apply_glitch,
))


# ---------------------------------------------------------------------------
# 17. Halftone  (CMY dot-screen newspaper effect)
# ---------------------------------------------------------------------------
def _apply_halftone(frame: np.ndarray, params: dict) -> np.ndarray:
    dot_size = max(2, int(params.get("dot_size", 4)))
    if dot_size < 2:
        return frame
    h, w = frame.shape[:2]
    # Work at reduced resolution then upscale for efficiency
    small_w = max(1, w // dot_size)
    small_h = max(1, h // dot_size)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    # For each cell, decide fill fraction per channel
    canvas = np.zeros_like(frame)
    for row in range(small_h):
        for col in range(small_w):
            pixel = small[row, col].astype(np.float32) / 255.0  # BGR
            x0, y0 = col * dot_size, row * dot_size
            x1, y1 = min(w, x0 + dot_size), min(h, y0 + dot_size)
            # Draw coloured dot scaled to pixel brightness
            for ch in range(3):
                radius = int(pixel[ch] * (dot_size // 2))
                if radius > 0:
                    cx = (x0 + x1) // 2
                    cy = (y0 + y1) // 2
                    cv2.circle(canvas, (cx, cy), radius, (
                        255 if ch == 0 else 0,
                        255 if ch == 1 else 0,
                        255 if ch == 2 else 0,
                    ), -1)
    return canvas


registry.register(EffectDefinition(
    name="halftone",
    label="Halftone",
    description="Newspaper-style dot-screen effect with per-channel circles.",
    category="Stylize",
    params=[
        EffectParam(name="dot_size", label="Dot Size", type="int",
                    default=4, min_val=2, max_val=20, step=1),
    ],
    apply_frame=_apply_halftone,
))


# ---------------------------------------------------------------------------
# 18. Old Film  (dust, scratches, vignette flicker, desaturation)
# ---------------------------------------------------------------------------
_old_film_rng = np.random.default_rng(seed=42)


def _apply_old_film(frame: np.ndarray, params: dict) -> np.ndarray:
    strength = float(params.get("strength", 0))
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)

    # Slight desaturation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = cv2.addWeighted(out, 1.0 - strength * 0.6, gray_bgr, strength * 0.6, 0)

    # Sepia-ish warm tint
    out[:, :, 2] = np.clip(out[:, :, 2] + strength * 30, 0, 255)   # R
    out[:, :, 0] = np.clip(out[:, :, 0] - strength * 15, 0, 255)   # B

    # Vignette
    cx, cy = w / 2, h / 2
    max_d = np.sqrt(cx ** 2 + cy ** 2)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    vig = np.clip(1.0 - (dist / max_d) * strength * 1.2, 0, 1)[:, :, np.newaxis]
    out = out * vig

    # Dust/scratch lines
    n_scratches = int(strength * 4)
    for _ in range(n_scratches):
        x = int(_old_film_rng.integers(0, w))
        scratch_h = int(_old_film_rng.integers(h // 4, h))
        y_start = int(_old_film_rng.integers(0, h - scratch_h))
        bright = float(_old_film_rng.uniform(180, 255))
        out[y_start:y_start + scratch_h, max(0, x - 1):x + 1] = bright

    return _to_uint8(out)


registry.register(EffectDefinition(
    name="old_film",
    label="Old Film",
    description="Vintage film look: desaturation, sepia tint, vignette, dust scratches.",
    category="Stylize",
    params=[
        EffectParam(name="strength", label="Strength", type="float",
                    default=0, min_val=0.0, max_val=1.0, step=0.01),
    ],
    apply_frame=_apply_old_film,
))


# ---------------------------------------------------------------------------
# Auto-load plugins from effects/ subfolder
# ---------------------------------------------------------------------------
_PLUGIN_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "effects")
registry.load_plugin_folder(_PLUGIN_FOLDER)
