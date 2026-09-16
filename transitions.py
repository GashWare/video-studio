# transitions.py
"""
Engine-level transition compositing functions.
All operations work on RGB uint8 numpy arrays.
"""

import cv2
import numpy as np

TRANSITION_TYPES = [
    "None", "Cross Dissolve", "Fade to Black",
    "Wipe Left", "Wipe Right", "Zoom Blur",
    "Slide Up", "Slide Down", "Iris Wipe", "Glitch Cut",
]


# ---------------------------------------------------------------------------
# Individual transition implementations
# ---------------------------------------------------------------------------

def _cross_dissolve(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """Linear alpha blend from frame_a → frame_b."""
    alpha = np.float32(progress)
    return cv2.addWeighted(frame_a, float(1.0 - alpha), frame_b, float(alpha), 0)


def _fade_to_black(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """Fade A to black (0-0.5), then black to B (0.5-1.0)."""
    black = np.zeros_like(frame_a)
    if progress <= 0.5:
        t = progress * 2.0          # 0 → 1 during first half
        return cv2.addWeighted(frame_a, float(1.0 - t), black, float(t), 0)
    else:
        t = (progress - 0.5) * 2.0  # 0 → 1 during second half
        return cv2.addWeighted(black, float(1.0 - t), frame_b, float(t), 0)


def _wipe_left(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """Horizontal wipe: frame_b enters from the right, pushing left."""
    H, W = frame_a.shape[:2]
    split = int(W * progress)           # right edge of B region
    out = frame_a.copy()
    if split > 0:
        out[:, W - split:] = frame_b[:, W - split:]
    return out


def _wipe_right(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """Horizontal wipe: frame_b enters from the left, pushing right."""
    H, W = frame_a.shape[:2]
    split = int(W * progress)           # left edge of B region
    out = frame_a.copy()
    if split > 0:
        out[:, :split] = frame_b[:, :split]
    return out


def _zoom_blur(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """
    frame_a zooms/scales out and fades, frame_b scales in and fades up.
    The two scaled frames are blended together.
    """
    H, W = frame_a.shape[:2]

    # frame_a: starts at full size, shrinks toward the centre
    scale_a = 1.0 + progress * 0.5        # 1.0 → 1.5  (zoom out illusion: we scale up and crop)
    # frame_b: starts small, grows to full size
    scale_b = 0.5 + progress * 0.5        # 0.5 → 1.0

    def _scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))
        scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Centre-crop or centre-pad to original size
        if scale >= 1.0:
            x0 = (new_w - W) // 2
            y0 = (new_h - H) // 2
            return scaled[y0:y0 + H, x0:x0 + W]
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            x0 = (W - new_w) // 2
            y0 = (H - new_h) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w] = scaled
            return canvas

    fa = _scale_frame(frame_a, scale_a)
    fb = _scale_frame(frame_b, scale_b)
    alpha = float(progress)
    return cv2.addWeighted(fa, 1.0 - alpha, fb, alpha, 0)


def _slide_up(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """frame_b slides up from the bottom, pushing frame_a upward."""
    H, W = frame_a.shape[:2]
    offset = int(H * progress)
    out = np.empty_like(frame_a)
    # Top part: frame_a shifted up
    if offset < H:
        out[:H - offset, :] = frame_a[offset:, :]
    # Bottom part: frame_b entering from bottom
    if offset > 0:
        out[H - offset:, :] = frame_b[:offset, :]
    return out


def _slide_down(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """frame_b slides down from the top, pushing frame_a downward."""
    H, W = frame_a.shape[:2]
    offset = int(H * progress)
    out = np.empty_like(frame_a)
    if offset < H:
        out[offset:, :] = frame_a[:H - offset, :]
    if offset > 0:
        out[:offset, :] = frame_b[H - offset:, :]
    return out


def _iris_wipe(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """Circular iris expanding from centre revealing frame_b."""
    H, W = frame_a.shape[:2]
    max_r = int(np.sqrt((W / 2) ** 2 + (H / 2) ** 2)) + 1
    radius = int(progress * max_r)
    cx, cy = W // 2, H // 2
    Y, X = np.ogrid[:H, :W]
    mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2
    out = frame_a.copy()
    out[mask] = frame_b[mask]
    return out


def _glitch_cut(frame_a: np.ndarray, frame_b: np.ndarray, progress: float) -> np.ndarray:
    """
    Glitchy horizontal-slice transition: random rows switch to frame_b
    as progress increases, with channel offset artefacts.
    """
    H, W = frame_a.shape[:2]
    rng = np.random.default_rng(seed=int(progress * 1000))
    out = frame_a.copy()

    # How many rows to flip (proportional to progress)
    n_rows = max(0, int(progress * H))
    if n_rows > 0:
        rows = rng.choice(H, size=n_rows, replace=False)
        out[rows] = frame_b[rows]

    # Channel shift artefact
    shift = int(rng.integers(1, max(2, int(W * 0.03))))
    out[:, :, 2] = np.roll(out[:, :, 2], shift, axis=1)
    out[:, :, 0] = np.roll(out[:, :, 0], -shift, axis=1)
    return out


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def apply_transition(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    progress: float,
    transition_type: str,
) -> np.ndarray:
    """
    Composite two frames using a transition effect.

    Parameters
    ----------
    frame_a : RGB uint8 ndarray (H, W, 3)
    frame_b : RGB uint8 ndarray (H, W, 3) — must match frame_a shape
    progress : float in [0.0, 1.0]  — 0 = fully frame_a, 1 = fully frame_b
    transition_type : one of TRANSITION_TYPES

    Returns
    -------
    Composited RGB uint8 ndarray (H, W, 3).
    """
    progress = float(np.clip(progress, 0.0, 1.0))

    if transition_type == "None" or progress == 0.0:
        return frame_a.copy()
    if progress == 1.0:
        return frame_b.copy()

    # Ensure same shape
    if frame_a.shape != frame_b.shape:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

    dispatch = {
        "Cross Dissolve": _cross_dissolve,
        "Fade to Black":  _fade_to_black,
        "Wipe Left":      _wipe_left,
        "Wipe Right":     _wipe_right,
        "Zoom Blur":      _zoom_blur,
        "Slide Up":       _slide_up,
        "Slide Down":     _slide_down,
        "Iris Wipe":      _iris_wipe,
        "Glitch Cut":     _glitch_cut,
    }
    fn = dispatch.get(transition_type, _cross_dissolve)
    return fn(frame_a, frame_b, progress)


# ---------------------------------------------------------------------------
# Pre-render helper
# ---------------------------------------------------------------------------

def get_transition_frames(
    clip_a_path: str,
    clip_b_path: str,
    transition_type: str,
    duration_s: float,
    fps: float,
) -> list:
    """
    Pre-render transition frames between the tail of clip_a and the head of clip_b.

    Parameters
    ----------
    clip_a_path : path to the first video file
    clip_b_path : path to the second video file
    transition_type : one of TRANSITION_TYPES
    duration_s : duration of the transition in seconds
    fps : frames per second for sampling

    Returns
    -------
    list of RGB uint8 ndarrays, length = round(duration_s * fps)
    """
    if transition_type == "None":
        return []

    n_frames = max(1, round(duration_s * fps))

    cap_a = cv2.VideoCapture(clip_a_path)
    cap_b = cv2.VideoCapture(clip_b_path)

    try:
        # Seek clip_a to its tail  (last duration_s seconds)
        total_frames_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_a = cap_a.get(cv2.CAP_PROP_FPS) or fps
        start_frame_a = max(0, total_frames_a - round(duration_s * fps_a))
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame_a)

        # clip_b starts from the beginning
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Grab one representative frame from each
        ok_a, bgr_a = cap_a.read()
        ok_b, bgr_b = cap_b.read()

        if not ok_a or not ok_b:
            return []

        rgb_a = cv2.cvtColor(bgr_a, cv2.COLOR_BGR2RGB)
        rgb_b = cv2.cvtColor(bgr_b, cv2.COLOR_BGR2RGB)

        frames: list = []
        for i in range(n_frames):
            progress = i / max(n_frames - 1, 1)

            # Try to advance each capture by the appropriate number of frames
            frame_offset = round(i * fps_a / fps)
            cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame_a + frame_offset)
            ok, frm = cap_a.read()
            if ok:
                rgb_a = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

            fps_b = cap_b.get(cv2.CAP_PROP_FPS) or fps
            cap_b.set(cv2.CAP_PROP_POS_FRAMES, round(i * fps_b / fps))
            ok, frm = cap_b.read()
            if ok:
                rgb_b = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

            frames.append(apply_transition(rgb_a, rgb_b, progress, transition_type))

    finally:
        cap_a.release()
        cap_b.release()

    return frames
