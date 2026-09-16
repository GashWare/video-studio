# waveform.py
"""
Audio waveform extraction and timeline rendering.

Uses soundfile for fast audio extraction (waveform-only use).
Falls back to moviepy.AudioFileClip if soundfile is unavailable.
Samples at ~200 samples per second, then downsamples to the requested
pixel width.
"""

import threading
import tkinter as tk
import numpy as np


# Target internal sample rate for waveform extraction (samples per second)
_INTERNAL_SAMPLES_PER_SEC = 200


class WaveformCache:
    """
    Extracts and caches audio waveform data (RMS per pixel column)
    for timeline rendering. Uses a background thread per clip.
    Results cached in memory by file path.
    """

    def __init__(self):
        self._cache: dict = {}       # path -> np.ndarray (normalised RMS)
        self._pending: set = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, path: str, width_px: int, callback=None):
        """
        Returns cached waveform array of shape (width_px,) with values
        0.0–1.0, or None if not yet computed. Starts background
        computation if not pending.
        callback(path) is called from the background thread when done.
        """
        with self._lock:
            if path in self._cache:
                return self._resample(self._cache[path], width_px)
            if path not in self._pending:
                self._pending.add(path)
                t = threading.Thread(
                    target=self._compute,
                    args=(path, width_px, callback),
                    daemon=True,
                )
                t.start()
        return None

    def invalidate(self, path: str):
        """Remove cached waveform for path."""
        with self._lock:
            self._cache.pop(path, None)
            self._pending.discard(path)

    def draw_waveform(self, canvas: tk.Canvas, waveform: np.ndarray,
                      x1: float, y1: float, x2: float, y2: float,
                      color: str = "#2ecc71", tag: str = "waveform"):
        """
        Draw the waveform as a filled polygon on a tk.Canvas between
        the given pixel bounds. Clips to [x1, x2] range.
        """
        if waveform is None or len(waveform) == 0:
            return

        width = int(x2 - x1)
        height = int(y2 - y1)
        if width <= 0 or height <= 0:
            return

        mid_y = (y1 + y2) / 2.0
        half_h = height / 2.0

        # Resample waveform to pixel width
        data = self._resample(waveform, width)

        # Build polygon: top line (positive) then bottom (mirrored), left-to-right
        top_pts = []
        bot_pts = []
        for i, rms in enumerate(data):
            x = x1 + i
            amp = float(rms) * half_h
            top_pts.append((x, mid_y - amp))
            bot_pts.append((x, mid_y + amp))

        # Close the polygon
        points = top_pts + list(reversed(bot_pts))
        flat = [coord for pt in points for coord in pt]

        if len(flat) >= 4:
            canvas.create_polygon(flat, fill=color, outline="", tags=tag)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute(self, path: str, width_px: int, callback):
        """
        Background thread: extract audio, compute RMS per chunk,
        normalise, store in cache.
        """
        try:
            samples, sr = self._load_audio(path)
            if samples is None or sr is None:
                return

            # Mix down to mono
            if samples.ndim > 1:
                samples = samples.mean(axis=1 if samples.shape[1] > samples.shape[0] else 0)

            # Compute RMS over chunks of ~(sr / _INTERNAL_SAMPLES_PER_SEC) samples
            chunk_size = max(1, int(sr / _INTERNAL_SAMPLES_PER_SEC))
            n_chunks = max(1, len(samples) // chunk_size)
            rms_array = np.zeros(n_chunks, dtype=np.float32)
            for i in range(n_chunks):
                chunk = samples[i * chunk_size: (i + 1) * chunk_size]
                rms_array[i] = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            # Normalise to 0.0–1.0
            peak = rms_array.max()
            if peak > 0:
                rms_array /= peak

            with self._lock:
                self._cache[path] = rms_array
                self._pending.discard(path)

            if callback is not None:
                try:
                    callback(path)
                except Exception:
                    pass

        except Exception:
            with self._lock:
                self._pending.discard(path)

    def _load_audio(self, path: str):
        """
        Load raw audio samples. Tries soundfile first (fast), then
        moviepy as fallback. Returns (samples_ndarray, sample_rate) or
        (None, None) on failure.
        """
        # --- attempt 1: soundfile ---
        try:
            import soundfile as sf
            samples, sr = sf.read(path, always_2d=False)
            return samples, sr
        except Exception:
            pass

        # --- attempt 2: moviepy AudioFileClip ---
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(path)
            sr = int(clip.fps)
            # nframes can be large; read in one shot
            samples = clip.to_soundarray(fps=sr)
            clip.close()
            return samples, sr
        except Exception:
            pass

        return None, None

    @staticmethod
    def _resample(rms_array: np.ndarray, width_px: int) -> np.ndarray:
        """
        Downsample (or upsample) rms_array to exactly width_px points
        using simple mean-pooling for downsampling and linear interp
        for upsampling.
        """
        n = len(rms_array)
        if n == 0 or width_px <= 0:
            return np.zeros(max(width_px, 0), dtype=np.float32)

        if n == width_px:
            return rms_array

        # Use numpy interpolation (linear) — works for both up and downsampling
        old_x = np.linspace(0.0, 1.0, n)
        new_x = np.linspace(0.0, 1.0, width_px)
        return np.interp(new_x, old_x, rms_array).astype(np.float32)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

waveform_cache = WaveformCache()
