"""
Async proxy builder and real-time frame-queue playback worker.

ProxyWorker   — background thread that builds the MoviePy composite proxy.
FrameQueue    — producer/consumer pair for smooth wall-clock-accurate playback with sounddevice audio.
"""

import copy
import logging
import os
import queue
import threading
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    VideoFileClip,
)
import moviepy.audio.fx.all as afx
from engine import apply_audio_pan

log = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif')


def _is_image_file(path: str) -> bool:
    if not isinstance(path, str):
        return False
    ext = os.path.splitext(path.lower())[1]
    return ext in _IMAGE_EXTENSIONS


def _track_layer_rank(trk_str: str) -> int:
    if not trk_str:
        return 1000
    prefix = trk_str[0].upper()
    idx = int(trk_str[1:]) if trk_str[1:].isdigit() else 1
    ranks = {'V': 1000, 'I': 2000, 'T': 3000, 'S': 3000}
    return ranks.get(prefix, 1000) + idx


# ---------------------------------------------------------------------------
# ProxyWorker
# ---------------------------------------------------------------------------

class ProxyWorker(threading.Thread):
    """
    Builds a low-res MoviePy CompositeVideoClip in a background thread.
    Cancels any previous in-flight build when a new one is requested.
    Delivers the finished clip to the UI thread via app.after(0, callback).
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._lock      = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending: Optional[Tuple] = None
        self._cancel    = threading.Event()
        self._running   = True
        self.start()

    def request(
        self,
        project_clips: list,
        proxy_w: int,
        proxy_h: int,
        proj_w: int,
        proj_h: int,
        app,
        on_done: Callable,
        track_volumes: Optional[dict] = None,
        track_pans: Optional[dict] = None,
        track_muted: Optional[dict] = None,
        track_solo: Optional[dict] = None,
    ) -> None:
        """
        Schedule a new proxy build. Cancels the current one if running.
        """
        clips_copy = copy.deepcopy(project_clips)
        t_vols = dict(track_volumes) if track_volumes else {}
        t_pans = dict(track_pans) if track_pans else {}
        t_mute = dict(track_muted) if track_muted else {}
        t_solo = dict(track_solo) if track_solo else {}

        with self._condition:
            self._cancel.set()
            self._pending = (
                clips_copy, proxy_w, proxy_h, proj_w, proj_h, app, on_done,
                t_vols, t_pans, t_mute, t_solo
            )
            self._condition.notify()

    def _build(
        self,
        clips: list,
        proxy_w: int,
        proxy_h: int,
        proj_w: int,
        proj_h: int,
        track_volumes: dict,
        track_pans: dict,
        track_muted: dict,
        track_solo: dict,
    ) -> Tuple:
        """
        Perform the MoviePy composite build across all video and audio tracks.
        """
        max_dur = max(
            c["start"] + (c["trim_out"] - c["trim_in"]) for c in clips
        )
        max_dur = max(max_dur, 1.0)

        v_layers: list = [
            ColorClip(size=(proxy_w, proxy_h), color=(0, 0, 0), duration=max_dur)
        ]
        a_layers: list = []
        open_clips: List = []

        any_solo = any(track_solo.values())

        def is_track_active(trk: str) -> bool:
            if any_solo:
                return track_solo.get(trk, False)
            return not track_muted.get(trk, False)

        # ── Visual layers (Video, Image, Subtitle/Title) ────────────
        v_data = sorted(
            [c for c in clips if not c.get("track", "").startswith("A")],
            key=lambda x: _track_layer_rank(x.get("track", "V1")),
        )

        for d in v_data:
            if self._cancel.is_set():
                for oc in open_clips:
                    try:
                        oc.close()
                    except Exception:
                        pass
                return None, []

            trk = d.get("track", "V1")
            active = is_track_active(trk)
            path = d.get("path", "")
            trim_in = float(d.get("trim_in", 0.0))
            trim_out = float(d.get("trim_out", 3.0))
            duration = max(0.01, trim_out - trim_in)

            try:
                if path == "__title__" or "title_config" in d or "title_text" in d:
                    from title_generator import TitleClip
                    if "title_config" in d:
                        tc = TitleClip.from_config(d["title_config"])
                    else:
                        text = d.get("title_text", "Subtitle")
                        f_size = max(12, int(d.get("title_font_size", 48) * proxy_h / proj_h))
                        f_color = d.get("title_color", (255, 255, 255))
                        bg_color = d.get("title_bg_color", (0, 0, 0, 0))
                        anim = d.get("title_animation", "Fade In/Out")
                        align = d.get("title_alignment", "center")
                        tc = TitleClip(
                            text=text,
                            duration=duration,
                            fps=30.0,
                            width=proxy_w,
                            height=proxy_h,
                            font_size=f_size,
                            font_color=f_color,
                            bg_color=bg_color,
                            alignment=align,
                            animation=anim,
                        )
                    clip = tc.to_moviepy_clip().resize((proxy_w, proxy_h))
                elif _is_image_file(path) or trk.startswith("I"):
                    from moviepy.editor import ImageClip
                    clip = ImageClip(path).set_duration(duration).resize(height=proxy_h)
                else:
                    clip = (
                        VideoFileClip(path)
                        .subclip(trim_in, trim_out)
                        .resize(height=proxy_h)
                    )
                    open_clips.append(clip)

                if d.get("scale", 1.0) != 1.0:
                    clip = clip.resize(d["scale"])

                px = (
                    d.get("x")
                    if isinstance(d.get("x"), str)
                    else int(d["x"] * proxy_w / proj_w)
                )
                py = (
                    d.get("y")
                    if isinstance(d.get("y"), str)
                    else int(d["y"] * proxy_h / proj_h)
                )

                clip = clip.set_pos((px, py)).set_start(d["start"])

                # Handle audio on video clips
                if getattr(clip, "audio", None):
                    if not active or d.get("mute_audio", False):
                        clip = clip.without_audio()
                    else:
                        t_vol = track_volumes.get(trk, 1.0)
                        t_pan = track_pans.get(trk, 0.0)
                        eff_vol = d.get("volume", 1.0) * t_vol
                        if eff_vol == 0:
                            clip = clip.without_audio()
                        else:
                            if eff_vol != 1.0:
                                clip.audio = clip.audio.fx(afx.volumex, eff_vol)
                            if t_pan != 0.0:
                                clip.audio = apply_audio_pan(clip.audio, t_pan)

                v_layers.append(clip)
            except Exception as exc:
                log.warning("ProxyWorker skipped visual clip '%s': %s", d.get("path", ""), exc)

        # ── Audio layers ────────────────────────────────────────────
        a_data = [c for c in clips if c.get("track", "").startswith("A")]
        for d in a_data:
            if self._cancel.is_set():
                for oc in open_clips:
                    try:
                        oc.close()
                    except Exception:
                        pass
                return None, []

            trk = d.get("track", "A1")
            if not is_track_active(trk) or d.get("mute_audio", False):
                continue

            t_vol = track_volumes.get(trk, 1.0)
            t_pan = track_pans.get(trk, 0.0)
            eff_vol = d.get("volume", 1.0) * t_vol
            if eff_vol == 0:
                continue

            try:
                a = AudioFileClip(d["path"]).subclip(d["trim_in"], d["trim_out"])
                open_clips.append(a)
                if eff_vol != 1.0:
                    a = a.fx(afx.volumex, eff_vol)
                if t_pan != 0.0:
                    a = apply_audio_pan(a, t_pan)
                a = a.set_start(d["start"])
                a_layers.append(a)
            except Exception as exc:
                log.warning("ProxyWorker skipped audio clip '%s': %s", d.get("path", ""), exc)

        # ── Master Composite ────────────────────────────────────────
        master = (
            CompositeVideoClip(v_layers, size=(proxy_w, proxy_h))
            .set_duration(max_dur)
        )

        all_audio = a_layers + [
            v.audio.set_start(v.start)
            for v in v_layers
            if getattr(v, "audio", None)
        ]
        if all_audio:
            master = master.set_audio(CompositeAudioClip(all_audio).set_duration(max_dur))

        return master, open_clips

    def run(self) -> None:
        """Idle loop: wait for _pending, build, deliver via app.after(0, ...)."""
        while True:
            with self._condition:
                while self._pending is None and self._running:
                    self._condition.wait()
                if not self._running:
                    return
                work = self._pending
                self._pending = None

            (
                clips, proxy_w, proxy_h, proj_w, proj_h, app, on_done,
                t_vols, t_pans, t_mute, t_solo
            ) = work
            self._cancel.clear()

            if not clips:
                app.after(0, lambda cb=on_done: cb(None, []))
                continue

            try:
                master, open_clips = self._build(
                    clips, proxy_w, proxy_h, proj_w, proj_h,
                    t_vols, t_pans, t_mute, t_solo
                )
            except Exception as exc:
                log.exception("Proxy build failed: %s", exc)
                master, open_clips = None, []

            if self._cancel.is_set() or master is None:
                for oc in open_clips:
                    try:
                        oc.close()
                    except Exception:
                        pass
                continue

            app.after(0, lambda m=master, oc=open_clips, cb=on_done: cb(m, oc))

    def shutdown(self) -> None:
        """Signal the worker thread to exit cleanly."""
        with self._condition:
            self._running = False
            self._condition.notify()


# ---------------------------------------------------------------------------
# FrameQueue
# ---------------------------------------------------------------------------

class FrameQueue:
    """
    Producer/consumer frame buffer for smooth real-time playback with
    sounddevice audio streaming.
    """

    def __init__(self, maxsize: int = 4) -> None:
        self._q            = queue.Queue(maxsize=maxsize)
        self._stop_evt     = threading.Event()
        self._producer: Optional[threading.Thread] = None
        self._audio_stream = None

    def start_playback(
        self,
        proxy_clip,
        fps: float,
        start_t: float,
        app,
        on_frame: Callable,
        on_end: Callable,
    ) -> None:
        """
        Start producer thread and audio output stream.
        """
        self.stop()
        self._stop_evt.clear()

        # Start audio stream if clip has audio
        if getattr(proxy_clip, "audio", None) and proxy_clip.audio.duration > start_t:
            try:
                import sounddevice as sd
                sr = 44100
                audio_clip = proxy_clip.audio
                audio_dur = audio_clip.duration
                t_audio = [start_t]
                stop_audio_evt = self._stop_evt

                def _audio_cb(outdata, frames, time_info, status):
                    if stop_audio_evt.is_set():
                        outdata.fill(0)
                        raise sd.CallbackStop
                    t0 = t_audio[0]
                    t1 = t0 + frames / sr
                    if t0 >= audio_dur:
                        outdata.fill(0)
                        raise sd.CallbackStop
                    t_space = np.linspace(t0, t1, frames, endpoint=False)
                    try:
                        data = audio_clip.get_frame(t_space)
                        if data.ndim == 1:
                            data = np.column_stack((data, data))
                        elif data.shape[1] == 1:
                            data = np.repeat(data, 2, axis=1)
                        outdata[:] = data.astype(np.float32)
                    except Exception:
                        outdata.fill(0)
                    t_audio[0] = t1

                self._audio_stream = sd.OutputStream(
                    samplerate=sr, channels=2, callback=_audio_cb, blocksize=2048
                )
                self._audio_stream.start()
            except Exception as exc:
                log.debug("Audio playback init failed: %s", exc)
                self._audio_stream = None

        self._producer = threading.Thread(
            target=self._produce,
            args=(proxy_clip, fps, start_t, app, on_frame, on_end),
            daemon=True,
        )
        self._producer.start()

    def stop(self) -> None:
        """Stop producer thread and audio stream, then drain queue."""
        self._stop_evt.set()

        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        if self._producer is not None:
            self._producer.join(timeout=2.0)
            self._producer = None

        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _produce(
        self,
        proxy_clip,
        fps: float,
        start_t: float,
        app,
        on_frame: Callable,
        on_end: Callable,
    ) -> None:
        """
        Background thread: iterate frames and push into queue with wall-clock pacing.
        """
        duration    = proxy_clip.duration
        frame_dur   = 1.0 / fps
        t           = start_t
        wall_origin = time.perf_counter()
        frame_index = 0

        while not self._stop_evt.is_set() and t < duration:
            try:
                frame = proxy_clip.get_frame(t)
            except Exception:
                break

            current_t = t

            while not self._stop_evt.is_set():
                try:
                    self._q.put((frame, current_t), timeout=0.05)
                    break
                except queue.Full:
                    continue

            if self._stop_evt.is_set():
                return

            frame_index += 1
            wall_target  = wall_origin + frame_index * frame_dur
            delay_s      = max(0.0, wall_target - time.perf_counter())
            delay_ms     = int(delay_s * 1000)

            def _deliver(cb=on_frame, q=self._q, stop=self._stop_evt):
                if stop.is_set():
                    return
                try:
                    frm, frm_t = q.get_nowait()
                    cb(frm, frm_t)
                except queue.Empty:
                    pass

            app.after(delay_ms, _deliver)
            t += frame_dur

        if not self._stop_evt.is_set():
            app.after(0, on_end)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

proxy_worker: ProxyWorker = ProxyWorker()

