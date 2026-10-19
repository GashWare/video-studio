import cv2
import os
import copy
import threading
import logging
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip, ColorClip  # requires moviepy<2
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx
from proglog import ProgressBarLogger
from config import free_memory

# Optional colour grading / effects pipeline
try:
    from color_grading import ColorGradingPanel, LUT_PRESETS
    _HAS_COLOR_GRADING = True
except ImportError:
    _HAS_COLOR_GRADING = False

try:
    from effects_registry import registry as _effects_registry
    _HAS_EFFECTS = True
except ImportError:
    _HAS_EFFECTS = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

log = logging.getLogger(__name__)


class BGREngine:
    """Optimized AI Background Removal Engine with frame-diff caching."""
    def __init__(self, custom_masks):
        if _ORT_AVAILABLE:
            from rembg import new_session, remove
            _providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in _providers:
                try:
                    self._session = new_session('u2net', providers=['CUDAExecutionProvider'])
                    log.info("BGREngine: using CUDA GPU inference")
                except Exception:
                    self._session = new_session('u2net')
                    log.info("BGREngine: CUDA failed, using CPU inference")
            else:
                self._session = new_session('u2net')
                log.info("BGREngine: using CPU inference")
            self._remove = lambda img: remove(img, session=self._session)
        else:
            from rembg import remove
            self._remove = remove
        self._c_masks  = custom_masks or {}
        self._p_frame  = None
        self._p_mask   = None

    def get(self, frame, f_idx):
        if f_idx in self._c_masks:
            return self._c_masks[f_idx]
        if self._p_frame is not None:
            mse = float((cv2.absdiff(frame, self._p_frame) ** 2).mean())
            if mse < 5.0:
                return self._p_mask
        mask = self._remove(frame)[:, :, 3]
        self._p_frame, self._p_mask = frame, mask
        return mask


class _ProgressLogger(ProgressBarLogger):
    """Bridges proglog callbacks to our UI progress callback."""
    def __init__(self, progress_cb, cancel_flag):
        super().__init__()
        self._progress_cb  = progress_cb
        self._cancel_flag  = cancel_flag

    def bars_callback(self, bar, attr, value, old_value=None):
        if self._cancel_flag.is_set():
            raise InterruptedError("Cancelled by user")
        total = self.bars[bar].get('total', 0)
        if total > 0 and self._progress_cb:
            self._progress_cb((value / total) * 100)


class VideoTask(threading.Thread):
    """Base class for render/export background tasks."""
    def __init__(self, on_success, on_error, on_progress):
        super().__init__(daemon=True)
        self._on_success  = on_success
        self._on_error    = on_error
        self._on_progress = on_progress
        self._cancel_flag = threading.Event()

    def cancel(self):
        self._cancel_flag.set()

    def _progress(self, pct):
        if self._on_progress:
            self._on_progress(pct)

    def _success(self):
        if self._on_success:
            self._on_success()

    def _error(self, msg):
        if self._on_error:
            self._on_error(msg)


_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif')


def is_image_file(path: str) -> bool:
    if not isinstance(path, str):
        return False
    ext = os.path.splitext(path.lower())[1]
    return ext in _IMAGE_EXTENSIONS


def process_clip_data(c_data, cancel_flag):
    """
    Parses Inspector properties into a fully-decorated MoviePy clip.
    Returns the clip positioned in the timeline (set_start applied).
    """
    path = c_data.get('path', '')
    trim_in = float(c_data.get('trim_in', 0.0))
    trim_out = float(c_data.get('trim_out', 3.0))
    duration = max(0.01, trim_out - trim_in)

    if path == '__title__' or 'title_config' in c_data or 'title_text' in c_data:
        try:
            from title_generator import TitleClip
            if 'title_config' in c_data:
                tc = TitleClip.from_config(c_data['title_config'])
            else:
                text = c_data.get('title_text', 'Subtitle')
                f_size = c_data.get('title_font_size', 48)
                f_color = c_data.get('title_color', (255, 255, 255))
                bg_color = c_data.get('title_bg_color', (0, 0, 0, 0))
                anim = c_data.get('title_animation', 'Fade In/Out')
                align = c_data.get('title_alignment', 'center')
                tc = TitleClip(
                    text=text,
                    duration=duration,
                    fps=30.0,
                    width=1920,
                    height=1080,
                    font_size=f_size,
                    font_color=f_color,
                    bg_color=bg_color,
                    alignment=align,
                    animation=anim,
                )
            c = tc.to_moviepy_clip()
        except Exception as exc:
            log.warning("TitleClip processing fallback: %s", exc)
            c = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    elif is_image_file(path) or c_data.get('track', '').startswith('I'):
        try:
            from moviepy.editor import ImageClip
            c = ImageClip(path).set_duration(duration)
        except Exception as exc:
            log.warning("ImageClip processing fallback: %s", exc)
            c = ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    else:
        c = VideoFileClip(path).subclip(trim_in, trim_out)

    # Custom Crop Box (normalized 0.0 - 1.0 coords: x1, y1, x2, y2)
    crop_box = c_data.get('crop')
    if crop_box and isinstance(crop_box, dict):
        w, h = c.size
        cx1 = int(round(float(crop_box.get('x1', 0.0)) * w))
        cy1 = int(round(float(crop_box.get('y1', 0.0)) * h))
        cx2 = int(round(float(crop_box.get('x2', 1.0)) * w))
        cy2 = int(round(float(crop_box.get('y2', 1.0)) * h))
        cx1 = max(0, min(cx1, w - 2))
        cy1 = max(0, min(cy1, h - 2))
        cx2 = max(cx1 + 2, min(cx2, w))
        cy2 = max(cy1 + 2, min(cy2, h))
        c = c.crop(x1=cx1, y1=cy1, x2=cx2, y2=cy2)

    # Aspect ratio crop
    aspect = c_data.get('aspect', 'Original')
    if aspect and aspect != 'Original':
        w, h = c.size
        ratio_map = {
            '16:9': 16/9, '9:16': 9/16, '1:1': 1.0, '4:3': 4/3,
            '21:9': 21/9, '4:5': 4/5, '2:3': 2/3
        }
        tr  = ratio_map.get(aspect, w / h)
        cr  = w / h
        if cr > tr + 0.005:
            new_w = max(2, int(round(h * tr)))
            c = c.crop(x1=(w - new_w) // 2, y1=0, width=new_w, height=h)
        elif cr < tr - 0.005:
            new_h = max(2, int(round(w / tr)))
            c = c.crop(x1=0, y1=(h - new_h) // 2, width=w, height=new_h)

    # Scale
    if c_data.get('scale', 1.0) != 1.0:
        c = c.resize(c_data['scale'])

    # Rotation
    rot_str = c_data.get('rot', '0°')
    if rot_str != '0°':
        angle = int(rot_str.replace('°', ''))
        c = c.fx(vfx.rotate, angle)

    # Speed
    eff_speed = float(c_data.get('speed', 1.0) or 1.0) * float(c_data.get('track_speed', 1.0) or 1.0)
    if eff_speed <= 0:
        eff_speed = 1.0
    if eff_speed != 1.0:
        c = c.fx(vfx.speedx, eff_speed)

    # Audio
    if c_data.get('mute_audio', False) and c.audio:
        c = c.without_audio()
    elif c.audio:
        vol = c_data.get('volume', 1.0)
        if vol != 1.0:
            c = c.fx(afx.volumex, vol)
        if c_data.get('audio_effects'):
            c.audio = apply_clip_audio_effects(c.audio, c_data.get('audio_effects'))

    # Position
    x = c_data.get('x', 'center')
    y = c_data.get('y', 'center')
    c = c.set_pos((x, y))

    # Opacity
    opacity = float(c_data.get('opacity', 1.0))
    if opacity < 1.0:
        c = c.set_opacity(max(0.0, min(1.0, opacity)))

    # Watermark overlay
    wm_text = c_data.get('wm_text', '')
    if wm_text:
        wm_pos = c_data.get('wm_pos', 'Bottom Right')
        def _add_watermark(frame):
            img   = frame.copy()
            fh, fw = img.shape[:2]
            scale  = max(1, int(fh / 400))
            thick  = max(2, int(fh / 250))
            tsize  = cv2.getTextSize(wm_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
            coords = {
                'Top Left':     (20, 30 + tsize[1]),
                'Top Right':    (fw - tsize[0] - 20, 30 + tsize[1]),
                'Bottom Left':  (20, fh - 20),
                'Bottom Right': (fw - tsize[0] - 20, fh - 20),
                'Center':       ((fw - tsize[0]) // 2, (fh + tsize[1]) // 2),
            }
            org = coords.get(wm_pos, coords['Bottom Right'])
            cv2.putText(img, wm_text, org, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), thick, cv2.LINE_AA)
            return img
        c = c.fl_image(_add_watermark)

    # AI background removal
    if c_data.get('rm_bg', False):
        engine = BGREngine(c_data.get('custom_masks'))
        fps    = c.fps or 30

        def _apply_bg(get_frame, t):
            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")
            frame = get_frame(t)
            mask  = engine.get(frame, int(t * fps))
            return cv2.bitwise_and(frame, frame, mask=mask)

        def _mask_func(get_frame, t):
            return engine.get(get_frame(t), int(t * fps)) / 255.0

        c = c.fl(_apply_bg)
        from moviepy.video.VideoClip import VideoClip
        c = c.set_mask(
            VideoClip(lambda t: _mask_func(c.get_frame, t),
                      duration=c.duration, ismask=True)
        )

    # ── Per-clip colour grading ─────────────────────────────────────────
    if _HAS_COLOR_GRADING and c_data.get('color'):
        color_data = c_data['color']
        # Build a temporary panel-less grading function using the dict values
        def _make_grade_fn(cd):
            def _grade(frame):
                # Reuse apply_to_frame logic inline to avoid needing a UI panel
                import cv2 as _cv2
                import numpy as _np
                img = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR).astype(_np.float32)
                brightness = float(cd.get('brightness', 0))
                contrast   = float(cd.get('contrast', 0))
                alpha = 1.0 + contrast / 100.0
                beta  = brightness
                img = _np.clip(img * alpha + beta, 0, 255)
                # LUT preset
                preset = cd.get('lut_preset', 'None')
                lut = LUT_PRESETS.get(preset, LUT_PRESETS['None'])
                img_u8 = _np.clip(img, 0, 255).astype(_np.uint8)
                b_ch = lut[img_u8[:, :, 0], 0]
                g_ch = lut[img_u8[:, :, 1], 1]
                r_ch = lut[img_u8[:, :, 2], 2]
                img  = _np.stack([b_ch, g_ch, r_ch], axis=2).astype(_np.float32)
                # Saturation / hue
                sat = float(cd.get('saturation', 0))
                hue = float(cd.get('hue', 0))
                if sat != 0 or hue != 0:
                    tmp = _np.clip(img, 0, 255).astype(_np.uint8)
                    hsv = _cv2.cvtColor(tmp, _cv2.COLOR_BGR2HSV).astype(_np.float32)
                    if hue != 0:
                        hsv[:, :, 0] = (hsv[:, :, 0] + hue / 2.0) % 180.0
                    if sat != 0:
                        hsv[:, :, 1] = _np.clip(hsv[:, :, 1] + sat * 2.55, 0, 255)
                    img = _cv2.cvtColor(_np.clip(hsv, 0, 255).astype(_np.uint8),
                                        _cv2.COLOR_HSV2BGR).astype(_np.float32)
                # Lift / Gamma / Gain
                lift  = cd.get('lift',  [1.0, 1.0, 1.0])
                gamma = cd.get('gamma', [1.0, 1.0, 1.0])
                gain  = cd.get('gain',  [1.0, 1.0, 1.0])
                channel_map = [(2, 0), (1, 1), (0, 2)]
                img_norm = img / 255.0
                for bgr_i, rgb_i in channel_map:
                    g = max(gamma[rgb_i], 1e-6)
                    img_norm[:, :, bgr_i] = gain[rgb_i] * _np.power(
                        _np.clip(img_norm[:, :, bgr_i] * lift[rgb_i], 0, None), 1.0 / g)
                img = _np.clip(img_norm * 255.0, 0, 255).astype(_np.uint8)
                return _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
            return _grade
        c = c.fl_image(_make_grade_fn(color_data))

    # ── Per-clip effects chain ──────────────────────────────────────────
    if _HAS_EFFECTS and c_data.get('effects'):
        effects_data = c_data['effects']
        enabled = {k: v for k, v in effects_data.items() if v.get('enabled', False)}
        if enabled:
            def _make_effects_fn(ed):
                def _apply_effects(frame):
                    # frame is RGB uint8 from moviepy — convert to BGR for OpenCV effects
                    import cv2 as _cv2
                    bgr = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
                    bgr = _effects_registry.apply_all_enabled(bgr, ed)
                    return _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
                return _apply_effects
            c = c.fl_image(_make_effects_fn(enabled))

    c = c.set_start(c_data['start'])
    return c


def match_clip_colors(source_frame: np.ndarray, target_frame: np.ndarray) -> np.ndarray:
    """
    Match the colour histogram of source_frame to target_frame.
    Returns source_frame adjusted to match target_frame's colour distribution.
    Uses per-channel CDF matching (histogram equalisation across frames).
    Both frames must be RGB uint8, same or different sizes.
    """
    import numpy as np
    result = np.zeros_like(source_frame)
    for ch in range(3):
        src_ch = source_frame[:, :, ch].ravel()
        tgt_ch = target_frame[:, :, ch].ravel()
        # Build CDFs
        src_hist, _ = np.histogram(src_ch, bins=256, range=(0, 256))
        tgt_hist, _ = np.histogram(tgt_ch, bins=256, range=(0, 256))
        src_cdf = src_hist.cumsum() / src_hist.sum()
        tgt_cdf = tgt_hist.cumsum() / tgt_hist.sum()
        # Build lookup: for each source intensity, find target intensity with closest CDF
        lut = np.searchsorted(tgt_cdf, src_cdf).astype('uint8')
        result[:, :, ch] = lut[source_frame[:, :, ch]]
    return result


def apply_blend_mode(base_rgb: np.ndarray, top_rgb: np.ndarray, mode: str = "Normal", opacity: float = 1.0) -> np.ndarray:
    """
    Combines top_rgb over base_rgb using the specified blend mode and opacity.
    Both images are RGB float32 in [0, 1] or uint8 in [0, 255].
    Returns RGB uint8.
    """
    if mode == "Normal" and opacity >= 0.999:
        return top_rgb

    # Work in [0.0, 1.0] float32
    a = base_rgb.astype(np.float32) / 255.0   # Background
    b = top_rgb.astype(np.float32) / 255.0    # Foreground

    if mode == "Multiply":
        blended = a * b
    elif mode == "Screen":
        blended = 1.0 - (1.0 - a) * (1.0 - b)
    elif mode == "Overlay":
        blended = np.where(a <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    elif mode == "Hard Light":
        blended = np.where(b <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    elif mode == "Soft Light":
        blended = np.where(b <= 0.5, 2.0 * a * b + a * a * (1.0 - 2.0 * b), 2.0 * a * (1.0 - b) + np.sqrt(np.maximum(a, 0.0)) * (2.0 * b - 1.0))
    elif mode == "Color Dodge":
        blended = np.where(b >= 1.0, 1.0, np.minimum(1.0, a / (1.0 - b + 1e-6)))
    elif mode == "Color Burn":
        blended = np.where(b <= 0.0, 0.0, 1.0 - np.minimum(1.0, (1.0 - a) / (b + 1e-6)))
    elif mode == "Darken":
        blended = np.minimum(a, b)
    elif mode == "Lighten":
        blended = np.maximum(a, b)
    elif mode == "Difference":
        blended = np.abs(a - b)
    elif mode == "Exclusion":
        blended = a + b - 2.0 * a * b
    elif mode == "Add":
        blended = np.minimum(1.0, a + b)
    elif mode == "Luminosity":
        y_top = 0.2126 * b[:, :, 0] + 0.7152 * b[:, :, 1] + 0.0722 * b[:, :, 2]
        y_base = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
        diff = (y_top - y_base)[:, :, np.newaxis]
        blended = np.clip(a + diff, 0.0, 1.0)
    else:  # Normal
        blended = b

    if opacity < 1.0:
        blended = a * (1.0 - opacity) + blended * opacity

    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def apply_clip_audio_effects(clip, effect_chain_list):
    """Apply an audio effect chain to a MoviePy AudioClip."""
    if not effect_chain_list or clip is None:
        return clip
    try:
        from audio_effects import apply_audio_chain
        fps = int(getattr(clip, 'fps', 44100) or 44100)
        sound_array = clip.to_soundarray(fps=fps)
        processed = apply_audio_chain(sound_array, fps, effect_chain_list)
        from moviepy.audio.AudioClip import AudioArrayClip
        return AudioArrayClip(processed, fps=fps)
    except Exception as exc:
        log.warning("Audio effect chain processing fallback: %s", exc)
        return clip


def apply_audio_pan(clip, pan: float):
    """Apply stereo panning (-1.0 left to +1.0 right) to a MoviePy AudioClip."""
    if pan == 0.0 or clip is None:
        return clip
    left_gain = min(1.0, 1.0 - pan)
    right_gain = min(1.0, 1.0 + pan)

    def _pan_filter(gf, t):
        arr = gf(t)
        if arr.ndim == 1:
            return np.column_stack((arr * left_gain, arr * right_gain))
        elif arr.shape[1] >= 2:
            res = arr.copy()
            res[:, 0] *= left_gain
            res[:, 1] *= right_gain
            return res
        return arr

    return clip.fl(_pan_filter)


class RenderTask(VideoTask):
    """Compiles the full multi-track timeline to a video file."""

    def __init__(self, project_clips, out_path,
                 proj_w, proj_h, fps, codec, bitrate,
                 on_success, on_error, on_progress,
                 track_volumes=None, track_pans=None,
                 track_audio_effects=None, track_speeds=None):
        super().__init__(on_success, on_error, on_progress)
        self._clips               = [copy.deepcopy(c) for c in project_clips]
        self._out                 = out_path
        self._w                   = proj_w
        self._h                   = proj_h
        self._fps                 = fps
        self._codec               = codec
        self._bitrate             = bitrate
        self._track_volumes       = track_volumes or {}
        self._track_pans          = track_pans or {}
        self._track_audio_effects = track_audio_effects or {}
        self._track_speeds        = track_speeds or {}

    def run(self):
        open_clips = []
        try:
            def get_eff_dur(c):
                trk = c.get('track', 'V1')
                eff_spd = float(c.get('speed', 1.0) or 1.0) * float(self._track_speeds.get(trk, 1.0))
                if eff_spd <= 0:
                    eff_spd = 1.0
                return max(0.01, (float(c.get('trim_out', 3.0)) - float(c.get('trim_in', 0.0))) / eff_spd)

            max_dur = max([c['start'] + get_eff_dur(c) for c in self._clips] or [1.0])
            max_dur = max(max_dur, 1.0)

            # ── Visual layers (Video, Image, Titles/Subtitles) ───────
            def _track_layer_rank(trk_str):
                if not trk_str:
                    return 1000
                prefix = trk_str[0].upper()
                idx = int(trk_str[1:]) if trk_str[1:].isdigit() else 1
                ranks = {'V': 1000, 'I': 2000, 'T': 3000, 'S': 3000}
                return ranks.get(prefix, 1000) + idx

            v_layers = [ColorClip(size=(self._w, self._h),
                                   color=(0, 0, 0), duration=max_dur)]
            v_data   = sorted(
                [c for c in self._clips if not c.get('track', '').startswith('A')],
                key=lambda x: _track_layer_rank(x.get('track', 'V1'))
            )
            for d in v_data:
                if self._cancel_flag.is_set():
                    return
                d['track_speed'] = self._track_speeds.get(d.get('track', 'V1'), 1.0)
                clip = process_clip_data(d, self._cancel_flag)
                open_clips.append(clip)

                # Apply track-level volume, pan & audio effects to video's embedded audio if present
                if getattr(clip, 'audio', None):
                    t_vol = self._track_volumes.get(d['track'], 1.0)
                    t_pan = self._track_pans.get(d['track'], 0.0)
                    t_effs = self._track_audio_effects.get(d['track'], [])
                    if t_vol != 1.0:
                        clip.audio = clip.audio.fx(afx.volumex, t_vol)
                    if t_pan != 0.0:
                        clip.audio = apply_audio_pan(clip.audio, t_pan)
                    if t_effs:
                        clip.audio = apply_clip_audio_effects(clip.audio, t_effs)

                v_layers.append(clip)

            # ── Audio layers ────────────────────────────────────────
            a_layers = []
            for d in [c for c in self._clips if c.get('track', '').startswith('A')]:
                if self._cancel_flag.is_set():
                    return
                eff_spd = float(d.get('speed', 1.0) or 1.0) * float(self._track_speeds.get(d.get('track', 'A1'), 1.0))
                if eff_spd <= 0:
                    eff_spd = 1.0

                a = AudioFileClip(d['path']).subclip(d['trim_in'], d['trim_out'])
                if eff_spd != 1.0:
                    a = a.fx(vfx.speedx, eff_spd)

                t_vol = self._track_volumes.get(d['track'], 1.0)
                t_pan = self._track_pans.get(d['track'], 0.0)
                clip_vol = d.get('volume', 1.0)
                effective_vol = 0.0 if d.get('mute_audio', False) else clip_vol * t_vol

                if effective_vol == 0.0:
                    a = a.fx(afx.volumex, 0)
                elif effective_vol != 1.0:
                    a = a.fx(afx.volumex, effective_vol)

                if t_pan != 0.0 and effective_vol > 0.0:
                    a = apply_audio_pan(a, t_pan)

                # Apply clip audio effects
                if d.get('audio_effects'):
                    a = apply_clip_audio_effects(a, d['audio_effects'])

                # Apply track audio effects
                t_effs = self._track_audio_effects.get(d['track'], [])
                if t_effs:
                    a = apply_clip_audio_effects(a, t_effs)

                a = a.set_start(d['start'])
                open_clips.append(a)
                a_layers.append(a)

            # ── Composite ───────────────────────────────────────────
            final = (CompositeVideoClip(v_layers, size=(self._w, self._h))
                     .set_duration(max_dur))

            all_audio = a_layers + [
                v.audio.set_start(v.start)
                for v in v_layers
                if getattr(v, 'audio', None)
            ]
            if all_audio:
                comp_audio = CompositeAudioClip(all_audio).set_duration(max_dur)
                master_effs = self._track_audio_effects.get('Master', [])
                if master_effs:
                    comp_audio = apply_clip_audio_effects(comp_audio, master_effs)
                final = final.set_audio(comp_audio)

            if self._cancel_flag.is_set():
                return

            clean_codec = self._codec.split()[0] if self._codec else 'libx264'
            audio_codec = 'libvorbis' if clean_codec == 'libvpx-vp9' else 'aac'

            if 'nvenc' in clean_codec:
                preset = 'p4'
            elif 'qsv' in clean_codec:
                preset = 'veryfast'
            elif 'amf' in clean_codec:
                preset = 'speed'
            else:
                preset = 'fast'

            kwargs = {
                'logger':       logger,
                'codec':        clean_codec,
                'audio_codec':  audio_codec,
                'threads':      max(2, (os.cpu_count() or 4) - 2),
                'preset':       preset,
                'fps':          self._fps,
            }
            if self._bitrate:
                kwargs['bitrate'] = self._bitrate

            final.write_videofile(self._out, **kwargs)

            if not self._cancel_flag.is_set():
                self._success()

        except InterruptedError:
            log.info("RenderTask cancelled.")
        except Exception as exc:
            log.exception("RenderTask failed")
            if not self._cancel_flag.is_set():
                self._error(str(exc))
        finally:
            for c in open_clips:
                try:
                    c.close()
                except Exception:
                    pass
            free_memory()


class SingleClipExportTask(VideoTask):
    """Exports one clip as a transparent GIF or PNG frame sequence."""

    def __init__(self, c_data, out_path, is_gif,
                 on_success, on_error, on_progress):
        super().__init__(on_success, on_error, on_progress)
        self._c_data = copy.deepcopy(c_data)
        self._out    = out_path
        self._is_gif = is_gif

    def run(self):
        clip = None
        try:
            import imageio
            clip       = process_clip_data(self._c_data, self._cancel_flag)
            fps_val    = int(self._c_data.get('exp_fps', clip.fps or 15))
            total      = max(1, int(clip.duration * fps_val))

            if self._is_gif:
                frames = []
                for i, frame in enumerate(clip.iter_frames(fps=fps_val, dtype='uint8')):
                    if self._cancel_flag.is_set():
                        raise InterruptedError()
                    if self._c_data.get('rm_bg') and clip.mask:
                        mask = (clip.mask.get_frame(i / fps_val) * 255).astype('uint8')
                        rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                        rgba[:, :, 3] = mask
                        frames.append(rgba)
                    else:
                        frames.append(frame)
                    self._progress((i / total) * 100)
                imageio.mimsave(self._out, frames, format='GIF',
                                fps=fps_val, loop=0)
            else:
                for i, frame in enumerate(clip.iter_frames(fps=fps_val, dtype='uint8')):
                    if self._cancel_flag.is_set():
                        raise InterruptedError()
                    if self._c_data.get('rm_bg') and clip.mask:
                        mask = (clip.mask.get_frame(i / fps_val) * 255).astype('uint8')
                        rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                        rgba[:, :, 3] = mask
                        cv2.imwrite(
                            os.path.join(self._out, f"frame_{i:06d}.png"),
                            cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                        )
                    else:
                        cv2.imwrite(
                            os.path.join(self._out, f"frame_{i:06d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        )
                    self._progress((i / total) * 100)

            if not self._cancel_flag.is_set():
                self._success()

        except InterruptedError:
            log.info("SingleClipExportTask cancelled.")
        except Exception as exc:
            log.exception("SingleClipExportTask failed")
            if not self._cancel_flag.is_set():
                self._error(str(exc))
        finally:
            if clip:
                try:
                    clip.close()
                except Exception:
                    pass
            free_memory()
