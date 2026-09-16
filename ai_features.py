# ai_features.py
"""
AI-powered feature classes for Video Studio.
All heavy imports are done lazily inside methods with user-friendly error
messages when the required packages are missing.
"""

import os
import uuid


# ---------------------------------------------------------------------------
# WhisperSubtitler
# ---------------------------------------------------------------------------

class WhisperSubtitler:
    """
    Transcribes audio from a video/audio clip using openai-whisper
    and generates subtitle clips on a title track.
    """

    @staticmethod
    def is_available() -> bool:
        """Returns True if openai-whisper is importable."""
        try:
            import importlib.util
            return importlib.util.find_spec("whisper") is not None
        except Exception:
            return False

    @staticmethod
    def transcribe(media_path: str, model_size: str = "base",
                   language: str = None) -> list:
        """
        Runs Whisper on the file. Returns list of segments:
        [{'start': float, 'end': float, 'text': str}, ...]
        Raises ImportError with install instruction if whisper not installed.
        """
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. "
                "Install it with: pip install openai-whisper"
            )

        model = whisper.load_model(model_size)
        options = {}
        if language:
            options["language"] = language

        result = model.transcribe(media_path, **options)
        segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            for seg in result.get("segments", [])
        ]
        return segments

    @staticmethod
    def segments_to_title_clips(segments: list, track: str = "T1",
                                font_size: int = 48,
                                color: tuple = (255, 255, 255)) -> list:
        """
        Converts Whisper segments to title clip dicts ready to append
        to project_clips. Each segment becomes one title clip.
        """
        clips = []
        for seg in segments:
            start = float(seg["start"])
            end = float(seg["end"])
            duration = max(0.1, end - start)
            clip = {
                "id": str(uuid.uuid4()),
                "path": "__title__",
                "track": track,
                "start": start,
                "trim_in": 0.0,
                "trim_out": duration,
                "scale": 1.0,
                "x": "center",
                "y": "center",
                "volume": 1.0,
                "rm_bg": False,
                # Title-specific metadata
                "title_text": seg["text"],
                "title_font_size": font_size,
                "title_color": list(color),
                "title_animation": "Fade In/Out",
                "title_alignment": "center",
                "title_bg_color": [0, 0, 0, 0],
            }
            clips.append(clip)
        return clips


# ---------------------------------------------------------------------------
# SceneDetector
# ---------------------------------------------------------------------------

class SceneDetector:
    """
    Detects scene changes in a video file using PySceneDetect.
    """

    @staticmethod
    def is_available() -> bool:
        """Returns True if scenedetect is importable."""
        try:
            import importlib.util
            return importlib.util.find_spec("scenedetect") is not None
        except Exception:
            return False

    @staticmethod
    def detect_scenes(video_path: str, threshold: float = 27.0) -> list:
        """
        Returns list of scene-change timestamps in seconds.
        Uses ContentDetector from PySceneDetect.
        Raises ImportError with install instruction if not installed.
        """
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError:
            raise ImportError(
                "PySceneDetect is not installed. "
                "Install it with: pip install scenedetect[opencv]"
            )

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)

        scene_list = scene_manager.get_scene_list()
        # Each entry is (start_timecode, end_timecode); collect start of each scene > 0
        timestamps = []
        for i, (start_tc, _end_tc) in enumerate(scene_list):
            t = start_tc.get_seconds()
            if i > 0 and t > 0.0:
                timestamps.append(t)
        return timestamps

    @staticmethod
    def auto_split_clip(clip: dict, scene_times: list) -> list:
        """
        Given a clip dict and scene timestamps, returns a list of new clip dicts
        (trimmed sub-clips) split at each scene boundary.
        Each split clip gets a new uuid.
        """
        clip_start = clip["start"]
        trim_in = clip["trim_in"]
        trim_out = clip["trim_out"]
        clip_duration = trim_out - trim_in

        # Convert absolute scene timestamps to trim-space cut points
        cut_points = []
        for t in sorted(scene_times):
            # t is relative to the start of the file; map to position in clip
            offset = t - clip_start
            if 0.0 < offset < clip_duration:
                cut_points.append(trim_in + offset)

        # Build boundary list
        boundaries = [trim_in] + cut_points + [trim_out]

        sub_clips = []
        current_start = clip_start
        for i in range(len(boundaries) - 1):
            seg_trim_in = boundaries[i]
            seg_trim_out = boundaries[i + 1]
            if seg_trim_out - seg_trim_in < 0.05:
                continue
            new_clip = dict(clip)
            new_clip["id"] = str(uuid.uuid4())
            new_clip["start"] = current_start
            new_clip["trim_in"] = seg_trim_in
            new_clip["trim_out"] = seg_trim_out
            sub_clips.append(new_clip)
            current_start += seg_trim_out - seg_trim_in

        return sub_clips if sub_clips else [clip]


# ---------------------------------------------------------------------------
# BeatDetector
# ---------------------------------------------------------------------------

class BeatDetector:
    """
    Detects beat onsets in an audio/video file using librosa.
    """

    @staticmethod
    def is_available() -> bool:
        """Returns True if librosa is importable."""
        try:
            import importlib.util
            return importlib.util.find_spec("librosa") is not None
        except Exception:
            return False

    @staticmethod
    def detect_beats(media_path: str) -> list:
        """
        Returns list of beat timestamps in seconds using librosa.beat.beat_track.
        Raises ImportError with install instruction if not installed.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is not installed. "
                "Install it with: pip install librosa"
            )

        y, sr = librosa.load(media_path, mono=True)
        _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        return beat_times


# ---------------------------------------------------------------------------
# SmartReframer
# ---------------------------------------------------------------------------

class SmartReframer:
    """
    Auto-crops a 16:9 clip to 9:16 (or any target ratio) by tracking
    the dominant face/subject using MediaPipe face detection.
    """

    @staticmethod
    def is_available() -> bool:
        """Returns True if mediapipe and cv2 are importable."""
        try:
            import importlib.util
            return (
                importlib.util.find_spec("mediapipe") is not None
                and importlib.util.find_spec("cv2") is not None
            )
        except Exception:
            return False

    @staticmethod
    def compute_crop_path(video_path: str,
                          target_w: int, target_h: int,
                          sample_every_n_frames: int = 5) -> list:
        """
        Samples every N frames, detects face/subject bounding box using mediapipe,
        returns a smoothed list of crop parameters:
        [{'frame': int, 'cx': float, 'cy': float}, ...]
        cx, cy are normalized (0.0-1.0) centre coordinates of the subject.
        Uses exponential moving average for smooth camera motion.
        """
        try:
            import mediapipe as mp
            import cv2
        except ImportError:
            raise ImportError(
                "mediapipe is not installed. "
                "Install it with: pip install mediapipe opencv-python"
            )

        mp_face = mp.solutions.face_detection
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        crop_path = []
        ema_cx, ema_cy = 0.5, 0.5
        alpha = 0.15  # EMA smoothing factor — lower = smoother

        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as detector:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_every_n_frames == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = detector.process(rgb)

                    if results.detections:
                        det = results.detections[0]
                        bbox = det.location_data.relative_bounding_box
                        face_cx = bbox.xmin + bbox.width / 2.0
                        face_cy = bbox.ymin + bbox.height / 2.0
                        # Clamp to valid range
                        face_cx = max(0.0, min(1.0, face_cx))
                        face_cy = max(0.0, min(1.0, face_cy))
                        ema_cx = alpha * face_cx + (1 - alpha) * ema_cx
                        ema_cy = alpha * face_cy + (1 - alpha) * ema_cy

                    crop_path.append({"frame": frame_idx, "cx": ema_cx, "cy": ema_cy})
                frame_idx += 1

        cap.release()
        return crop_path

    @staticmethod
    def apply_crop_path(video_path: str, crop_path: list,
                        target_w: int, target_h: int,
                        output_path: str, fps: float = 30.0):
        """
        Renders the reframed video by applying the crop path frame by frame
        using cv2. Writes output_path using cv2.VideoWriter.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is not installed. "
                "Install it with: pip install opencv-python"
            )

        cap = cv2.VideoCapture(video_path)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (target_w, target_h))

        # Build a frame-index → (cx, cy) lookup from the sampled crop_path
        lookup = {entry["frame"]: (entry["cx"], entry["cy"]) for entry in crop_path}

        # Fill gaps by forward-filling
        last_cx, last_cy = 0.5, 0.5
        frame_idx = 0
        max_frame = max(lookup.keys()) if lookup else 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in lookup:
                last_cx, last_cy = lookup[frame_idx]

            cx_px = int(last_cx * src_w)
            cy_px = int(last_cy * src_h)

            x1 = max(0, cx_px - target_w // 2)
            y1 = max(0, cy_px - target_h // 2)
            x2 = x1 + target_w
            y2 = y1 + target_h

            # Clamp to source bounds
            if x2 > src_w:
                x2 = src_w
                x1 = max(0, src_w - target_w)
            if y2 > src_h:
                y2 = src_h
                y1 = max(0, src_h - target_h)

            cropped = frame[y1:y2, x1:x2]
            if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
                cropped = cv2.resize(cropped, (target_w, target_h))

            writer.write(cropped)
            frame_idx += 1

        cap.release()
        writer.release()


# ---------------------------------------------------------------------------
# NoiseReducer
# ---------------------------------------------------------------------------

class NoiseReducer:
    """
    Applies AI noise reduction to audio using noisereduce library.
    """

    @staticmethod
    def is_available() -> bool:
        """Returns True if noisereduce is importable."""
        try:
            import importlib.util
            return importlib.util.find_spec("noisereduce") is not None
        except Exception:
            return False

    @staticmethod
    def reduce_noise(input_path: str, output_path: str,
                     prop_decrease: float = 0.8) -> str:
        """
        Reads audio from input_path (via soundfile/librosa),
        applies noisereduce.reduce_noise(), writes to output_path as WAV.
        Returns output_path.
        Raises ImportError with install instruction if not installed.
        """
        try:
            import noisereduce as nr
        except ImportError:
            raise ImportError(
                "noisereduce is not installed. "
                "Install it with: pip install noisereduce"
            )

        # Load audio — prefer soundfile, fall back to librosa
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(input_path, always_2d=False)
        except ImportError:
            try:
                import librosa
                audio, sample_rate = librosa.load(input_path, sr=None, mono=False)
            except ImportError:
                raise ImportError(
                    "soundfile or librosa is required for audio loading. "
                    "Install with: pip install soundfile  or  pip install librosa"
                )

        import numpy as np
        reduced = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=prop_decrease)

        try:
            import soundfile as sf
            sf.write(output_path, reduced, sample_rate)
        except ImportError:
            import scipy.io.wavfile as wav
            # scipy expects int16 if float audio
            if reduced.dtype.kind == "f":
                reduced_int = (reduced * 32767).astype(np.int16)
            else:
                reduced_int = reduced
            wav.write(output_path, sample_rate, reduced_int)

        return output_path


# ---------------------------------------------------------------------------
# AudioDucker
# ---------------------------------------------------------------------------

class AudioDucker:
    """
    Automatic audio ducking for dialogues and background music.
    Detects dialogue presence and applies a smooth attenuation envelope
    to the target background music / audio track.
    """

    @staticmethod
    def compute_ducking_envelope(voice_samples, sample_rate: int = 44100,
                                  threshold_db: float = -28.0,
                                  duck_db: float = -12.0,
                                  attack_sec: float = 0.15,
                                  release_sec: float = 0.5):
        """
        Calculates a smooth volume multiplier array [0..1] aligned with voice_samples.
        """
        import numpy as np

        if voice_samples.ndim > 1:
            voice_mono = np.mean(voice_samples, axis=1)
        else:
            voice_mono = np.array(voice_samples, dtype=np.float32)

        if voice_mono.dtype.kind == 'i':
            voice_mono = voice_mono / 32768.0

        eps = 1e-9
        frame_len = max(1, int(sample_rate * 0.05))  # 50ms window
        remainder = len(voice_mono) % frame_len
        pad_len = (frame_len - remainder) if remainder != 0 else 0
        padded = np.pad(voice_mono, (0, pad_len))
        frames = padded.reshape(-1, frame_len)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + eps)
        rms_db = 20 * np.log10(np.maximum(rms, eps))

        duck_multiplier = 10.0 ** (duck_db / 20.0)
        target_gains = np.where(rms_db > threshold_db, duck_multiplier, 1.0)

        attack_alpha = float(np.exp(-1.0 / max(1.0, attack_sec / 0.05)))
        release_alpha = float(np.exp(-1.0 / max(1.0, release_sec / 0.05)))

        envelope_frames = np.ones(len(target_gains), dtype=np.float32)
        curr_gain = 1.0
        for i, target in enumerate(target_gains):
            if target < curr_gain:
                curr_gain = attack_alpha * curr_gain + (1 - attack_alpha) * target
            else:
                curr_gain = release_alpha * curr_gain + (1 - release_alpha) * target
            envelope_frames[i] = curr_gain

        sample_indices = np.linspace(0, len(envelope_frames) - 1, len(voice_samples))
        full_envelope = np.interp(sample_indices, np.arange(len(envelope_frames)), envelope_frames)
        return full_envelope.astype(np.float32)

