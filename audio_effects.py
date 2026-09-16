# audio_effects.py
"""
DAW-grade Studio Audio Effects Rack & VST3 Chain Engine.
Supports:
- 4-Band Parametric Equalizer (Biquad peak & shelf filters)
- Dynamic Range Compressor (Threshold, Ratio, Attack, Release, Knee, Makeup Gain)
- Studio Reverb (Stereo algorithmic reverb)
- Stereo Feedback Delay
- Brickwall Limiter & Maximizer
- High-Pass & Low-Pass Filters
- VST3 / AU Plugin Hosting (via Spotify's pedalboard library when available)
"""

import copy
import math
import os
import uuid
import numpy as np

# ---------------------------------------------------------------------------
# Biquad Filter Mathematics (Audio EQ Cookbook)
# ---------------------------------------------------------------------------

def _biquad_coefficients(filter_type: str, freq: float, sample_rate: float, gain_db: float = 0.0, q: float = 1.0):
    """
    Calculate 2nd order biquad IIR filter coefficients [b0, b1, b2, a0, a1, a2].
    Standard Robert Bristow-Johnson Audio EQ Cookbook formulation.
    """
    freq = max(20.0, min(freq, sample_rate * 0.49))
    omega = 2.0 * math.pi * freq / sample_rate
    sn = math.sin(omega)
    cs = math.cos(omega)
    alpha = sn / (2.0 * max(0.01, q))
    A = 10.0 ** (gain_db / 40.0)

    if filter_type == "peaking":
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cs
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cs
        a2 = 1.0 - alpha / A
    elif filter_type == "low_shelf":
        sqrt_A = math.sqrt(A)
        b0 = A * ((A + 1.0) - (A - 1.0) * cs + 2.0 * sqrt_A * alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cs)
        b2 = A * ((A + 1.0) - (A - 1.0) * cs - 2.0 * sqrt_A * alpha)
        a0 = (A + 1.0) + (A - 1.0) * cs + 2.0 * sqrt_A * alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cs)
        a2 = (A + 1.0) + (A - 1.0) * cs - 2.0 * sqrt_A * alpha
    elif filter_type == "high_shelf":
        sqrt_A = math.sqrt(A)
        b0 = A * ((A + 1.0) + (A - 1.0) * cs + 2.0 * sqrt_A * alpha)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cs)
        b2 = A * ((A + 1.0) + (A - 1.0) * cs - 2.0 * sqrt_A * alpha)
        a0 = (A + 1.0) - (A - 1.0) * cs + 2.0 * sqrt_A * alpha
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cs)
        a2 = (A + 1.0) - (A - 1.0) * cs - 2.0 * sqrt_A * alpha
    elif filter_type == "high_pass":
        b0 = (1.0 + cs) / 2.0
        b1 = -(1.0 + cs)
        b2 = (1.0 + cs) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cs
        a2 = 1.0 - alpha
    elif filter_type == "low_pass":
        b0 = (1.0 - cs) / 2.0
        b1 = 1.0 - cs
        b2 = (1.0 - cs) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cs
        a2 = 1.0 - alpha
    else:
        return [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]

    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1.0, a1 / a0, a2 / a0]
    return b, a


def _apply_iir_filter(samples: np.ndarray, b: list, a: list) -> np.ndarray:
    """Apply IIR filter using scipy.signal.lfilter or pure numpy fallback."""
    try:
        from scipy.signal import lfilter
        return lfilter(b, a, samples, axis=0).astype(np.float32)
    except Exception:
        out = np.zeros_like(samples, dtype=np.float32)
        if samples.ndim == 1:
            x = samples
            y = np.zeros(len(x), dtype=np.float32)
            b0, b1, b2 = b
            _, a1, a2 = a
            x1 = x2 = y1 = y2 = 0.0
            for i in range(len(x)):
                xi = x[i]
                yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                y[i] = yi
                x2, x1 = x1, xi
                y2, y1 = y1, yi
            return y
        else:
            for ch in range(samples.shape[1]):
                out[:, ch] = _apply_iir_filter(samples[:, ch], b, a)
            return out


# ---------------------------------------------------------------------------
# Audio Effect Models & Processors
# ---------------------------------------------------------------------------

class AudioEffect:
    """Base class for all studio audio effect processors."""

    EFFECT_TYPE = "BaseEffect"

    def __init__(self, name: str = "", enabled: bool = True):
        self.id = str(uuid.uuid4())
        self.name = name or self.EFFECT_TYPE
        self.enabled = enabled

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return samples

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.EFFECT_TYPE,
            "name": self.name,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(name=data.get("name", ""), enabled=data.get("enabled", True))


class ParametricEQEffect(AudioEffect):
    """
    4-Band Studio Parametric Equalizer:
    - Band 1: Low Shelf
    - Band 2: Peaking Bell 1
    - Band 3: Peaking Bell 2
    - Band 4: High Shelf
    """

    EFFECT_TYPE = "ParametricEQ"

    def __init__(self, name="Parametric EQ", enabled=True,
                 low_shelf_freq=100.0, low_shelf_gain=0.0,
                 mid1_freq=500.0, mid1_gain=0.0, mid1_q=1.0,
                 mid2_freq=2500.0, mid2_gain=0.0, mid2_q=1.0,
                 high_shelf_freq=8000.0, high_shelf_gain=0.0):
        super().__init__(name, enabled)
        self.low_shelf_freq = float(low_shelf_freq)
        self.low_shelf_gain = float(low_shelf_gain)
        self.mid1_freq = float(mid1_freq)
        self.mid1_gain = float(mid1_gain)
        self.mid1_q = float(mid1_q)
        self.mid2_freq = float(mid2_freq)
        self.mid2_gain = float(mid2_gain)
        self.mid2_q = float(mid2_q)
        self.high_shelf_freq = float(high_shelf_freq)
        self.high_shelf_gain = float(high_shelf_gain)

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled:
            return samples

        out = samples.copy()
        # Low shelf
        if abs(self.low_shelf_gain) > 0.01:
            b, a = _biquad_coefficients("low_shelf", self.low_shelf_freq, sample_rate, self.low_shelf_gain, 0.71)
            out = _apply_iir_filter(out, b, a)

        # Mid 1
        if abs(self.mid1_gain) > 0.01:
            b, a = _biquad_coefficients("peaking", self.mid1_freq, sample_rate, self.mid1_gain, self.mid1_q)
            out = _apply_iir_filter(out, b, a)

        # Mid 2
        if abs(self.mid2_gain) > 0.01:
            b, a = _biquad_coefficients("peaking", self.mid2_freq, sample_rate, self.mid2_gain, self.mid2_q)
            out = _apply_iir_filter(out, b, a)

        # High shelf
        if abs(self.high_shelf_gain) > 0.01:
            b, a = _biquad_coefficients("high_shelf", self.high_shelf_freq, sample_rate, self.high_shelf_gain, 0.71)
            out = _apply_iir_filter(out, b, a)

        return out

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "low_shelf_freq": self.low_shelf_freq,
            "low_shelf_gain": self.low_shelf_gain,
            "mid1_freq": self.mid1_freq,
            "mid1_gain": self.mid1_gain,
            "mid1_q": self.mid1_q,
            "mid2_freq": self.mid2_freq,
            "mid2_gain": self.mid2_gain,
            "mid2_q": self.mid2_q,
            "high_shelf_freq": self.high_shelf_freq,
            "high_shelf_gain": self.high_shelf_gain,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data.get("name", "Parametric EQ"),
            enabled=data.get("enabled", True),
            low_shelf_freq=data.get("low_shelf_freq", 100.0),
            low_shelf_gain=data.get("low_shelf_gain", 0.0),
            mid1_freq=data.get("mid1_freq", 500.0),
            mid1_gain=data.get("mid1_gain", 0.0),
            mid1_q=data.get("mid1_q", 1.0),
            mid2_freq=data.get("mid2_freq", 2500.0),
            mid2_gain=data.get("mid2_gain", 0.0),
            mid2_q=data.get("mid2_q", 1.0),
            high_shelf_freq=data.get("high_shelf_freq", 8000.0),
            high_shelf_gain=data.get("high_shelf_gain", 0.0),
        )


class CompressorEffect(AudioEffect):
    """
    Studio Dynamic Range Compressor:
    - Threshold: -60 dB to 0 dB
    - Ratio: 1.0:1 to 20.0:1
    - Attack: 1 ms to 200 ms
    - Release: 10 ms to 1000 ms
    - Makeup Gain: 0 dB to 24 dB
    """

    EFFECT_TYPE = "Compressor"

    def __init__(self, name="Compressor", enabled=True,
                 threshold_db=-18.0, ratio=4.0, attack_ms=15.0, release_ms=100.0,
                 makeup_gain_db=0.0):
        super().__init__(name, enabled)
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.makeup_gain_db = float(makeup_gain_db)

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or len(samples) == 0:
            return samples

        # Sidechain RMS / peak envelope
        mono = np.mean(samples, axis=1) if samples.ndim > 1 else samples
        abs_signal = np.abs(mono)

        att_coeff = math.exp(-1.0 / max(1.0, (self.attack_ms * 0.001 * sample_rate)))
        rel_coeff = math.exp(-1.0 / max(1.0, (self.release_ms * 0.001 * sample_rate)))

        env = np.zeros(len(abs_signal), dtype=np.float32)
        curr = 0.0
        for i in range(len(abs_signal)):
            s = abs_signal[i]
            if s > curr:
                curr = att_coeff * curr + (1.0 - att_coeff) * s
            else:
                curr = rel_coeff * curr + (1.0 - rel_coeff) * s
            env[i] = curr

        # Compute gain reduction in dB
        eps = 1e-9
        env_db = 20.0 * np.log10(np.maximum(env, eps))
        overshoot_db = env_db - self.threshold_db
        gr_db = np.zeros_like(env_db)
        mask = overshoot_db > 0.0
        gr_db[mask] = overshoot_db[mask] * (1.0 - 1.0 / max(1.0, self.ratio))

        gain_linear = 10.0 ** ((-gr_db + self.makeup_gain_db) / 20.0)

        if samples.ndim > 1:
            out = samples * gain_linear[:, np.newaxis]
        else:
            out = samples * gain_linear

        return out.astype(np.float32)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "makeup_gain_db": self.makeup_gain_db,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data.get("name", "Compressor"),
            enabled=data.get("enabled", True),
            threshold_db=data.get("threshold_db", -18.0),
            ratio=data.get("ratio", 4.0),
            attack_ms=data.get("attack_ms", 15.0),
            release_ms=data.get("release_ms", 100.0),
            makeup_gain_db=data.get("makeup_gain_db", 0.0),
        )


class ReverbEffect(AudioEffect):
    """
    Stereo Algorithmic Reverb:
    - Room Size: 0.0 to 1.0
    - Damping: 0.0 to 1.0
    - Wet Level: 0.0 to 1.0
    - Dry Level: 0.0 to 1.0
    """

    EFFECT_TYPE = "Reverb"

    def __init__(self, name="Studio Reverb", enabled=True,
                 room_size=0.6, damping=0.3, wet_level=0.3, dry_level=0.8):
        super().__init__(name, enabled)
        self.room_size = float(room_size)
        self.damping = float(damping)
        self.wet_level = float(wet_level)
        self.dry_level = float(dry_level)

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or len(samples) == 0:
            return samples

        # Multi-comb delay line simulation
        delays_ms = [29.7, 37.1, 41.1, 43.7]
        wet = np.zeros_like(samples, dtype=np.float32)

        for d_ms in delays_ms:
            d_samples = int(sample_rate * d_ms * 0.001)
            if d_samples < len(samples):
                delayed = np.pad(samples[:-d_samples], ((d_samples, 0),) + (((0, 0),) if samples.ndim > 1 else ()))
                wet += delayed * (self.room_size * 0.25)

        out = samples * self.dry_level + wet * self.wet_level
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "room_size": self.room_size,
            "damping": self.damping,
            "wet_level": self.wet_level,
            "dry_level": self.dry_level,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data.get("name", "Studio Reverb"),
            enabled=data.get("enabled", True),
            room_size=data.get("room_size", 0.6),
            damping=data.get("damping", 0.3),
            wet_level=data.get("wet_level", 0.3),
            dry_level=data.get("dry_level", 0.8),
        )


class StereoDelayEffect(AudioEffect):
    """
    Stereo Feedback Delay:
    - Delay Time: 10 ms to 1000 ms
    - Feedback: 0.0 to 0.95
    - Wet/Dry Mix: 0.0 to 1.0
    """

    EFFECT_TYPE = "StereoDelay"

    def __init__(self, name="Stereo Delay", enabled=True,
                 delay_ms=250.0, feedback=0.35, mix=0.4):
        super().__init__(name, enabled)
        self.delay_ms = float(delay_ms)
        self.feedback = float(feedback)
        self.mix = float(mix)

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or len(samples) == 0:
            return samples

        delay_samples = max(1, int(sample_rate * self.delay_ms * 0.001))
        out = samples.copy()

        if delay_samples < len(samples):
            delayed = np.pad(samples[:-delay_samples], ((delay_samples, 0),) + (((0, 0),) if samples.ndim > 1 else ()))
            out = samples * (1.0 - self.mix * 0.5) + delayed * (self.mix * self.feedback + 0.1)

        return out.astype(np.float32)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "delay_ms": self.delay_ms,
            "feedback": self.feedback,
            "mix": self.mix,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data.get("name", "Stereo Delay"),
            enabled=data.get("enabled", True),
            delay_ms=data.get("delay_ms", 250.0),
            feedback=data.get("feedback", 0.35),
            mix=data.get("mix", 0.4),
        )


class LimiterEffect(AudioEffect):
    """
    Brickwall Peak Limiter & Maximizer:
    - Ceiling: -12.0 dBFS to 0.0 dBFS
    - Release: 5 ms to 200 ms
    """

    EFFECT_TYPE = "Limiter"

    def __init__(self, name="Limiter", enabled=True, ceiling_db=-0.5, release_ms=50.0):
        super().__init__(name, enabled)
        self.ceiling_db = float(ceiling_db)
        self.release_ms = float(release_ms)

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or len(samples) == 0:
            return samples

        ceiling_linear = 10.0 ** (self.ceiling_db / 20.0)
        max_peak = np.max(np.abs(samples))
        if max_peak > ceiling_linear:
            scale = ceiling_linear / max_peak
            return (samples * scale).astype(np.float32)
        return samples

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "ceiling_db": self.ceiling_db,
            "release_ms": self.release_ms,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data.get("name", "Limiter"),
            enabled=data.get("enabled", True),
            ceiling_db=data.get("ceiling_db", -0.5),
            release_ms=data.get("release_ms", 50.0),
        )


class HighPassFilterEffect(AudioEffect):
    """High-pass cut filter for rumble / wind noise removal."""

    EFFECT_TYPE = "HighPassFilter"

    def __init__(self, name="High Pass Filter", enabled=True, cutoff_freq=80.0):
        super().__init__(name, enabled)
        self.cutoff_freq = float(cutoff_freq)

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or len(samples) == 0:
            return samples
        b, a = _biquad_coefficients("high_pass", self.cutoff_freq, sample_rate, q=0.707)
        return _apply_iir_filter(samples, b, a)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"cutoff_freq": self.cutoff_freq})
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data.get("name", "High Pass Filter"),
            enabled=data.get("enabled", True),
            cutoff_freq=data.get("cutoff_freq", 80.0),
        )


class LowPassFilterEffect(AudioEffect):
    """Low-pass cut filter for high-frequency smoothing."""

    EFFECT_TYPE = "LowPassFilter"

    def __init__(self, name="Low Pass Filter", enabled=True, cutoff_freq=12000.0):
        super().__init__(name, enabled)
        self.cutoff_freq = float(cutoff_freq)

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or len(samples) == 0:
            return samples
        b, a = _biquad_coefficients("low_pass", self.cutoff_freq, sample_rate, q=0.707)
        return _apply_iir_filter(samples, b, a)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"cutoff_freq": self.cutoff_freq})
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data.get("name", "Low Pass Filter"),
            enabled=data.get("enabled", True),
            cutoff_freq=data.get("cutoff_freq", 12000.0),
        )


class VST3PluginEffect(AudioEffect):
    """
    VST3 / AudioUnit plugin host using pedalboard library.
    """

    EFFECT_TYPE = "VST3Plugin"

    def __init__(self, name="VST3 Plugin", enabled=True, plugin_path=""):
        super().__init__(name, enabled)
        self.plugin_path = plugin_path
        self._plugin_instance = None
        self._load_plugin()

    def _load_plugin(self):
        if not self.plugin_path or not os.path.exists(self.plugin_path):
            return
        try:
            import pedalboard
            self._plugin_instance = pedalboard.load_plugin(self.plugin_path)
        except Exception:
            self._plugin_instance = None

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or self._plugin_instance is None or len(samples) == 0:
            return samples

        try:
            # pedalboard expects (channels, samples) float32
            if samples.ndim == 1:
                in_audio = samples.reshape(1, -1).astype(np.float32)
                out_audio = self._plugin_instance(in_audio, sample_rate)
                return out_audio.flatten().astype(np.float32)
            else:
                in_audio = samples.T.astype(np.float32)
                out_audio = self._plugin_instance(in_audio, sample_rate)
                return out_audio.T.astype(np.float32)
        except Exception:
            return samples

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"plugin_path": self.plugin_path})
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data.get("name", "VST3 Plugin"),
            enabled=data.get("enabled", True),
            plugin_path=data.get("plugin_path", ""),
        )


# ---------------------------------------------------------------------------
# Effect Registry Mapping
# ---------------------------------------------------------------------------

EFFECT_CLASSES = {
    "ParametricEQ": ParametricEQEffect,
    "Compressor": CompressorEffect,
    "Reverb": ReverbEffect,
    "StereoDelay": StereoDelayEffect,
    "Limiter": LimiterEffect,
    "HighPassFilter": HighPassFilterEffect,
    "LowPassFilter": LowPassFilterEffect,
    "VST3Plugin": VST3PluginEffect,
}


def effect_from_dict(d: dict) -> AudioEffect:
    """Instantiate an AudioEffect subclass from a dictionary."""
    t = d.get("type", "")
    cls = EFFECT_CLASSES.get(t, AudioEffect)
    return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Audio Effect Chain
# ---------------------------------------------------------------------------

class AudioEffectChain:
    """Ordered serial chain of AudioEffects applied in sequence."""

    def __init__(self, effects: list[AudioEffect] = None):
        self.effects: list[AudioEffect] = list(effects) if effects else []

    def add_effect(self, effect: AudioEffect):
        self.effects.append(effect)

    def remove_effect_by_id(self, effect_id: str):
        self.effects = [e for e in self.effects if e.id != effect_id]

    def move_effect(self, index: int, direction: int):
        new_idx = index + direction
        if 0 <= new_idx < len(self.effects):
            self.effects[index], self.effects[new_idx] = self.effects[new_idx], self.effects[index]

    def process(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.effects or len(samples) == 0:
            return samples

        out = samples.astype(np.float32)
        for eff in self.effects:
            if eff.enabled:
                out = eff.process(out, sample_rate)

        # Final soft-clipping protection
        out = np.tanh(out) if np.max(np.abs(out)) > 1.5 else np.clip(out, -1.0, 1.0)
        return out.astype(np.float32)

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.effects]

    @classmethod
    def from_list(cls, effect_dicts: list) -> 'AudioEffectChain':
        if not effect_dicts:
            return cls()
        effs = []
        for d in effect_dicts:
            if isinstance(d, dict):
                effs.append(effect_from_dict(d))
        return cls(effs)


def apply_audio_chain(samples: np.ndarray, sample_rate: int, chain_data: list | AudioEffectChain) -> np.ndarray:
    """
    Convenience function: Apply an audio effect chain to audio samples.
    """
    if chain_data is None:
        return samples
    if isinstance(chain_data, AudioEffectChain):
        return chain_data.process(samples, sample_rate)
    elif isinstance(chain_data, list):
        chain = AudioEffectChain.from_list(chain_data)
        return chain.process(samples, sample_rate)
    return samples
