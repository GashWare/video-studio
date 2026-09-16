# clip_data.py
"""
Typed schema for NLE clip data.
Uses TypedDict so existing dict-access patterns work unchanged.
ClipData instances ARE regular dicts — no migration needed.
"""
from typing import TypedDict, Optional, Any
import uuid


class ColorGradingDict(TypedDict):
    brightness: float   # -100 to +100
    contrast:   float   # -100 to +100
    saturation: float   # -100 to +100
    hue:        float   # -180 to +180
    lift:       list    # [R, G, B] each 0.0-2.0
    gamma:      list    # [R, G, B] each 0.0-2.0
    gain:       list    # [R, G, B] each 0.0-2.0
    lut_preset: str     # "None" | "Cinematic" | ...


class ClipData(TypedDict, total=False):
    # Identity
    id:           str
    path:         str          # file path or "__title__"
    track:        str          # "V1" | "V2" | "A1" | "A2" | "T1"

    # Timeline position
    start:        float        # start time in sequence (seconds)
    trim_in:      float        # in-point in source (seconds)
    trim_out:     float        # out-point in source (seconds)

    # Transform
    scale:        float        # 1.0 = original size
    x:            Any          # int pixels or "center"/"left"/"right"
    y:            Any          # int pixels or "center"/"top"/"bottom"
    aspect:       str          # "Original" | "16:9" | "9:16" | "1:1" | "4:3"
    rot:          str          # "0°" | "90°" | "180°" | "270°"

    # Audio
    volume:       float        # 0.0-2.0
    speed:        float        # 0.25-4.0
    mute_audio:   bool
    audio_effects: list        # list of AudioEffect dicts

    # Effects
    rm_bg:        bool
    custom_masks: dict         # {frame_idx: np.ndarray}
    wm_text:      str
    wm_pos:       str
    blend_mode:   str          # "Normal" | "Multiply" | "Screen" | ...
    opacity:      float        # 0.0 - 1.0
    color:        ColorGradingDict

    # Transition (into this clip from previous)
    transition_type: str       # "None" | "Cross Dissolve" | ...
    transition_dur:  float     # seconds

    # Metadata
    label:        Optional[str]  # colour label name
    notes:        str
    title_text:   str            # for T1 title clips
    title_config: dict           # full TitleClip config dict

    # Keyframes
    keyframes:    dict           # {prop_name: {t: value, easing: str}}

    # Linking (for audio/video pairs)
    linked_clip_id: Optional[str]  # ID of linked companion clip


DEFAULT_COLOR_GRADING: ColorGradingDict = {
    "brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0,
    "lift": [1.0, 1.0, 1.0], "gamma": [1.0, 1.0, 1.0], "gain": [1.0, 1.0, 1.0],
    "lut_preset": "None",
}

BLEND_MODES = [
    "Normal", "Multiply", "Screen", "Overlay", "Soft Light",
    "Hard Light", "Color Dodge", "Color Burn", "Darken",
    "Lighten", "Difference", "Exclusion", "Luminosity", "Add",
]

# Valid track identifiers (base set; any V#, A#, T#, S#, I# is valid)
_VALID_TRACKS = {"V1", "V2", "A1", "A2", "T1", "S1", "I1"}


def make_clip(path: str, track: str, start: float, duration: float, **overrides) -> ClipData:
    """
    Factory: create a new ClipData dict with all required defaults set.
    All fields have sane defaults so callers only specify what differs.

    Args:
        path:      File path on disk, or ``"__title__"`` for title clips.
        track:     Timeline track identifier, e.g. ``"V1"`` or ``"A1"``.
        start:     Clip start position in the sequence, in seconds.
        duration:  Source duration in seconds (sets ``trim_out``).
        **overrides: Any ClipData field to override after defaults are applied.

    Returns:
        A fully-populated :class:`ClipData` dict.
    """
    clip: ClipData = {
        'id':              str(uuid.uuid4()),
        'path':            path,
        'track':           track,
        'start':           start,
        'trim_in':         0.0,
        'trim_out':        duration,
        'scale':           1.0,
        'x':               'center',
        'y':               'center',
        'aspect':          'Original',
        'rot':             '0°',
        'volume':          1.0,
        'speed':           1.0,
        'mute_audio':      False,
        'audio_effects':   [],
        'rm_bg':           False,
        'custom_masks':    {},
        'wm_text':         '',
        'wm_pos':          'Bottom Right',
        'blend_mode':      'Normal',
        'opacity':         1.0,
        'color':           dict(DEFAULT_COLOR_GRADING),
        'transition_type': 'None',
        'transition_dur':  0.5,
        'label':           None,
        'notes':           '',
        'keyframes':       {},
        'linked_clip_id':  None,
    }
    clip.update(overrides)
    return clip


def get_color(clip: ClipData) -> ColorGradingDict:
    """Return the colour grading dict, filling defaults for any missing keys.

    Always returns a complete :class:`ColorGradingDict` — if the clip's
    ``color`` field is absent or partially populated, the missing keys are
    filled from :data:`DEFAULT_COLOR_GRADING`.
    """
    base = dict(DEFAULT_COLOR_GRADING)
    base.update(clip.get('color', {}))
    return base


def clip_duration(clip: ClipData) -> float:
    """Return the effective clip duration in seconds (``trim_out - trim_in``)."""
    return clip['trim_out'] - clip['trim_in']


def clip_end_time(clip: ClipData) -> float:
    """Return the sequence time at which this clip ends, in seconds."""
    return clip['start'] + clip_duration(clip)


def validate_clip(clip: dict) -> list:
    """Validate a clip dict and return a list of error strings.

    An empty return value means the clip is valid.

    Checks performed:
    - ``id``       — must be a non-empty string.
    - ``track``    — must be a valid track identifier (V#, A#, T#).
    - ``trim_in``  — must be >= 0.
    - ``trim_out`` — must be > ``trim_in``.
    - ``start``    — must be >= 0.
    - ``scale``    — must be > 0.
    - ``volume``   — must be >= 0.

    Args:
        clip: Any dict (typically a :class:`ClipData`) to validate.

    Returns:
        A list of human-readable error strings; empty when the clip is valid.
    """
    errors: list = []

    # id — non-empty string
    clip_id = clip.get('id')
    if not isinstance(clip_id, str) or not clip_id.strip():
        errors.append("'id' must be a non-empty string.")

    # track — must be a known or dynamic track identifier (e.g. V1, V2, A1, A2, S1, I1, T1...)
    track = clip.get('track')
    is_valid_track = (
        isinstance(track, str)
        and len(track) >= 2
        and track[0] in ('V', 'A', 'T', 'S', 'I')
        and track[1:].isdigit()
    )
    if not is_valid_track and track not in _VALID_TRACKS:
        errors.append(
            f"'track' must be a valid track identifier (e.g. 'V1', 'A1'); got {track!r}."
        )

    # trim_in — must be >= 0
    trim_in = clip.get('trim_in')
    if not isinstance(trim_in, (int, float)) or trim_in < 0:
        errors.append(f"'trim_in' must be >= 0; got {trim_in!r}.")

    # trim_out — must be > trim_in (only when both are valid numbers)
    trim_out = clip.get('trim_out')
    if not isinstance(trim_out, (int, float)):
        errors.append(f"'trim_out' must be a number; got {trim_out!r}.")
    elif isinstance(trim_in, (int, float)) and trim_out <= trim_in:
        errors.append(
            f"'trim_out' ({trim_out}) must be greater than 'trim_in' ({trim_in})."
        )

    # start — must be >= 0
    start = clip.get('start')
    if not isinstance(start, (int, float)) or start < 0:
        errors.append(f"'start' must be >= 0; got {start!r}.")

    # scale — must be > 0
    scale = clip.get('scale')
    if not isinstance(scale, (int, float)) or scale <= 0:
        errors.append(f"'scale' must be > 0; got {scale!r}.")

    # volume — must be >= 0
    volume = clip.get('volume')
    if not isinstance(volume, (int, float)) or volume < 0:
        errors.append(f"'volume' must be >= 0; got {volume!r}.")

    return errors
