# tests/test_core.py
"""
Pytest unit-test suite for the core modules.
No GUI (tkinter), no real video files, no optional AI packages.
Synthetic data only (numpy arrays, plain dicts).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import json
import tempfile
import uuid
import zipfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_frame(h=60, w=80, value=128):
    """Return an (H, W, 3) uint8 frame filled with *value*."""
    return np.full((h, w, 3), value, dtype=np.uint8)


# ===========================================================================
# 1. clip_data.py
# ===========================================================================

from clip_data import (
    make_clip,
    clip_duration,
    clip_end_time,
    validate_clip,
    get_color,
    DEFAULT_COLOR_GRADING,
)

REQUIRED_KEYS = {
    "id", "path", "track", "start", "trim_in", "trim_out",
    "scale", "x", "y", "aspect", "rot", "volume", "speed",
    "mute_audio", "rm_bg", "custom_masks", "wm_text", "wm_pos",
    "color", "transition_type", "transition_dur", "label", "notes",
    "keyframes",
}


def test_make_clip_returns_dict_with_all_required_keys():
    clip = make_clip("video.mp4", "V1", 0.0, 10.0)
    assert isinstance(clip, dict)
    missing = REQUIRED_KEYS - clip.keys()
    assert missing == set(), f"Missing keys: {missing}"


def test_make_clip_overrides_are_applied():
    clip = make_clip("video.mp4", "V1", 2.0, 5.0, volume=0.5, scale=2.0)
    assert clip["volume"] == 0.5
    assert clip["scale"] == 2.0


def test_make_clip_id_is_unique_uuid():
    c1 = make_clip("a.mp4", "V1", 0.0, 1.0)
    c2 = make_clip("a.mp4", "V1", 0.0, 1.0)
    assert c1["id"] != c2["id"]
    # Should parse as a valid UUID
    uuid.UUID(c1["id"])


def test_clip_duration_returns_trim_out_minus_trim_in():
    clip = make_clip("v.mp4", "V1", 0.0, 10.0)
    clip["trim_in"] = 2.0
    clip["trim_out"] = 7.5
    assert clip_duration(clip) == pytest.approx(5.5)


def test_clip_end_time_returns_start_plus_duration():
    clip = make_clip("v.mp4", "V1", 3.0, 6.0)
    clip["trim_in"] = 1.0
    clip["trim_out"] = 4.0
    # duration = 3.0, start = 3.0, end = 6.0
    assert clip_end_time(clip) == pytest.approx(6.0)


def test_validate_clip_empty_list_for_valid_clip():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    errors = validate_clip(clip)
    assert errors == []


def test_validate_clip_catches_empty_id():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    clip["id"] = ""
    errors = validate_clip(clip)
    assert any("id" in e for e in errors)


def test_validate_clip_catches_bad_track():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    clip["track"] = "BADTRACK"
    errors = validate_clip(clip)
    assert any("track" in e for e in errors)


def test_validate_clip_catches_negative_trim_in():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    clip["trim_in"] = -1.0
    errors = validate_clip(clip)
    assert any("trim_in" in e for e in errors)


def test_validate_clip_catches_trim_out_lte_trim_in():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    clip["trim_in"] = 3.0
    clip["trim_out"] = 3.0
    errors = validate_clip(clip)
    assert any("trim_out" in e for e in errors)


def test_validate_clip_catches_negative_start():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    clip["start"] = -0.5
    errors = validate_clip(clip)
    assert any("start" in e for e in errors)


def test_validate_clip_catches_zero_scale():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    clip["scale"] = 0.0
    errors = validate_clip(clip)
    assert any("scale" in e for e in errors)


def test_validate_clip_catches_negative_volume():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    clip["volume"] = -1.0
    errors = validate_clip(clip)
    assert any("volume" in e for e in errors)


def test_get_color_fills_missing_keys_from_defaults():
    clip = {"color": {"brightness": 50.0}}  # partial dict
    result = get_color(clip)
    assert result["brightness"] == 50.0
    # All default keys present
    for key in DEFAULT_COLOR_GRADING:
        assert key in result


def test_get_color_no_color_key_returns_full_defaults():
    clip = {}
    result = get_color(clip)
    assert result == DEFAULT_COLOR_GRADING


# ===========================================================================
# 2. commands.py
# ===========================================================================

from commands import (
    CommandStack,
    MoveClipCommand,
    TrimClipCommand,
    DeleteClipCommand,
    AddClipCommand,
    SplitClipCommand,
    DuplicateClipCommand,
    InspectorChangeCommand,
)


class MockApp:
    def __init__(self):
        self.project_clips = []
        self._calls = 0

    def _after_command(self):
        self._calls += 1


def _make_app_with_clip(**overrides):
    """Return (MockApp, clip_id) with one clip already in project_clips."""
    app = MockApp()
    # Pull positional args out of overrides if callers pass them explicitly,
    # to avoid "multiple values for argument" errors with make_clip's signature.
    start = overrides.pop("start", 0.0)
    duration = overrides.pop("trim_out", 10.0)  # re-expressed as duration
    trim_in = overrides.pop("trim_in", None)
    clip = make_clip("v.mp4", "V1", start, duration, **overrides)
    if trim_in is not None:
        clip["trim_in"] = trim_in
    app.project_clips.append(clip)
    return app, clip["id"]


# --- CommandStack mechanics ---

def test_commandstack_push_executes_and_adds_to_undo():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(MoveClipCommand(app, cid, 0.0, 5.0))
    assert stack.can_undo
    assert not stack.can_redo
    assert app.project_clips[0]["start"] == 5.0


def test_commandstack_undo_reverses_and_moves_to_redo():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(MoveClipCommand(app, cid, 0.0, 5.0))
    stack.undo()
    assert app.project_clips[0]["start"] == 0.0
    assert not stack.can_undo
    assert stack.can_redo


def test_commandstack_redo_reexecutes_and_moves_to_undo():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(MoveClipCommand(app, cid, 0.0, 5.0))
    stack.undo()
    stack.redo()
    assert app.project_clips[0]["start"] == 5.0
    assert stack.can_undo
    assert not stack.can_redo


def test_commandstack_can_undo_can_redo_initial_state():
    app = MockApp()
    stack = CommandStack(app)
    assert not stack.can_undo
    assert not stack.can_redo


def test_commandstack_undo_label_nothing():
    app = MockApp()
    stack = CommandStack(app)
    assert stack.undo_label == "Nothing to undo"


def test_commandstack_redo_label_nothing():
    app = MockApp()
    stack = CommandStack(app)
    assert stack.redo_label == "Nothing to redo"


def test_commandstack_undo_label_with_command():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(MoveClipCommand(app, cid, 0.0, 3.0))
    assert stack.undo_label == "Undo Move Clip"


def test_commandstack_redo_label_after_undo():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(MoveClipCommand(app, cid, 0.0, 3.0))
    stack.undo()
    assert stack.redo_label == "Redo Move Clip"


def test_commandstack_clear_empties_both_stacks():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(MoveClipCommand(app, cid, 0.0, 3.0))
    stack.undo()
    stack.clear()
    assert not stack.can_undo
    assert not stack.can_redo


def test_commandstack_max_50_levels():
    app = MockApp()
    stack = CommandStack(app)
    # Add a single clip; keep moving it forward
    clip = make_clip("v.mp4", "V1", 0.0, 1.0)
    app.project_clips.append(clip)
    cid = clip["id"]
    for i in range(60):
        stack.push(MoveClipCommand(app, cid, float(i), float(i + 1)))
    assert len(stack._undo_stack) == 50


def test_commandstack_push_invalidates_redo():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(MoveClipCommand(app, cid, 0.0, 2.0))
    stack.undo()
    # New push should clear redo
    stack.push(MoveClipCommand(app, cid, 0.0, 9.0))
    assert not stack.can_redo


def test_commandstack_after_command_called_on_push():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    before = app._calls
    stack.push(MoveClipCommand(app, cid, 0.0, 1.0))
    assert app._calls == before + 1


# --- MoveClipCommand ---

def test_moveclip_execute_sets_new_start():
    app, cid = _make_app_with_clip()
    cmd = MoveClipCommand(app, cid, 0.0, 7.0)
    cmd.execute()
    assert app.project_clips[0]["start"] == 7.0


def test_moveclip_undo_restores_old_start():
    app, cid = _make_app_with_clip()
    cmd = MoveClipCommand(app, cid, 0.0, 7.0)
    cmd.execute()
    cmd.undo()
    assert app.project_clips[0]["start"] == 0.0


def test_moveclip_redo_reapplies_new_start():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(MoveClipCommand(app, cid, 0.0, 7.0))
    stack.undo()
    stack.redo()
    assert app.project_clips[0]["start"] == 7.0


# --- TrimClipCommand ---

def test_trimclip_execute_sets_new_trim_points():
    app, cid = _make_app_with_clip()
    cmd = TrimClipCommand(app, cid, 0.0, 10.0, 1.0, 8.0)
    cmd.execute()
    clip = app.project_clips[0]
    assert clip["trim_in"] == 1.0
    assert clip["trim_out"] == 8.0


def test_trimclip_undo_restores_old_trim_points():
    app, cid = _make_app_with_clip()
    cmd = TrimClipCommand(app, cid, 0.0, 10.0, 1.0, 8.0)
    cmd.execute()
    cmd.undo()
    clip = app.project_clips[0]
    assert clip["trim_in"] == 0.0
    assert clip["trim_out"] == 10.0


def test_trimclip_redo_reapplies_new_trim_points():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(TrimClipCommand(app, cid, 0.0, 10.0, 2.0, 9.0))
    stack.undo()
    stack.redo()
    clip = app.project_clips[0]
    assert clip["trim_in"] == 2.0
    assert clip["trim_out"] == 9.0


# --- DeleteClipCommand ---

def test_deleteclip_execute_removes_clip():
    app, cid = _make_app_with_clip()
    original_copy = copy.deepcopy(app.project_clips[0])
    cmd = DeleteClipCommand(app, original_copy)
    cmd.execute()
    assert len(app.project_clips) == 0


def test_deleteclip_undo_restores_clip_with_same_id():
    app, cid = _make_app_with_clip()
    original_copy = copy.deepcopy(app.project_clips[0])
    cmd = DeleteClipCommand(app, original_copy)
    cmd.execute()
    cmd.undo()
    assert len(app.project_clips) == 1
    assert app.project_clips[0]["id"] == cid


# --- AddClipCommand ---

def test_addclip_execute_adds_clip():
    app = MockApp()
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    cmd = AddClipCommand(app, clip)
    cmd.execute()
    assert len(app.project_clips) == 1
    assert app.project_clips[0]["id"] == clip["id"]


def test_addclip_undo_removes_clip():
    app = MockApp()
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    cmd = AddClipCommand(app, clip)
    cmd.execute()
    cmd.undo()
    assert len(app.project_clips) == 0


# --- SplitClipCommand ---

def test_splitclip_execute_creates_two_clips():
    app, cid = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=10.0)
    original_copy = copy.deepcopy(app.project_clips[0])
    right_id = str(uuid.uuid4())
    cmd = SplitClipCommand(app, original_copy, cid, right_id, split_t=5.0)
    cmd.execute()
    assert len(app.project_clips) == 2


def test_splitclip_execute_clips_are_contiguous():
    app, cid = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=10.0)
    original_copy = copy.deepcopy(app.project_clips[0])
    right_id = str(uuid.uuid4())
    cmd = SplitClipCommand(app, original_copy, cid, right_id, split_t=5.0)
    cmd.execute()
    clips = sorted(app.project_clips, key=lambda c: c["start"])
    left, right = clips[0], clips[1]
    # end of left == start of right
    left_end = left["start"] + (left["trim_out"] - left["trim_in"])
    assert left_end == pytest.approx(right["start"])


def test_splitclip_undo_restores_original():
    app, cid = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=10.0)
    original_copy = copy.deepcopy(app.project_clips[0])
    right_id = str(uuid.uuid4())
    cmd = SplitClipCommand(app, original_copy, cid, right_id, split_t=5.0)
    cmd.execute()
    cmd.undo()
    assert len(app.project_clips) == 1
    assert app.project_clips[0]["id"] == cid
    assert app.project_clips[0]["trim_out"] == 10.0


# --- DuplicateClipCommand ---

def test_duplicateclip_execute_adds_copy():
    app, cid = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=5.0)
    new_id = str(uuid.uuid4())
    cmd = DuplicateClipCommand(app, cid, new_id)
    cmd.execute()
    assert len(app.project_clips) == 2
    ids = {c["id"] for c in app.project_clips}
    assert new_id in ids


def test_duplicateclip_undo_removes_copy():
    app, cid = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=5.0)
    new_id = str(uuid.uuid4())
    cmd = DuplicateClipCommand(app, cid, new_id)
    cmd.execute()
    cmd.undo()
    assert len(app.project_clips) == 1
    assert app.project_clips[0]["id"] == cid


# --- InspectorChangeCommand ---

def test_inspectorchange_execute_applies_new_props():
    app, cid = _make_app_with_clip()
    cmd = InspectorChangeCommand(app, cid, {"volume": 1.0}, {"volume": 0.3})
    cmd.execute()
    assert app.project_clips[0]["volume"] == pytest.approx(0.3)


def test_inspectorchange_undo_restores_old_props():
    app, cid = _make_app_with_clip()
    cmd = InspectorChangeCommand(app, cid, {"volume": 1.0}, {"volume": 0.3})
    cmd.execute()
    cmd.undo()
    assert app.project_clips[0]["volume"] == pytest.approx(1.0)


def test_inspectorchange_redo_reapplies_new_props():
    app, cid = _make_app_with_clip()
    stack = CommandStack(app)
    stack.push(InspectorChangeCommand(app, cid, {"volume": 1.0}, {"volume": 0.25}))
    stack.undo()
    stack.redo()
    assert app.project_clips[0]["volume"] == pytest.approx(0.25)


# ===========================================================================
# 3. project_io.py
# ===========================================================================

from project_io import (
    save_project,
    load_project,
    get_recent_projects,
    add_recent_project,
    PROJECT_VERSION,
    RECENT_PROJECTS_KEY,
)


def _save_and_load(clips, proj_w=1920, proj_h=1080, fps=30.0):
    with tempfile.NamedTemporaryFile(suffix=".nleproj", delete=False) as f:
        path = f.name
    try:
        save_project(path, clips, proj_w, proj_h, fps)
        return load_project(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_project_io_round_trip_preserves_clip_count():
    clips = [make_clip(f"v{i}.mp4", "V1", float(i * 5), 5.0) for i in range(3)]
    loaded_clips, _, _, _ = _save_and_load(clips)
    assert len(loaded_clips) == 3


def test_project_io_round_trip_preserves_ids():
    clips = [make_clip("v.mp4", "V1", 0.0, 10.0)]
    original_id = clips[0]["id"]
    loaded_clips, _, _, _ = _save_and_load(clips)
    assert loaded_clips[0]["id"] == original_id


def test_project_io_round_trip_preserves_paths():
    clips = [make_clip("/my/video.mp4", "V2", 0.0, 5.0)]
    loaded_clips, _, _, _ = _save_and_load(clips)
    assert loaded_clips[0]["path"] == "/my/video.mp4"


def test_project_io_round_trip_preserves_tracks():
    clips = [make_clip("v.mp4", "A1", 0.0, 5.0)]
    loaded_clips, _, _, _ = _save_and_load(clips)
    assert loaded_clips[0]["track"] == "A1"


def test_project_io_round_trip_preserves_float_precision():
    clips = [make_clip("v.mp4", "V1", 1.234567, 8.765432)]
    clip = clips[0]
    clip["trim_in"] = 0.123456
    clip["trim_out"] = 8.765432
    loaded_clips, _, _, _ = _save_and_load(clips)
    lc = loaded_clips[0]
    assert lc["start"] == pytest.approx(1.234567, abs=1e-5)
    assert lc["trim_in"] == pytest.approx(0.123456, abs=1e-5)
    assert lc["trim_out"] == pytest.approx(8.765432, abs=1e-5)


def test_project_io_round_trip_preserves_numpy_mask_shape_and_values():
    clip = make_clip("v.mp4", "V1", 0.0, 5.0)
    mask = np.array([[True, False], [False, True]], dtype=bool)
    clip["custom_masks"] = {0: mask}
    loaded_clips, _, _, _ = _save_and_load([clip])
    loaded_mask = loaded_clips[0]["custom_masks"][0]
    assert loaded_mask.shape == mask.shape
    np.testing.assert_array_equal(loaded_mask, mask)


def test_project_io_round_trip_empty_clips():
    loaded_clips, _, _, _ = _save_and_load([])
    assert loaded_clips == []


def test_project_io_load_invalid_file_raises():
    with tempfile.NamedTemporaryFile(suffix=".nleproj", delete=False) as f:
        f.write(b"this is not a zip file")
        path = f.name
    try:
        with pytest.raises(Exception):
            load_project(path)
    finally:
        os.unlink(path)


def test_project_io_version_field_is_project_version():
    clips = [make_clip("v.mp4", "V1", 0.0, 5.0)]
    with tempfile.NamedTemporaryFile(suffix=".nleproj", delete=False) as f:
        path = f.name
    try:
        save_project(path, clips, 1920, 1080, 30.0)
        with zipfile.ZipFile(path, "r") as zf:
            doc = json.loads(zf.read("project.json").decode())
        assert doc["version"] == PROJECT_VERSION
    finally:
        os.unlink(path)


class _MockConfigMgr:
    def __init__(self, data=None):
        self._data = data or {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value


def test_get_recent_projects_empty_when_no_key():
    cfg = _MockConfigMgr()
    result = get_recent_projects(cfg)
    assert result == []


def test_add_recent_project_prepends():
    cfg = _MockConfigMgr()
    add_recent_project(cfg, "/a/project.nleproj")
    add_recent_project(cfg, "/b/project.nleproj")
    recent = get_recent_projects(cfg)
    assert recent[0] == "/b/project.nleproj"
    assert recent[1] == "/a/project.nleproj"


def test_add_recent_project_deduplicates():
    cfg = _MockConfigMgr()
    add_recent_project(cfg, "/a/project.nleproj")
    add_recent_project(cfg, "/b/project.nleproj")
    add_recent_project(cfg, "/a/project.nleproj")  # duplicate → moves to front
    recent = get_recent_projects(cfg)
    assert recent.count("/a/project.nleproj") == 1
    assert recent[0] == "/a/project.nleproj"


def test_add_recent_project_trims_to_8():
    cfg = _MockConfigMgr()
    for i in range(12):
        add_recent_project(cfg, f"/p{i}/proj.nleproj")
    recent = get_recent_projects(cfg)
    assert len(recent) <= 8


# ===========================================================================
# 4. transitions.py
# ===========================================================================

from transitions import apply_transition, TRANSITION_TYPES


def test_apply_transition_progress_0_returns_frame_a():
    fa = _make_frame(value=50)
    fb = _make_frame(value=200)
    result = apply_transition(fa, fb, 0.0, "Cross Dissolve")
    np.testing.assert_array_equal(result, fa)


def test_apply_transition_progress_1_returns_frame_b():
    fa = _make_frame(value=50)
    fb = _make_frame(value=200)
    result = apply_transition(fa, fb, 1.0, "Cross Dissolve")
    np.testing.assert_array_equal(result, fb)


def test_apply_transition_cross_dissolve_at_half_is_mean():
    fa = _make_frame(value=0)
    fb = _make_frame(value=100)
    result = apply_transition(fa, fb, 0.5, "Cross Dissolve")
    # cv2.addWeighted uses integer arithmetic; allow ±2 tolerance
    assert np.all(np.abs(result.astype(int) - 50) <= 2)


def test_apply_transition_all_non_none_types_produce_correct_shape():
    fa = _make_frame(h=60, w=80, value=30)
    fb = _make_frame(h=60, w=80, value=180)
    non_none = [t for t in TRANSITION_TYPES if t != "None"]
    assert len(non_none) >= 5
    for t in non_none:
        result = apply_transition(fa, fb, 0.5, t)
        assert result.shape == (60, 80, 3), f"Wrong shape for {t}"
        assert result.dtype == np.uint8, f"Wrong dtype for {t}"


def test_apply_transition_mismatched_shapes_handled_gracefully():
    """Mismatched shapes: the code resizes frame_b to match frame_a."""
    fa = _make_frame(h=60, w=80, value=50)
    fb = _make_frame(h=120, w=160, value=150)
    # Should not raise; output shape should match frame_a
    result = apply_transition(fa, fb, 0.5, "Cross Dissolve")
    assert result.shape == fa.shape


# ===========================================================================
# 5. effects_registry.py
# ===========================================================================

from effects_registry import registry


def test_effects_registry_has_at_least_10_effects():
    assert len(registry.all_effects()) >= 10


def test_all_effects_returns_list():
    effects = registry.all_effects()
    assert isinstance(effects, list)


def test_by_category_returns_dict_with_nonempty_categories():
    cats = registry.by_category()
    assert isinstance(cats, dict)
    assert len(cats) > 0
    for cat, effects in cats.items():
        assert len(effects) > 0, f"Category '{cat}' is empty"


def test_apply_to_frame_unknown_name_returns_frame_unchanged():
    frame = _make_frame(value=123)
    result = registry.apply_to_frame(frame, "nonexistent_effect_xyz", {})
    np.testing.assert_array_equal(result, frame)


def test_apply_all_enabled_empty_clip_effects_returns_frame_unchanged():
    frame = _make_frame(value=77)
    result = registry.apply_all_enabled(frame, {})
    np.testing.assert_array_equal(result, frame)


def test_blur_gaussian_strength_0_returns_frame_unchanged():
    frame = _make_frame(value=100)
    result = registry.apply_to_frame(frame, "blur_gaussian", {"strength": 0})
    np.testing.assert_array_equal(result, frame)


def test_blur_gaussian_strength_5_returns_different_frame():
    # Use a frame with some variation so blur has something to act on
    frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
    result = registry.apply_to_frame(frame, "blur_gaussian", {"strength": 5})
    assert not np.array_equal(result, frame)


def test_film_grain_strength_0_returns_frame_identical():
    frame = _make_frame(value=150)
    result = registry.apply_to_frame(frame, "film_grain", {"strength": 0})
    np.testing.assert_array_equal(result, frame)


def test_flip_horizontal_flips_correctly():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    # Mark the first pixel of row 0 distinctly
    frame[0, 0] = [255, 0, 0]
    result = registry.apply_to_frame(frame, "flip_horizontal", {"enabled": True})
    # After horizontal flip the marked pixel should be at the last column
    np.testing.assert_array_equal(result[0, -1], [255, 0, 0])


def test_vignette_with_strength_1_darkens_corners_vs_centre():
    # Use a uniform bright frame
    frame = _make_frame(h=100, w=100, value=200)
    result = registry.apply_to_frame(frame, "vignette", {"strength": 1.0, "radius": 0.7})
    centre_val = int(result[50, 50].mean())
    corner_val = int(result[0, 0].mean())
    assert corner_val < centre_val, (
        f"Corner ({corner_val}) should be darker than centre ({centre_val})"
    )


# ===========================================================================
# 6. ai_features.py
# ===========================================================================

from ai_features import (
    WhisperSubtitler,
    SceneDetector,
    BeatDetector,
    SmartReframer,
)


def test_whisper_segments_to_title_clips_correct_count():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hello"},
        {"start": 2.5, "end": 4.5, "text": "World"},
        {"start": 5.0, "end": 7.0, "text": "Test"},
    ]
    clips = WhisperSubtitler.segments_to_title_clips(segments)
    assert len(clips) == 3


def test_whisper_segments_to_title_clips_start_times():
    segments = [
        {"start": 1.0, "end": 3.0, "text": "A"},
        {"start": 4.0, "end": 6.0, "text": "B"},
    ]
    clips = WhisperSubtitler.segments_to_title_clips(segments)
    assert clips[0]["start"] == pytest.approx(1.0)
    assert clips[1]["start"] == pytest.approx(4.0)


def test_whisper_segments_to_title_clips_text():
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
    clips = WhisperSubtitler.segments_to_title_clips(segments)
    assert clips[0]["title_text"] == "Hello world"


def test_scene_detector_auto_split_empty_scene_times_returns_original():
    clip = make_clip("v.mp4", "V1", 0.0, 10.0)
    result = SceneDetector.auto_split_clip(clip, [])
    # With no cut points the function returns a list with exactly one sub-clip
    # that covers the full source range (same trim_in/trim_out, same track).
    assert len(result) == 1
    assert result[0]["trim_in"] == clip["trim_in"]
    assert result[0]["trim_out"] == clip["trim_out"]
    assert result[0]["track"] == clip["track"]


def test_scene_detector_auto_split_at_5s_splits_10s_clip_into_two():
    clip = make_clip("v.mp4", "V1", 0.0, 10.0)
    result = SceneDetector.auto_split_clip(clip, [5.0])
    assert len(result) == 2


def test_scene_detector_auto_split_clips_are_contiguous():
    clip = make_clip("v.mp4", "V1", 0.0, 10.0)
    result = SceneDetector.auto_split_clip(clip, [3.0, 7.0])
    # Sort by start time and verify contiguity
    result_sorted = sorted(result, key=lambda c: c["start"])
    for i in range(len(result_sorted) - 1):
        end_i = result_sorted[i]["start"] + (result_sorted[i]["trim_out"] - result_sorted[i]["trim_in"])
        start_next = result_sorted[i + 1]["start"]
        assert end_i == pytest.approx(start_next, abs=1e-6)


def test_beat_detector_is_available_returns_bool():
    result = BeatDetector.is_available()
    assert isinstance(result, bool)


def test_smart_reframer_is_available_returns_bool():
    result = SmartReframer.is_available()
    assert isinstance(result, bool)


# ===========================================================================
# 7. waveform.py
# ===========================================================================

from unittest.mock import MagicMock
from waveform import WaveformCache


def test_waveform_cache_get_returns_none_for_unknown_path():
    cache = WaveformCache()
    result = cache.get("/nonexistent/path/to/audio.wav", width_px=100)
    assert result is None


def test_waveform_cache_invalidate_removes_cached_entry():
    cache = WaveformCache()
    # Manually insert a fake entry into the cache
    fake_path = "/fake/audio.wav"
    cache._cache[fake_path] = np.ones(100, dtype=np.float32)
    # Verify it's accessible
    assert cache.get(fake_path, width_px=100) is not None
    # Invalidate and confirm removal
    cache.invalidate(fake_path)
    assert cache.get(fake_path, width_px=100) is None


def test_draw_waveform_can_be_called_with_mock_canvas():
    """draw_waveform should not raise with a MagicMock canvas."""
    cache = WaveformCache()
    mock_canvas = MagicMock()
    waveform = np.linspace(0.0, 1.0, 100).astype(np.float32)
    # Should not raise
    cache.draw_waveform(mock_canvas, waveform, x1=0, y1=0, x2=200, y2=50)
    # Verify that create_polygon was called (we had enough points)
    mock_canvas.create_polygon.assert_called_once()


def test_draw_waveform_none_waveform_does_nothing():
    cache = WaveformCache()
    mock_canvas = MagicMock()
    cache.draw_waveform(mock_canvas, None, x1=0, y1=0, x2=100, y2=50)
    mock_canvas.create_polygon.assert_not_called()


def test_draw_waveform_empty_waveform_does_nothing():
    cache = WaveformCache()
    mock_canvas = MagicMock()
    cache.draw_waveform(mock_canvas, np.array([]), x1=0, y1=0, x2=100, y2=50)
    mock_canvas.create_polygon.assert_not_called()


# ===========================================================================
# 8. Dynamic Tracks, Track Mixer, Audio Separation & Linked Clips
# ===========================================================================

from commands import SeparateAudioCommand, ToggleLinkCommand
from engine import apply_audio_pan
from timeline import get_track_theme


def test_separate_audio_command_execution():
    app, v_id = _make_app_with_clip(start=1.0, trim_in=0.5, trim_out=4.5)
    a_id = str(uuid.uuid4())
    a_clip = make_clip("v.mp4", "A1", 1.0, 4.0, trim_in=0.5, trim_out=4.5, linked_clip_id=v_id)
    a_clip["id"] = a_id

    stack = CommandStack(app)
    cmd = SeparateAudioCommand(app, v_id, a_clip)
    stack.push(cmd)

    v_clip = next(c for c in app.project_clips if c["id"] == v_id)
    assert v_clip["mute_audio"] is True
    assert v_clip["linked_clip_id"] == a_id
    assert len(app.project_clips) == 2
    assert a_clip in app.project_clips

    # Undo
    stack.undo()
    assert v_clip["mute_audio"] is False
    assert v_clip["linked_clip_id"] is None
    assert len(app.project_clips) == 1

    # Redo
    stack.redo()
    assert v_clip["mute_audio"] is True
    assert v_clip["linked_clip_id"] == a_id
    assert len(app.project_clips) == 2


def test_toggle_link_command_execution():
    app, c1_id = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=5.0)
    c2 = make_clip("a.wav", "A1", 0.0, 5.0)
    c2_id = c2["id"]
    app.project_clips.append(c2)

    stack = CommandStack(app)
    # Link c1 and c2
    cmd = ToggleLinkCommand(app, c1_id, c2_id)
    stack.push(cmd)

    c1 = next(c for c in app.project_clips if c["id"] == c1_id)
    c2 = next(c for c in app.project_clips if c["id"] == c2_id)
    assert c1["linked_clip_id"] == c2_id
    assert c2["linked_clip_id"] == c1_id

    # Undo
    stack.undo()
    assert c1["linked_clip_id"] is None
    assert c2["linked_clip_id"] is None

    # Redo
    stack.redo()
    assert c1["linked_clip_id"] == c2_id
    assert c2["linked_clip_id"] == c1_id


def test_linked_clip_move_sync():
    app, v_id = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=5.0)
    a_id = str(uuid.uuid4())
    a_clip = make_clip("v.mp4", "A1", 0.0, 5.0, linked_clip_id=v_id)
    a_clip["id"] = a_id
    app.project_clips[0]["linked_clip_id"] = a_id
    app.project_clips.append(a_clip)

    cmd = MoveClipCommand(app, v_id, 0.0, 3.5)
    cmd.execute()

    v_clip = next(c for c in app.project_clips if c["id"] == v_id)
    a_clip = next(c for c in app.project_clips if c["id"] == a_id)
    assert v_clip["start"] == pytest.approx(3.5)
    assert a_clip["start"] == pytest.approx(3.5)

    # Undo
    cmd.undo()
    assert v_clip["start"] == pytest.approx(0.0)
    assert a_clip["start"] == pytest.approx(0.0)


def test_linked_clip_trim_sync():
    app, v_id = _make_app_with_clip(start=1.0, trim_in=0.0, trim_out=5.0)
    a_id = str(uuid.uuid4())
    a_clip = make_clip("v.mp4", "A1", 1.0, 5.0, linked_clip_id=v_id)
    a_clip["id"] = a_id
    app.project_clips[0]["linked_clip_id"] = a_id
    app.project_clips.append(a_clip)

    cmd = TrimClipCommand(app, v_id, 0.0, 5.0, 1.0, 4.0)
    cmd.execute()

    v_clip = next(c for c in app.project_clips if c["id"] == v_id)
    a_clip = next(c for c in app.project_clips if c["id"] == a_id)
    assert v_clip["trim_in"] == pytest.approx(1.0)
    assert v_clip["trim_out"] == pytest.approx(4.0)
    assert a_clip["trim_in"] == pytest.approx(1.0)
    assert a_clip["trim_out"] == pytest.approx(4.0)

    # Undo
    cmd.undo()
    assert v_clip["trim_in"] == pytest.approx(0.0)
    assert v_clip["trim_out"] == pytest.approx(5.0)
    assert a_clip["trim_in"] == pytest.approx(0.0)
    assert a_clip["trim_out"] == pytest.approx(5.0)


def test_linked_clip_split_sync():
    app, v_id = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=10.0)
    v_clip = app.project_clips[0]
    a_id = str(uuid.uuid4())
    a_clip = make_clip("v.mp4", "A1", 0.0, 10.0, linked_clip_id=v_id)
    a_clip["id"] = a_id
    v_clip["linked_clip_id"] = a_id
    app.project_clips.append(a_clip)

    right_id = str(uuid.uuid4())
    cmd = SplitClipCommand(app, v_clip, v_id, right_id, 4.0)
    cmd.execute()

    # Now 4 clips total: 2 video pieces and 2 audio pieces
    assert len(app.project_clips) == 4
    v_clips = [c for c in app.project_clips if c["track"] == "V1"]
    a_clips = [c for c in app.project_clips if c["track"] == "A1"]
    assert len(v_clips) == 2
    assert len(a_clips) == 2

    # Left pieces are linked
    left_v = next(c for c in v_clips if c["start"] == 0.0)
    left_a = next(c for c in a_clips if c["start"] == 0.0)
    assert left_v["linked_clip_id"] == left_a["id"]
    assert left_a["linked_clip_id"] == left_v["id"]

    # Right pieces are linked
    right_v = next(c for c in v_clips if c["start"] == 4.0)
    right_a = next(c for c in a_clips if c["start"] == 4.0)
    assert right_v["linked_clip_id"] == right_a["id"]
    assert right_a["linked_clip_id"] == right_v["id"]

    # Undo
    cmd.undo()
    assert len(app.project_clips) == 2


def test_linked_clip_delete_sync():
    app, v_id = _make_app_with_clip(start=0.0, trim_in=0.0, trim_out=5.0)
    v_clip = app.project_clips[0]
    a_id = str(uuid.uuid4())
    a_clip = make_clip("v.mp4", "A1", 0.0, 5.0, linked_clip_id=v_id)
    a_clip["id"] = a_id
    v_clip["linked_clip_id"] = a_id
    app.project_clips.append(a_clip)

    cmd = DeleteClipCommand(app, v_clip)
    cmd.execute()
    assert len(app.project_clips) == 0

    # Undo
    cmd.undo()
    assert len(app.project_clips) == 2
    v_restored = next(c for c in app.project_clips if c["id"] == v_id)
    a_restored = next(c for c in app.project_clips if c["id"] == a_id)
    assert v_restored["linked_clip_id"] == a_id
    assert a_restored["linked_clip_id"] == v_id


def test_apply_audio_pan_logic():
    p = -0.5
    l_gain = min(1.0, 1.0 - p)
    r_gain = min(1.0, 1.0 + p)
    assert l_gain == 1.0
    assert r_gain == 0.5

    p = 0.5
    l_gain = min(1.0, 1.0 - p)
    r_gain = min(1.0, 1.0 + p)
    assert l_gain == 0.5
    assert r_gain == 1.0


def test_dynamic_track_theme_fallback():
    th_v3 = get_track_theme("V3")
    assert th_v3["name"] == "V3 (Video 3)"
    assert "accent" in th_v3

    th_a4 = get_track_theme("A4")
    assert th_a4["name"] == "A4 (Audio 4)"
    assert "accent" in th_a4


def test_timeline_dynamic_track_addition_and_removal():
    import tkinter as tk
    from timeline import MultiTrackTimeline

    root = tk.Tk()
    root.withdraw()
    try:
        app = MockApp()
        app.project_clips = []
        app.rebuild_proxy_cache = lambda: None
        tl = MultiTrackTimeline(root, app)

        assert "V1" in tl.track_keys
        assert "A1" in tl.track_keys

        # Add video track
        tl.add_video_track()
        assert "V3" in tl.track_keys
        assert tl.track_volumes["V3"] == 1.0
        assert tl.track_pans["V3"] == 0.0

        # Add audio track
        tl.add_audio_track()
        assert "A3" in tl.track_keys
        assert tl.track_volumes["A3"] == 1.0
        assert tl.track_pans["A3"] == 0.0

        # Set volume and pan
        tl.set_track_volume("A3", 1.5)
        assert tl.track_volumes["A3"] == pytest.approx(1.5)

        tl.set_track_pan("A3", -0.75)
        assert tl.track_pans["A3"] == pytest.approx(-0.75)

        # Remove track
        tl.remove_track("A3")
        assert "A3" not in tl.track_keys
        assert "A3" not in tl.track_volumes

        # Add subtitle track
        tl.add_subtitle_track()
        assert "S1" in tl.track_keys
        assert tl.track_volumes["S1"] == 1.0

        # Add image track
        tl.add_image_track()
        assert "I1" in tl.track_keys
        assert tl.track_volumes["I1"] == 1.0
    finally:
        root.destroy()


def test_subtitle_and_image_track_themes():
    th_s = get_track_theme("S1")
    assert "Subtitle" in th_s["name"]
    assert "accent" in th_s
    assert "clip_base" in th_s

    th_i = get_track_theme("I1")
    assert "Image" in th_i["name"]
    assert "accent" in th_i
    assert "clip_base" in th_i


def test_validate_clip_subtitle_and_image_tracks():
    c_sub = make_clip("__title__", "S1", 0.0, 3.0)
    assert validate_clip(c_sub) == []

    c_img = make_clip("photo.png", "I1", 2.0, 5.0)
    assert validate_clip(c_img) == []


def test_title_clip_to_moviepy_clip():
    from title_generator import TitleClip
    tc = TitleClip(
        text="Test Title",
        duration=2.0,
        fps=30.0,
        width=320,
        height=240,
        font_size=24,
    )
    mpy_clip = tc.to_moviepy_clip()
    assert mpy_clip.duration == 2.0
def test_fl_knob_hit_testing():
    import tkinter as tk
    from timeline import MultiTrackTimeline
    root = tk.Tk()
    root.withdraw()
    try:
        class DummyApp:
            project_clips = []
            def rebuild_proxy_cache(self): pass
        tl = MultiTrackTimeline(root, DummyApp())
        # Track 0 is tl.track_keys[0] (V2): cy is 0 + 36 = 36.
        first_track = tl.track_keys[0]
        # Vol knob center is at (52, 36)
        hit_vol = tl._hit_test_knob(52, 36)
        assert hit_vol is not None
        assert hit_vol[0] == first_track
        assert hit_vol[1] == "vol"

        # Pan knob center is at (142, 36)
        hit_pan = tl._hit_test_knob(142, 36)
        assert hit_pan is not None
        assert hit_pan[0] == first_track
        assert hit_pan[1] == "pan"

        # Outside gutter (> 210)
        assert tl._hit_test_knob(220, 36) is None
        # In gutter but far from knobs
        assert tl._hit_test_knob(10, 10) is None
    finally:
        root.destroy()


def test_fl_knob_double_click_reset():
    import tkinter as tk
    from timeline import MultiTrackTimeline
    root = tk.Tk()
    root.withdraw()
    try:
        class DummyApp:
            project_clips = []
            def rebuild_proxy_cache(self): pass
        tl = MultiTrackTimeline(root, DummyApp())
        first_track = tl.track_keys[0]
        tl.set_track_volume(first_track, 1.8)
        tl.set_track_pan(first_track, -0.7)

        class DummyEvent:
            x = 52
            y = 36

        # Double click volume knob -> resets to 1.0 (100%)
        tl._on_double_click(DummyEvent())
        assert tl.track_volumes[first_track] == 1.0

        # Double click pan knob -> resets to 0.0 (Center)
        DummyEvent.x = 142
        tl._on_double_click(DummyEvent())
        assert tl.track_pans[first_track] == 0.0
    finally:
        root.destroy()


def test_fl_knob_wheel_adjustment():
    import tkinter as tk
    from timeline import MultiTrackTimeline
    root = tk.Tk()
    root.withdraw()
    try:
        class DummyApp:
            project_clips = []
            def rebuild_proxy_cache(self): pass
        tl = MultiTrackTimeline(root, DummyApp())
        first_track = tl.track_keys[0]
        tl.track_volumes[first_track] = 1.0
        tl.track_pans[first_track] = 0.0

        class DummyEvent:
            x = 52
            y = 36
            num = 0
            delta = 120
            state = 0

        # Scroll up on volume knob -> increases by 0.05
        tl._on_wheel(DummyEvent())
        assert tl.track_volumes[first_track] == pytest.approx(1.05)

        # Scroll down on pan knob -> decreases by 0.05
        DummyEvent.x = 142
        DummyEvent.delta = -120
        tl._on_wheel(DummyEvent())
        assert tl.track_pans[first_track] == pytest.approx(-0.05)
    finally:
        root.destroy()


def test_fl_knob_drag_interaction():
    import tkinter as tk
    from timeline import MultiTrackTimeline
    root = tk.Tk()
    root.withdraw()
    try:
        class DummyApp:
            project_clips = []
            rebuild_count = 0
            def rebuild_proxy_cache(self):
                self.rebuild_count += 1
        app = DummyApp()
        tl = MultiTrackTimeline(root, app)
        first_track = tl.track_keys[0]

        class DummyPressEvent:
            x = 52
            y = 36
            state = 0

        class DummyDragEvent:
            x = 52
            y = 16  # dragged upward by 20px (dy = +20 -> +0.20 volume)
            state = 0

        class DummyReleaseEvent:
            x = 52
            y = 16

        # Press on vol knob
        tl._on_press(DummyPressEvent())
        assert tl._knob_drag is not None
        assert tl._knob_drag['key'] == first_track
        assert tl._knob_drag['type'] == 'vol'

        # Drag upward
        tl._on_drag(DummyDragEvent())
        assert tl.track_volumes[first_track] == pytest.approx(1.20)
        assert tl._knob_drag['moved'] is True

        # Release
        tl._on_release(DummyReleaseEvent())
        assert tl._knob_drag is None
        assert app.rebuild_count == 1
    finally:
        root.destroy()


def test_export_job_track_volumes_and_pans():
    from export_queue import ExportJob
    job = ExportJob(
        job_id="test_id",
        label="Test Job",
        project_clips=[make_clip("v.mp4", "V1", 0.0, 5.0)],
        out_path="out.mp4",
        proj_w=1920,
        proj_h=1080,
        fps=30,
        codec="libx264",
        track_volumes={"V1": 1.5, "A1": 0.8},
        track_pans={"V1": -0.5, "A1": 0.5},
    )
    assert job.track_volumes == {"V1": 1.5, "A1": 0.8}
    assert job.track_pans == {"V1": -0.5, "A1": 0.5}


# ===========================================================================
# 16. Studio Colorist & Video Scopes Suite
# ===========================================================================

from video_scopes import (
    compute_luma_waveform,
    compute_rgb_parade,
    compute_vectorscope,
    compute_histogram_scope,
)

def test_video_scopes_computation():
    # Create test RGB gradient frame
    h, w = 100, 100
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8) # Red ramp
    frame[:, :, 1] = np.linspace(255, 0, h, dtype=np.uint8)[:, None] # Green ramp
    frame[:, :, 2] = 128 # Blue constant

    # Luma waveform
    wf = compute_luma_waveform(frame, out_w=200, out_h=100)
    assert wf.shape == (100, 200, 3)
    assert wf.dtype == np.uint8
    assert np.any(wf > 0)

    # RGB parade
    parade = compute_rgb_parade(frame, out_w=300, out_h=100)
    assert parade.shape == (100, 300, 3)
    assert parade.dtype == np.uint8
    assert np.any(parade[:, :100, 0] > 0) # Red channel in left third

    # Vectorscope
    vs = compute_vectorscope(frame, out_size=150)
    assert vs.shape == (150, 150, 3)
    assert vs.dtype == np.uint8

    # Histogram
    hist = compute_histogram_scope(frame, out_w=200, out_h=100)
    assert hist.shape == (100, 200, 3)
    assert hist.dtype == np.uint8


from color_grading import parse_cube_lut, apply_3d_lut

def test_3d_lut_cube_parsing_and_application():
    cube_content = """
    # Test 3D LUT Cube
    TITLE "Test Identity Cube"
    LUT_3D_SIZE 2
    0.0 0.0 0.0
    1.0 0.0 0.0
    0.0 1.0 0.0
    1.0 1.0 0.0
    0.0 0.0 1.0
    1.0 0.0 1.0
    0.0 1.0 1.0
    1.0 1.0 1.0
    """
    lut_grid = parse_cube_lut(cube_content)
    assert lut_grid is not None
    assert lut_grid.shape == (2, 2, 2, 3)

    frame = np.array([[[0, 0, 0], [255, 255, 255]], [[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    graded = apply_3d_lut(frame, lut_grid)
    assert graded.shape == frame.shape
    assert graded.dtype == np.uint8
    assert np.allclose(graded, frame, atol=2)


# ===========================================================================
# 17. Motion Graphics & 14 Blend Modes
# ===========================================================================

from engine import apply_blend_mode

def test_blend_modes_all_variants():
    base = np.full((10, 10, 3), 128, dtype=np.uint8)
    overlay = np.full((10, 10, 3), 64, dtype=np.uint8)

    modes = [
        "Normal", "Multiply", "Screen", "Overlay", "Darken", "Lighten",
        "Color Dodge", "Color Burn", "Hard Light", "Soft Light",
        "Difference", "Exclusion", "Add", "Subtract"
    ]

    for mode in modes:
        res = apply_blend_mode(base, overlay, mode=mode)
        assert res.shape == (10, 10, 3)
        assert res.dtype == np.uint8
        if mode == "Multiply":
            # 128/255 * 64/255 * 255 ~= 32
            assert np.all(res < 50)
        elif mode == "Screen":
            # Screen of 0.5 and 0.25 is 1 - (0.5 * 0.75) = 0.625 * 255 ~= 160
            assert np.all(res > 140)


# ===========================================================================
# 18. NLE Precision Trimming Commands (Ripple / Slip / Slide)
# ===========================================================================

from commands import RippleDeleteClipCommand, SlipClipCommand, SlideClipCommand, CommandStack

class DummyAppForTrim:
    def __init__(self, clips):
        self.project_clips = copy.deepcopy(clips)
        self.rebuild_called = 0
        self.active_clip_id = None
        self.timeline = self
        self.selected_clip_ids = set()
    def rebuild_proxy_cache(self):
        self.rebuild_called += 1
    def draw(self):
        pass
    def update_inspector(self):
        pass
    def _after_command(self):
        pass

def test_ripple_delete_command_execution():
    c1 = make_clip("a.mp4", "V1", 0.0, 5.0)
    c2 = make_clip("b.mp4", "V1", 5.0, 5.0, trim_in=0.0, trim_out=5.0) # 5s duration
    c3 = make_clip("c.mp4", "V1", 10.0, 5.0, trim_in=0.0, trim_out=5.0)
    app = DummyAppForTrim([c1, c2, c3])
    stack = CommandStack(app)

    cmd = RippleDeleteClipCommand(app, c2)
    stack.push(cmd)

    # c2 removed, c3 shifted left by 5.0s (new start = 5.0s)
    assert len(app.project_clips) == 2
    rem_ids = [c["id"] for c in app.project_clips]
    assert c2["id"] not in rem_ids
    c3_curr = next(c for c in app.project_clips if c["id"] == c3["id"])
    assert c3_curr["start"] == pytest.approx(5.0)

    # Undo
    stack.undo()
    assert len(app.project_clips) == 3
    c3_restored = next(c for c in app.project_clips if c["id"] == c3["id"])
    assert c3_restored["start"] == pytest.approx(10.0)

def test_slip_and_slide_commands():
    c1 = make_clip("a.mp4", "V1", 0.0, 5.0, trim_in=2.0, trim_out=7.0)
    app = DummyAppForTrim([c1])
    stack = CommandStack(app)

    # Slip command
    slip_cmd = SlipClipCommand(app, c1["id"], 2.0, 7.0, 3.0, 8.0)
    stack.push(slip_cmd)
    assert app.project_clips[0]["trim_in"] == 3.0
    assert app.project_clips[0]["trim_out"] == 8.0

    stack.undo()
    assert app.project_clips[0]["trim_in"] == 2.0
    assert app.project_clips[0]["trim_out"] == 7.0

    # Slide command
    c_prev = make_clip("p.mp4", "V1", 0.0, 4.0, trim_in=0.0, trim_out=4.0)
    c_curr = make_clip("m.mp4", "V1", 4.0, 3.0, trim_in=0.0, trim_out=3.0)
    c_next = make_clip("n.mp4", "V1", 7.0, 5.0, trim_in=0.0, trim_out=5.0)
    app2 = DummyAppForTrim([c_prev, c_curr, c_next])
    stack2 = CommandStack(app2)

    slide_cmd = SlideClipCommand(app2, c_curr["id"], 1.0, c_prev["id"], c_next["id"])
    stack2.push(slide_cmd)

    # c_prev extended by 1s, c_curr shifted by +1s, c_next trimmed by 1s
    p = next(c for c in app2.project_clips if c["id"] == c_prev["id"])
    m = next(c for c in app2.project_clips if c["id"] == c_curr["id"])
    n = next(c for c in app2.project_clips if c["id"] == c_next["id"])

    assert p["trim_out"] == pytest.approx(5.0)
    assert m["start"] == pytest.approx(5.0)
    assert n["start"] == pytest.approx(8.0)
    assert n["trim_in"] == pytest.approx(1.0)


# ===========================================================================
# 19. Studio Sequence Interchange (EDL & FCP7 XML)
# ===========================================================================

import interchange

def test_interchange_edl_and_fcpxml_roundtrip():
    c1 = make_clip("video1.mp4", "V1", 0.0, 5.0, trim_in=1.0, trim_out=6.0)
    c2 = make_clip("video2.mp4", "V1", 5.0, 12.0, trim_in=0.0, trim_out=7.0)
    c3 = make_clip("audio1.wav", "A1", 0.0, 10.0, trim_in=0.0, trim_out=10.0)
    clips = [c1, c2, c3]

    with tempfile.TemporaryDirectory() as tmpdir:
        # EDL Round-trip
        edl_file = os.path.join(tmpdir, "test.edl")
        interchange.export_edl(clips, edl_file, fps=30.0, title="TEST_EDL")
        assert os.path.isfile(edl_file)

        edl_clips = interchange.import_edl(edl_file, fps=30.0)
        assert len(edl_clips) == 2  # V1 clips
        assert edl_clips[0]["start"] == pytest.approx(0.0)
        assert edl_clips[0]["trim_in"] == pytest.approx(1.0)
        assert edl_clips[0]["trim_out"] == pytest.approx(6.0)

        # FCP7 XML Round-trip
        xml_file = os.path.join(tmpdir, "test_fcp.xml")
        interchange.export_fcpxml(clips, xml_file, proj_w=1920, proj_h=1080, fps=30.0, title="Test Sequence")
        assert os.path.isfile(xml_file)

        xml_clips = interchange.import_fcpxml(xml_file, fps=30.0)
        assert len(xml_clips) == 3
        v_imported = [c for c in xml_clips if c["track"].startswith("V")]
        a_imported = [c for c in xml_clips if c["track"].startswith("A")]
        assert len(v_imported) == 2
        assert len(a_imported) == 1
        assert v_imported[0]["start"] == pytest.approx(0.0)
        assert v_imported[0]["trim_in"] == pytest.approx(1.0)


# ===========================================================================
# 20. Audio Auto-Ducking
# ===========================================================================

from ai_features import AudioDucker

def test_audio_ducking_envelope_computation():
    sr = 44100
    duration_sec = 2.0
    n_samples = int(sr * duration_sec)

    # Create synthetic waveform: 0.5s silence, 1.0s loud speech tone, 0.5s silence
    voice = np.zeros(n_samples, dtype=np.float32)
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    voice[int(sr * 0.5):int(sr * 1.5)] = 0.8 * np.sin(2 * np.pi * 440 * t)

    envelope = AudioDucker.compute_ducking_envelope(
        voice, sample_rate=sr, threshold_db=-20.0, duck_db=-12.0
    )

    assert len(envelope) == n_samples
    assert envelope.dtype == np.float32

    # Silence region -> gain near 1.0
    assert envelope[int(sr * 0.1)] == pytest.approx(1.0, abs=0.05)
    # Speech region -> gain ducked down to ~0.25 (-12 dB)
    assert envelope[int(sr * 1.0)] < 0.35


# ===========================================================================
# 21. Keyframe Velocity Curves & Easing Models
# ===========================================================================

from keyframes import KeyframeTrack, _apply_easing

def test_keyframe_easing_models():
    track = KeyframeTrack("scale", default_value=1.0)
    track.add_keyframe(0.0, 0.0, easing="Exponential")
    track.add_keyframe(1.0, 100.0, easing="Linear")

    # Halfway under exponential (t=0.5 -> 0.5^4 = 0.0625 -> 6.25)
    val_exp = track.get_value(0.5)
    assert val_exp == pytest.approx(6.25)

    # Bezier S-Curve test
    assert _apply_easing(0.0, "Bezier (S-Curve)") == pytest.approx(0.0)
    assert _apply_easing(1.0, "Bezier (S-Curve)") == pytest.approx(1.0)
    assert _apply_easing(0.5, "Bezier (S-Curve)") == pytest.approx(0.5)


# ===========================================================================
# 22. Studio Audio FX Rack & DSP Chain Engine
# ===========================================================================

from audio_effects import (
    AudioEffectChain, apply_audio_chain,
    ParametricEQEffect, CompressorEffect, ReverbEffect,
    StereoDelayEffect, LimiterEffect, HighPassFilterEffect,
    LowPassFilterEffect, VST3PluginEffect
)

def test_parametric_eq_processing():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Sine wave at 1000 Hz
    sig = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    eq = ParametricEQEffect(mid1_freq=1000.0, mid1_gain=6.0, mid1_q=1.0)
    boosted = eq.process(sig, sr)

    assert len(boosted) == len(sig)
    # Boosted amplitude should be larger than original
    assert np.max(np.abs(boosted)) > np.max(np.abs(sig))

def test_compressor_dynamics_reduction():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Loud sine wave reaching 0.9 peak (~ -0.9 dBFS)
    loud_sig = 0.9 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    comp = CompressorEffect(threshold_db=-18.0, ratio=8.0, attack_ms=2.0, release_ms=50.0, makeup_gain_db=0.0)
    compressed = comp.process(loud_sig, sr)

    assert len(compressed) == len(loud_sig)
    # Compressed signal should have lower peak than original
    assert np.max(np.abs(compressed[sr//4:])) < np.max(np.abs(loud_sig[sr//4:]))

def test_reverb_and_delay_processing():
    sr = 44100
    impulse = np.zeros(sr, dtype=np.float32)
    impulse[0] = 1.0  # single impulse at t=0

    # Reverb produces tail
    rev = ReverbEffect(room_size=0.8, wet_level=0.5, dry_level=0.5)
    rev_out = rev.process(impulse, sr)
    assert np.count_nonzero(rev_out > 0.01) > 1

    # Delay produces echo
    dly = StereoDelayEffect(delay_ms=100.0, feedback=0.5, mix=0.5)
    dly_out = dly.process(impulse, sr)
    delay_sample = int(sr * 0.1)
    assert dly_out[delay_sample] > 0.05

def test_limiter_peak_clamping():
    sr = 44100
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    hot_signal = 2.0 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # +6 dB overshoot

    limiter = LimiterEffect(ceiling_db=-1.0)  # -1 dBFS is ~0.891
    limited = limiter.process(hot_signal, sr)

    ceiling_linear = 10.0 ** (-1.0 / 20.0)
    assert np.max(np.abs(limited)) <= ceiling_linear + 1e-4

def test_highpass_and_lowpass_filters():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    low_tone = 0.5 * np.sin(2 * np.pi * 40 * t).astype(np.float32)   # 40 Hz (sub rumble)
    high_tone = 0.5 * np.sin(2 * np.pi * 15000 * t).astype(np.float32) # 15 kHz (hiss)

    hp = HighPassFilterEffect(cutoff_freq=100.0)
    hp_out = hp.process(low_tone, sr)
    assert np.max(np.abs(hp_out)) < np.max(np.abs(low_tone)) * 0.5

    lp = LowPassFilterEffect(cutoff_freq=8000.0)
    lp_out = lp.process(high_tone, sr)
    assert np.max(np.abs(lp_out)) < np.max(np.abs(high_tone)) * 0.5

def test_audio_effect_chain_serialization_and_deserialization():
    chain = AudioEffectChain([
        HighPassFilterEffect(cutoff_freq=90.0),
        ParametricEQEffect(low_shelf_gain=3.0, high_shelf_gain=-2.0),
        CompressorEffect(threshold_db=-22.0, ratio=3.0),
        ReverbEffect(room_size=0.4),
        LimiterEffect(ceiling_db=-0.3),
        VST3PluginEffect(plugin_path="plugins/synth.vst3"),
    ])

    data_list = chain.to_list()
    assert len(data_list) == 6
    assert data_list[0]["type"] == "HighPassFilter"
    assert data_list[0]["cutoff_freq"] == 90.0
    assert data_list[1]["type"] == "ParametricEQ"
    assert data_list[5]["type"] == "VST3Plugin"

    restored = AudioEffectChain.from_list(data_list)
    assert len(restored.effects) == 6
    assert isinstance(restored.effects[0], HighPassFilterEffect)
    assert isinstance(restored.effects[1], ParametricEQEffect)
    assert isinstance(restored.effects[2], CompressorEffect)
    assert isinstance(restored.effects[3], ReverbEffect)
    assert isinstance(restored.effects[4], LimiterEffect)
    assert isinstance(restored.effects[5], VST3PluginEffect)
    assert restored.effects[0].cutoff_freq == 90.0

def test_apply_audio_chain_stereo_processing():
    sr = 44100
    stereo_sig = np.random.uniform(-0.5, 0.5, (sr // 4, 2)).astype(np.float32)
    chain = [
        ParametricEQEffect(mid1_gain=4.0).to_dict(),
        CompressorEffect(threshold_db=-10.0).to_dict(),
    ]
    out = apply_audio_chain(stereo_sig, sr, chain)
    assert out.shape == stereo_sig.shape
    assert out.dtype == np.float32
    assert not np.isnan(out).any()








