import sys
import subprocess
import os
import importlib.util
import platform

# ---------------------------------------------------------------------------
# BOOTLOADER
# ---------------------------------------------------------------------------
# Dependency map: { importable_module_name: pip_package_name }
_REQUIREMENTS = {
    # Core
    'customtkinter':  'customtkinter',
    'decorator':      'decorator<5.0,>=4.0.2',
    'moviepy':        'moviepy<2',
    'cv2':            'opencv-python',
    'PIL':            'Pillow',
    'proglog':        'proglog',
    'rembg':          'rembg',
    'imageio':        'imageio',
    'numpy':          'numpy',
    # Audio / waveform
    'soundfile':      'soundfile',
    'noisereduce':    'noisereduce',
    'sounddevice':    'sounddevice',
    # AI features (optional — graceful degradation if missing)
    'whisper':        'openai-whisper',
    'scenedetect':    'scenedetect[opencv]',
    'librosa':        'librosa<1.0',
    # Export / titles
    'imageio_ffmpeg': 'imageio-ffmpeg',
    # GPU acceleration (optional)
    'onnxruntime':    'onnxruntime',
    # Drag-and-drop (optional — graceful degradation if missing)
    'tkinterdnd2':    'tkinterdnd2',
}

_IS_LINUX   = platform.system() == 'Linux'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_DIR   = os.path.join(_SCRIPT_DIR, '.venv')


def _venv_python():
    """Return the path to the Python executable inside the project venv."""
    if platform.system() == 'Windows':
        return os.path.join(_VENV_DIR, 'Scripts', 'python.exe')
    return os.path.join(_VENV_DIR, 'bin', 'python')


def _ensure_venv():
    """Create the project venv if it does not yet exist (Linux only)."""
    if not os.path.isdir(_VENV_DIR):
        print('[Bootloader] Creating project venv at', _VENV_DIR, '...')
        subprocess.check_call([sys.executable, '-m', 'venv', _VENV_DIR])
        print('[Bootloader] Venv created.')


def _running_in_venv():
    """Return True when the current interpreter lives inside _VENV_DIR."""
    return os.path.abspath(sys.executable).startswith(os.path.abspath(_VENV_DIR))


def _relaunch_in_venv():
    """Re-execute the current script using the venv interpreter (never returns)."""
    vpy = _venv_python()
    print('[Bootloader] Relaunching inside venv:', vpy)
    if platform.system() == 'Windows':
        result = subprocess.run([vpy] + sys.argv)
        sys.exit(result.returncode)
    else:
        os.execv(vpy, [vpy] + sys.argv)


def _find_missing(interpreter):
    """Return a list of pip package names that are not importable or incompatible under *interpreter*."""
    missing = []
    check_code = (
        'import sys, importlib.util, importlib.metadata\n'
        'try:\n'
        '    from packaging.requirements import Requirement\n'
        'except ImportError:\n'
        '    Requirement = None\n'
        'mod_name = sys.argv[1]\n'
        'pip_spec = sys.argv[2]\n'
        'if importlib.util.find_spec(mod_name) is None:\n'
        '    sys.exit(1)\n'
        'if Requirement:\n'
        '    try:\n'
        '        req = Requirement(pip_spec)\n'
        '        if req.specifier:\n'
        '            ver = importlib.metadata.version(req.name)\n'
        '            if not req.specifier.contains(ver, prereleases=True):\n'
        '                sys.exit(2)\n'
        '    except Exception:\n'
        '        pass\n'
        'sys.exit(0)\n'
    )
    for mod_name, pip_name in _REQUIREMENTS.items():
        check = subprocess.run(
            [interpreter, '-c', check_code, mod_name, pip_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode != 0:
            missing.append(pip_name)
    return missing


def _install(interpreter, packages):
    """Install *packages* using pip under *interpreter*."""
    print(f'[Bootloader] Installing: {", ".join(packages)}')
    subprocess.check_call([interpreter, '-m', 'pip', 'install', '--upgrade', 'pip'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([interpreter, '-m', 'pip', 'install', *packages])


def _bootloader():
    if _IS_LINUX:
        _ensure_venv()
        vpy = _venv_python()
        if not _running_in_venv():
            missing = _find_missing(vpy)
            if missing:
                _install(vpy, missing)
            _relaunch_in_venv()
            return
        missing = _find_missing(vpy)
        if missing:
            _install(vpy, missing)
            os.execv(vpy, [vpy] + sys.argv)
    else:
        missing = _find_missing(sys.executable)
        if missing:
            _install(sys.executable, missing)
            result = subprocess.run([sys.executable] + sys.argv)
            sys.exit(result.returncode)


_bootloader()
# ---------------------------------------------------------------------------
# End of bootloader — safe to import third-party packages below
# ---------------------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import customtkinter as ctk

# Optional drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _HAS_DND = True
except ImportError:
    _HAS_DND = False
import uuid
import copy
import cv2
import logging
from PIL import Image, ImageTk
from moviepy.editor import VideoFileClip, AudioFileClip, ColorClip, CompositeVideoClip  # requires moviepy<2

from config import config_mgr, setup_logging, toggle_console
from engine import RenderTask, SingleClipExportTask, is_image_file
from timeline import MultiTrackTimeline
from mask_painter import MaskPainter

# ---------------------------------------------------------------------------
# Optional module imports — graceful degradation if modules don't exist yet
# ---------------------------------------------------------------------------
try:
    from commands import (CommandStack, MoveClipCommand, TrimClipCommand,
                          DeleteClipCommand, AddClipCommand, SplitClipCommand,
                          DuplicateClipCommand, InspectorChangeCommand,
                          SeparateAudioCommand, ToggleLinkCommand,
                          RippleDeleteClipCommand, SlipClipCommand, SlideClipCommand)
    _HAS_COMMANDS = True
except ImportError:
    _HAS_COMMANDS = False

try:
    from project_io import save_project, load_project, get_recent_projects, add_recent_project, RECENT_PROJECTS_KEY
    _HAS_PROJECT_IO = True
except ImportError:
    _HAS_PROJECT_IO = False

try:
    from color_grading import ColorGradingPanel
    _HAS_COLOR_GRADING = True
except ImportError:
    _HAS_COLOR_GRADING = False

try:
    from export_queue import ExportQueueWindow, ExportJob
    _HAS_EXPORT_QUEUE = True
except ImportError:
    _HAS_EXPORT_QUEUE = False

try:
    from title_generator import TitleDialog
    _HAS_TITLE = True
except ImportError:
    _HAS_TITLE = False

try:
    from keyframes import KeyframeEditor
    _HAS_KEYFRAMES = True
except ImportError:
    _HAS_KEYFRAMES = False

try:
    from smart_features import SubtitleDialog, SceneDetectDialog, SmartReframeDialog
    _HAS_SMART = True
except ImportError:
    _HAS_SMART = False

try:
    from ai_features import BeatDetector, NoiseReducer
    _HAS_AI = True
except ImportError:
    _HAS_AI = False

try:
    from proxy_worker import proxy_worker as _proxy_worker, FrameQueue as _FrameQueue
    _HAS_PROXY_WORKER = True
except ImportError:
    _HAS_PROXY_WORKER = False

try:
    from clip_data import make_clip, validate_clip, clip_duration, clip_end_time
    _HAS_CLIP_DATA = True
except ImportError:
    _HAS_CLIP_DATA = False

try:
    from effects_registry import registry as _effects_registry
    _HAS_EFFECTS = True
except ImportError:
    _HAS_EFFECTS = False

try:
    from preferences import PreferencesDialog
    _HAS_PREFS = True
except ImportError:
    _HAS_PREFS = False

try:
    from video_scopes import VideoScopesWindow
    _HAS_VIDEO_SCOPES = True
except ImportError:
    _HAS_VIDEO_SCOPES = False

try:
    from autosave import AutoSaver, check_autosave_recovery, clear_autosave
    _HAS_AUTOSAVE = True
except ImportError:
    _HAS_AUTOSAVE = False

try:
    from crash_reporter import install_crash_handler
    _HAS_CRASH_REPORTER = True
except ImportError:
    _HAS_CRASH_REPORTER = False

try:
    from version_check import check_for_updates, get_current_version
    _HAS_VERSION_CHECK = True
except ImportError:
    _HAS_VERSION_CHECK = False

try:
    from engine import match_clip_colors
    _HAS_COLOR_MATCH = True
except ImportError:
    _HAS_COLOR_MATCH = False

try:
    from transitions import TRANSITION_TYPES
    _HAS_TRANSITIONS = True
except ImportError:
    TRANSITION_TYPES = ["None", "Cross Dissolve", "Fade to Black", "Wipe Left", "Wipe Right"]
    _HAS_TRANSITIONS = False

try:
    from crop_tool import CropAspectDialog
    _HAS_CROP_TOOL = True
except ImportError:
    _HAS_CROP_TOOL = False

setup_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------
class _Tooltip:
    """Simple hover tooltip for any widget."""
    def __init__(self, widget, text):
        self._tip = None
        widget.bind("<Enter>", lambda e: self._show(widget, text))
        widget.bind("<Leave>", lambda e: self._hide())

    def _show(self, widget, text):
        x = widget.winfo_rootx() + widget.winfo_width() // 2
        y = widget.winfo_rooty() + widget.winfo_height() + 4
        self._tip = tk.Toplevel(widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=text, bg="#1e1e2e", fg="#cccccc",
                 font=("Segoe UI", 8), padx=6, pady=3,
                 relief="flat", bd=1).pack()

    def _hide(self):
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ---------------------------------------------------------------------------
# Sequence Settings dialog
# ---------------------------------------------------------------------------
class _SequenceSettingsDialog(ctk.CTkToplevel):
    PRESETS = {
        "16:9 - 1080p Full HD (1920×1080)": (1920, 1080),
        "9:16 - Vertical / Shorts / TikTok (1080×1920)": (1080, 1920),
        "1:1 - Square Post (1080×1080)": (1080, 1080),
        "4:5 - Social Portrait (1080×1350)": (1080, 1350),
        "21:9 - Cinematic Ultrawide (2560×1080)": (2560, 1080),
        "16:9 - 4K Ultra HD (3840×2160)": (3840, 2160),
        "4:3 - Classic Standard (1440×1080)": (1440, 1080),
        "Custom": None,
    }

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Sequence Settings")
        self.geometry("380x280")
        self.resizable(False, False)
        self.grab_set()

        def row(lbl, widget_cls, **kw):
            f = ctk.CTkFrame(self, fg_color="transparent")
            f.pack(fill="x", padx=20, pady=4)
            ctk.CTkLabel(f, text=lbl, width=110, anchor="w", font=("Segoe UI", 11, "bold")).pack(side="left")
            w = widget_cls(f, **kw)
            w.pack(side="right", fill="x", expand=True)
            return w

        self.var_preset = ctk.StringVar(value="Custom")
        for pname, dims in self.PRESETS.items():
            if dims and dims == (app.proj_w, app.proj_h):
                self.var_preset.set(pname)
                break

        self.var_w   = ctk.StringVar(value=str(app.proj_w))
        self.var_h   = ctk.StringVar(value=str(app.proj_h))
        self.var_fps = ctk.StringVar(value=str(app.fps))

        def _on_preset_selected(choice):
            dims = self.PRESETS.get(choice)
            if dims:
                self.var_w.set(str(dims[0]))
                self.var_h.set(str(dims[1]))

        self.cmb_preset = row("Canvas Preset:", ctk.CTkComboBox,
                              variable=self.var_preset,
                              values=list(self.PRESETS.keys()),
                              command=_on_preset_selected)

        self.ent_w   = row("Width px:", ctk.CTkEntry, textvariable=self.var_w)
        self.ent_h   = row("Height px:", ctk.CTkEntry, textvariable=self.var_h)
        self.cmb_fps = row("FPS:", ctk.CTkComboBox,
                           variable=self.var_fps, values=["23.976", "24", "25", "29.97", "30", "50", "59.94", "60"])

        ctk.CTkButton(self, text="Apply Changes", command=self._apply,
                      font=("Segoe UI", 11, "bold"), fg_color="#16a34a", hover_color="#15803d").pack(pady=16, padx=20, fill="x")

    def _apply(self):
        try:
            w, h = int(self.var_w.get()), int(self.var_h.get())
            fps  = float(self.var_fps.get())
            if w <= 0 or h <= 0:
                raise ValueError("Dimensions must be positive")
            self.app.proj_w, self.app.proj_h = w, h
            self.app.fps = fps
            self.app.lbl_seq.configure(text=f"{w}×{h}  {fps}fps")
            self.app.rebuild_proxy_cache()
            self.app.timeline.draw()
            if hasattr(self.app, 'show_toast'):
                self.app.show_toast(f"Sequence set to {w}×{h} ({fps} fps)", success=True)
            self.destroy()
        except ValueError:
            messagebox.showerror("Invalid", "Enter valid positive numbers for dimensions and fps.", parent=self)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# DnD-aware root: use TkinterDnD.Tk when available, else plain CTk
# ---------------------------------------------------------------------------
if _HAS_DND:
    class _DnDBase(ctk.CTk, TkinterDnD.DnDWrapper):
        """Tk root that supports both TkinterDnD and CustomTkinter."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                self.TkdndVersion = TkinterDnD._require(self)
            except Exception as exc:
                logging.getLogger(__name__).warning("Failed to initialize TkinterDnD: %s", exc)
                global _HAS_DND
                _HAS_DND = False

    _AppBase = _DnDBase
else:
    _AppBase = ctk.CTk

def get_media_info_summary(filepath: str) -> dict:
    """Extract metadata summary (type, size, resolution, fps, duration, audio) from a media file."""
    if not filepath or not os.path.isfile(filepath):
        return {}
    ext = os.path.splitext(filepath)[1].lower()
    try:
        size_bytes = os.path.getsize(filepath)
        size_str = f"{size_bytes / (1024*1024):.1f} MB" if size_bytes < 1024*1024*1024 else f"{size_bytes / (1024*1024*1024):.2f} GB"
    except Exception:
        size_str = "—"

    info = {
        'filename':   os.path.basename(filepath),
        'size':       size_str,
        'type':       'Unknown',
        'resolution': '—',
        'fps':        '—',
        'duration':   '—',
        'audio':      '—',
    }

    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'):
        info['type'] = 'Image'
        try:
            with Image.open(filepath) as im:
                info['resolution'] = f"{im.width}×{im.height}"
        except Exception:
            pass
    elif ext in ('.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'):
        info['type'] = 'Audio'
        try:
            import soundfile as sf
            with sf.SoundFile(filepath) as s:
                dur_sec = len(s) / float(s.samplerate)
                mins, secs = divmod(int(dur_sec), 60)
                hrs, mins = divmod(mins, 60)
                info['duration'] = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                info['audio'] = f"{s.channels} ch • {s.samplerate} Hz"
        except Exception:
            pass
    else:
        info['type'] = 'Video'
        try:
            import cv2
            cap = cv2.VideoCapture(filepath)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if w > 0 and h > 0:
                info['resolution'] = f"{w}×{h}"
                info['fps'] = f"{fps:.2f} fps" if fps % 1 != 0 else f"{int(fps)} fps"
            if count > 0 and fps > 0:
                dur_sec = count / fps
                mins, secs = divmod(int(dur_sec), 60)
                hrs, mins = divmod(mins, 60)
                info['duration'] = f"{hrs:02d}:{mins:02d}:{secs:02d}"
        except Exception:
            pass
DEFAULT_TRANSPORT_LAYOUT = {
    "pod_order": ["transport", "timecode", "vumeter", "tools"],
    "alignment": "space_between",
    "vu_width": 340,
    "show_headers": True,
    "visible_pods": {
        "transport": True,
        "timecode": True,
        "vumeter": True,
        "tools": True
    }
}


# ---------------------------------------------------------------------------
# Transport Customizer Dialog
# ---------------------------------------------------------------------------
class _TransportCustomizerDialog(ctk.CTkToplevel):
    """Interactive customization dialog for adjusting the transport bar layout and modular pods."""
    POD_NAMES = {
        "transport": "⏯ Transport Controls (Play/Pause, Step, Shuttle, Loop)",
        "timecode": "⏱ Digital Timecode & Sequence Info Display",
        "vumeter": "📊 Master Stereo VU Level Meter",
        "tools": "🎛️ Studio Tools (Fullscreen, Scopes, Mixer, Layout)",
    }

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Customize Transport Bar Layout")
        self.geometry("520x600")
        self.resizable(False, False)
        self.grab_set()

        # Local working copy of layout
        self.layout_data = copy.deepcopy(getattr(app, "transport_layout", DEFAULT_TRANSPORT_LAYOUT))
        self._build_ui()

    def _build_ui(self):
        # Header title
        hdr = ctk.CTkFrame(self, fg_color="#182030", height=45, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="⚙️  Transport Bar Customization", font=("Segoe UI", 13, "bold"), text_color="#f1f5f9").pack(side="left", padx=16, pady=10)

        main_f = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main_f.pack(fill="both", expand=True, padx=16, pady=10)

        # Section 1: Modular Pod Order & Visibility
        ctk.CTkLabel(main_f, text="Modular Pods (Order & Visibility):", font=("Segoe UI", 11, "bold"), text_color="#38bdf8").pack(anchor="w", pady=(4, 6))

        self.pod_list_frame = ctk.CTkFrame(main_f, fg_color="#121622", border_width=1, border_color="#242c40", corner_radius=8)
        self.pod_list_frame.pack(fill="x", pady=(0, 12))
        self._refresh_pod_list()

        # Section 2: Bar Alignment
        ctk.CTkLabel(main_f, text="Transport Bar Alignment:", font=("Segoe UI", 11, "bold"), text_color="#38bdf8").pack(anchor="w", pady=(6, 4))
        self.align_var = ctk.StringVar(value=self.layout_data.get("alignment", "space_between"))

        align_box = ctk.CTkFrame(main_f, fg_color="#121622", border_width=1, border_color="#242c40", corner_radius=8)
        align_box.pack(fill="x", pady=(0, 12), padx=2)

        align_opts = [
            ("space_between", "Distributed (Space Between)"),
            ("center", "Centered in Frame"),
            ("left", "Left Aligned"),
            ("right", "Right Aligned"),
        ]
        for val, lbl in align_opts:
            rb = ctk.CTkRadioButton(align_box, text=lbl, value=val, variable=self.align_var,
                                    command=self._on_alignment_changed, font=("Segoe UI", 11))
            rb.pack(anchor="w", padx=14, pady=6)

        # Section 3: VU Meter Width
        ctk.CTkLabel(main_f, text="VU Meter Display Width:", font=("Segoe UI", 11, "bold"), text_color="#38bdf8").pack(anchor="w", pady=(6, 4))
        self.vu_width_var = ctk.IntVar(value=self.layout_data.get("vu_width", 340))

        vu_box = ctk.CTkFrame(main_f, fg_color="#121622", border_width=1, border_color="#242c40", corner_radius=8)
        vu_box.pack(fill="x", pady=(0, 12), padx=2)

        vu_opts = [
            (240, "Compact Studio Meter (240px)"),
            (340, "Standard Studio Meter (340px)"),
            (460, "Expanded Pro Mastering Meter (460px)"),
        ]
        for val, lbl in vu_opts:
            rb = ctk.CTkRadioButton(vu_box, text=lbl, value=val, variable=self.vu_width_var,
                                    command=self._on_vu_width_changed, font=("Segoe UI", 11))
            rb.pack(anchor="w", padx=14, pady=6)

        # Section 4: Pod Header Bar & Shift Arrows
        self.show_hdr_var = ctk.BooleanVar(value=self.layout_data.get("show_headers", True))
        cb_hdr = ctk.CTkCheckBox(main_f, text="Show Pod Header Bar with Quick-Shift Arrows (◀ / ▶)",
                                 variable=self.show_hdr_var, command=self._on_header_toggle,
                                 font=("Segoe UI", 11, "bold"))
        cb_hdr.pack(anchor="w", pady=(4, 12))

        # Bottom buttons
        btn_bar = ctk.CTkFrame(self, fg_color="#121622", height=50)
        btn_bar.pack(fill="x", side="bottom")

        ctk.CTkButton(btn_bar, text="↺ Reset Defaults", width=120, fg_color="#334155", hover_color="#475569",
                      command=self._reset_defaults).pack(side="left", padx=16, pady=10)

        ctk.CTkButton(btn_bar, text="Done / Close", width=110, fg_color="#0284c7", hover_color="#0369a1",
                      command=self.destroy).pack(side="right", padx=16, pady=10)

    def _refresh_pod_list(self):
        for w in self.pod_list_frame.winfo_children():
            w.destroy()

        order = self.layout_data.get("pod_order", ["transport", "timecode", "vumeter", "tools"])
        vis = self.layout_data.get("visible_pods", {})

        for idx, pod_id in enumerate(order):
            row = ctk.CTkFrame(self.pod_list_frame, fg_color="#171c2b" if idx % 2 == 0 else "#141824", corner_radius=4)
            row.pack(fill="x", padx=6, pady=3)

            is_vis = ctk.BooleanVar(value=vis.get(pod_id, True))
            cb = ctk.CTkCheckBox(row, text=self.POD_NAMES.get(pod_id, pod_id.title()),
                                 variable=is_vis,
                                 command=lambda pid=pod_id, var=is_vis: self._toggle_pod_vis(pid, var.get()),
                                 font=("Segoe UI", 10, "bold"), width=280)
            cb.pack(side="left", padx=10, pady=6)

            # Move up button
            btn_up = ctk.CTkButton(row, text="▲", width=28, height=24, fg_color="#242c40", hover_color="#334155",
                                   command=lambda i=idx: self._move_pod(i, -1))
            btn_up.pack(side="right", padx=(2, 8), pady=4)
            if idx == 0:
                btn_up.configure(state="disabled")

            # Move down button
            btn_down = ctk.CTkButton(row, text="▼", width=28, height=24, fg_color="#242c40", hover_color="#334155",
                                     command=lambda i=idx: self._move_pod(i, 1))
            btn_down.pack(side="right", padx=2, pady=4)
            if idx == len(order) - 1:
                btn_down.configure(state="disabled")

    def _move_pod(self, index, delta):
        order = self.layout_data.get("pod_order", [])
        new_idx = index + delta
        if 0 <= new_idx < len(order):
            order[index], order[new_idx] = order[new_idx], order[index]
            self.layout_data["pod_order"] = order
            self._apply_and_save()
            self._refresh_pod_list()

    def _toggle_pod_vis(self, pod_id, is_visible):
        self.layout_data.setdefault("visible_pods", {})[pod_id] = is_visible
        self._apply_and_save()

    def _on_alignment_changed(self):
        self.layout_data["alignment"] = self.align_var.get()
        self._apply_and_save()

    def _on_vu_width_changed(self):
        self.layout_data["vu_width"] = self.vu_width_var.get()
        self._apply_and_save()

    def _on_header_toggle(self):
        self.layout_data["show_headers"] = self.show_hdr_var.get()
        self._apply_and_save()

    def _reset_defaults(self):
        self.layout_data = copy.deepcopy(DEFAULT_TRANSPORT_LAYOUT)
        self.align_var.set(self.layout_data["alignment"])
        self.vu_width_var.set(self.layout_data["vu_width"])
        self.show_hdr_var.set(self.layout_data["show_headers"])
        self._apply_and_save()
        self._refresh_pod_list()

    def _apply_and_save(self):
        self.app.transport_layout = copy.deepcopy(self.layout_data)
        config_mgr.save_config("transport_layout", self.app.transport_layout)
        if hasattr(self.app, "vu_meter") and self.app.vu_meter:
            self.app.vu_meter.set_width(self.app.transport_layout.get("vu_width", 340))
        self.app._rebuild_transport_bar()


class StereoVUMeter(tk.Canvas):
    """
    Real-time studio stereo volume level meter displaying Left and Right channel LED bars
    with comprehensive dB graticule scale, green -> amber -> red clipping gradients, and peak hold lines.
    """
    def __init__(self, parent, width=340, height=50, bg="#0b0e14", **kwargs):
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=1, highlightbackground="#242c40",
                         relief="flat", bd=0, **kwargs)
        self.w = width
        self.h = height
        self.peak_l = 0.0
        self.peak_r = 0.0
        self.disp_l = 0.0
        self.disp_r = 0.0
        self._decay_rate = 0.045
        self._draw_meter()
        self._decay_loop()

    def set_width(self, width: int):
        """Dynamically resize the meter canvas width."""
        self.w = max(180, int(width))
        self.configure(width=self.w)
        self._draw_meter()

    def set_levels(self, l_level: float, r_level: float):
        """Update instantaneous peak levels [0.0, 1.2]."""
        self.peak_l = max(0.0, min(1.2, float(l_level)))
        self.peak_r = max(0.0, min(1.2, float(r_level)))
        if self.peak_l > self.disp_l:
            self.disp_l = self.peak_l
        if self.peak_r > self.disp_r:
            self.disp_r = self.peak_r
        self._draw_meter()

    def reset(self):
        """Reset meters to zero level."""
        self.peak_l = 0.0
        self.peak_r = 0.0
        self.disp_l = 0.0
        self.disp_r = 0.0
        self._draw_meter()

    def _decay_loop(self):
        if not self.winfo_exists():
            return
        if self.disp_l > 0.001 or self.disp_r > 0.001:
            self.disp_l = max(0.0, self.disp_l - self._decay_rate)
            self.disp_r = max(0.0, self.disp_r - self._decay_rate)
            self._draw_meter()
        self.after(35, self._decay_loop)

    def _draw_meter(self):
        self.delete("all")
        bar_x1 = 30
        bar_x2 = self.w - 8
        bar_w = max(10, bar_x2 - bar_x1)

        # Comprehensive studio dB graticule scale ticks and labels along top
        ticks = [
            (0.05, "-48"),
            (0.20, "-36"),
            (0.38, "-24"),
            (0.52, "-18"),
            (0.66, "-12"),
            (0.80, "-6"),
            (0.88, "-3"),
            (0.94, "0"),
            (1.00, "+3"),
        ]
        for frac, label in ticks:
            tx = bar_x1 + int(bar_w * frac)
            self.create_line(tx, 1, tx, 5, fill="#4a5568", width=1)
            self.create_text(tx, 8, text=label, fill="#8899aa", font=("Segoe UI", 7, "bold"))

        # Channel labels (L and R in glowing cyan)
        self.create_text(14, 21, text="L", fill="#00d2ff", font=("Segoe UI", 10, "bold"))
        self.create_rectangle(bar_x1, 15, bar_x2, 28, fill="#07090e", outline="#1e2536", width=1)

        self.create_text(14, 38, text="R", fill="#00d2ff", font=("Segoe UI", 10, "bold"))
        self.create_rectangle(bar_x1, 32, bar_x2, 45, fill="#07090e", outline="#1e2536", width=1)

        def draw_channel_bar(val, y1, y2):
            if val <= 0.001:
                return
            fill_w = int(bar_w * min(1.0, val))
            green_w = min(fill_w, int(bar_w * 0.68))
            if green_w > 0:
                self.create_rectangle(bar_x1, y1, bar_x1 + green_w, y2, fill="#00e676", outline="")
            if fill_w > int(bar_w * 0.68):
                yellow_w = min(fill_w, int(bar_w * 0.88)) - int(bar_w * 0.68)
                if yellow_w > 0:
                    self.create_rectangle(bar_x1 + int(bar_w * 0.68), y1,
                                          bar_x1 + int(bar_w * 0.68) + yellow_w, y2,
                                          fill="#ffb300", outline="")
            if fill_w > int(bar_w * 0.88):
                red_w = fill_w - int(bar_w * 0.88)
                if red_w > 0:
                    self.create_rectangle(bar_x1 + int(bar_w * 0.88), y1,
                                          bar_x1 + fill_w, y2,
                                          fill="#ff1744", outline="")

            # Authentic hardware LED segment dividers
            for nx in range(bar_x1 + 5, bar_x1 + fill_w, 6):
                self.create_line(nx, y1, nx, y2, fill="#07090e", width=1.5)

            # Peak hold needle
            peak_x = min(bar_x2 - 1, bar_x1 + int(bar_w * min(1.0, val)))
            self.create_line(peak_x, y1, peak_x, y2, fill="#ffffff", width=2.5)

        draw_channel_bar(self.disp_l, 15, 28)
        draw_channel_bar(self.disp_r, 32, 45)


class _MediaListboxCompat:
    """Compatibility wrapper for external code or tests expecting listbox methods."""
    def __init__(self, app):
        self._app = app
    def curselection(self):
        if getattr(self._app, "selected_media_index", None) is not None and 0 <= self._app.selected_media_index < len(self._app.media_bin):
            return (self._app.selected_media_index,)
        return ()
    def selection_clear(self, *args):
        self._app.selected_media_index = None
    def selection_set(self, idx):
        self._app.selected_media_index = int(idx)
    def insert(self, *args):
        pass
    def delete(self, *args):
        pass
    def drop_target_register(self, *args):
        pass
    def dnd_bind(self, *args):
        pass


class VideoStudioApp(_AppBase):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode(config_mgr.get("theme") or "Dark")
        ctk.set_default_color_theme(config_mgr.get("accent_color") or "blue")
        self.title("Video Studio")

        # Set Window and Taskbar Icon
        _icon_ico = os.path.join(_SCRIPT_DIR, "assets", "icon.ico")
        _icon_png = os.path.join(_SCRIPT_DIR, "assets", "icon.png")
        if os.path.exists(_icon_ico):
            try:
                self.iconbitmap(default=_icon_ico)
            except Exception:
                try:
                    self.iconbitmap(_icon_ico)
                except Exception:
                    pass
        if os.path.exists(_icon_png):
            try:
                self._app_icon_photo = ImageTk.PhotoImage(file=_icon_png)
                self.wm_iconphoto(True, self._app_icon_photo)
            except Exception:
                pass

        geom = config_mgr.get("geometry")
        self.geometry(geom if geom else "1500x940")
        self.minsize(1200, 750)

        # Project state
        self.project_clips     = []
        self.media_bin         = []          # list of file paths
        self.selected_media_index = None
        self._thumbnail_cache  = {}
        self.master_proxy_clip = None
        self.open_proxy_clips  = []
        self.proj_w, self.proj_h = 1920, 1080
        self.proxy_w, self.proxy_h = 640, 360
        self.fps               = 30.0
        self.is_playing        = False
        self.playhead_t        = 0.0
        self.current_task      = None
        self._play_after_id    = None        # for cancelling pending after() calls
        self._current_project_path = None
        self._export_queue_window  = None    # lazy singleton
        self._beat_markers: list   = []      # beat timestamps for snap
        self._fs_window            = None    # full-screen preview window
        self._proxy_building       = False   # async proxy build in flight

        # Command stack (undo/redo)
        if _HAS_COMMANDS:
            self.cmd_stack = CommandStack(self)
        else:
            self.cmd_stack = None

        # Auto-saver
        self._autosaver = None
        if _HAS_AUTOSAVE:
            self._autosaver = AutoSaver(self)
            self._autosaver.start()

        # FrameQueue for smooth playback (created once, reused per play session)
        if _HAS_PROXY_WORKER:
            self._frame_queue = _FrameQueue(maxsize=8)
        else:
            self._frame_queue = None

        self._scopes_window = None

        # Dynamic modular transport bar state
        self.is_looping = False
        self.is_muted   = False
        saved_tl = config_mgr.get("transport_layout")
        if isinstance(saved_tl, dict):
            self.transport_layout = {**DEFAULT_TRANSPORT_LAYOUT, **saved_tl}
            if "visible_pods" in saved_tl:
                self.transport_layout["visible_pods"] = {**DEFAULT_TRANSPORT_LAYOUT["visible_pods"], **saved_tl.get("visible_pods", {})}
        else:
            self.transport_layout = copy.deepcopy(DEFAULT_TRANSPORT_LAYOUT)

        self._build_menu()
        self._build_ui()
        self._bind_keys()
        self.update_inspector()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Post-startup tasks (run after mainloop starts)
        self.after(500, self._post_startup)

    # ------------------------------------------------------------------
    # After-command hook (called by CommandStack after every push/undo/redo)
    # ------------------------------------------------------------------
    def _after_command(self):
        self.rebuild_proxy_cache()
        self.timeline.draw()
        self.update_inspector()
        self._update_undo_menu()

    # ------------------------------------------------------------------
    # Undo menu state
    # ------------------------------------------------------------------
    def _update_undo_menu(self):
        if not hasattr(self, '_edit_menu') or self.cmd_stack is None:
            return
        em = self._edit_menu
        # Undo entry is index 0, Redo is index 1
        can_undo = self.cmd_stack.can_undo
        can_redo = self.cmd_stack.can_redo
        undo_lbl = self.cmd_stack.undo_label if can_undo else "Undo"
        redo_lbl = self.cmd_stack.redo_label if can_redo else "Redo"
        em.entryconfig(0, label=f"{undo_lbl}          Ctrl+Z",
                       state="normal" if can_undo else "disabled")
        em.entryconfig(1, label=f"{redo_lbl}     Ctrl+Shift+Z",
                       state="normal" if can_redo else "disabled")

    # ------------------------------------------------------------------
    # Toast HUD notification
    # ------------------------------------------------------------------
    def show_toast(self, message: str, success: bool = True, duration_ms: int = 2500):
        """Display a sleek, modern non-blocking HUD toast notification in the UI."""
        try:
            if hasattr(self, '_active_toast') and self._active_toast:
                try:
                    self._active_toast.destroy()
                except Exception:
                    pass
                self._active_toast = None

            bg_color = "#122a1c" if success else "#3a1313"
            border_color = "#00e676" if success else "#ff5252"
            text_color = "#d5fbe3" if success else "#ffd6d6"
            icon = "✔" if success else "✖"

            parent = getattr(self, 'tab_edit', self)
            toast = ctk.CTkFrame(
                parent,
                fg_color=bg_color,
                border_color=border_color,
                border_width=2,
                corner_radius=8
            )
            self._active_toast = toast

            def _dismiss(event=None):
                try:
                    if getattr(self, '_active_toast', None) == toast:
                        toast.destroy()
                        self._active_toast = None
                except Exception:
                    pass

            lbl = ctk.CTkLabel(
                toast,
                text=f"{icon}  {message}",
                text_color=text_color,
                font=("Segoe UI", 11, "bold"),
                wraplength=280,
                justify="center",
                cursor="hand2"
            )
            lbl.pack(padx=16, pady=10)
            lbl.bind("<Button-1>", _dismiss)
            toast.bind("<Button-1>", _dismiss)

            toast.place(relx=0.5, rely=0.93, anchor="center")
            self.after(duration_ms, _dismiss)
        except Exception as e:
            log.warning("Toast display error: %s", e)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _build_menu(self):
        bar = tk.Menu(self, bg="#1e1e1e", fg="white", activebackground="#3a7ebf",
                      activeforeground="white", bd=0, relief="flat")

        _menu_kw = dict(bg="#1e1e1e", fg="white",
                        activebackground="#3a7ebf", activeforeground="white",
                        tearoff=0)

        # ── File ──
        fm = tk.Menu(bar, **_menu_kw)
        fm.add_command(label="New Project              Ctrl+N",  command=self.new_project)
        fm.add_command(label="Open Project…            Ctrl+O",  command=self.open_project)
        fm.add_command(label="Save Project             Ctrl+S",  command=self.save_project)
        fm.add_command(label="Save Project As…    Ctrl+Shift+S", command=self.save_project_as)
        fm.add_separator()

        # Recent Projects submenu
        if _HAS_PROJECT_IO:
            recent = get_recent_projects(config_mgr)
        else:
            recent = []
        recents_menu = tk.Menu(fm, **_menu_kw)
        for rp in recent:
            recents_menu.add_command(label=os.path.basename(rp),
                                     command=lambda p=rp: self.open_project(p))
        if not recent:
            recents_menu.add_command(label="(none)", state="disabled")
        fm.add_cascade(label="Recent Projects", menu=recents_menu)
        fm.add_separator()

        fm.add_command(label="Import Media…            Ctrl+I",  command=self.import_media)
        fm.add_command(label="Import Sequence (EDL / FCP7 XML)…", command=self.import_sequence)
        fm.add_command(label="Export Sequence (EDL / FCP7 XML)…", command=self.export_sequence)
        fm.add_separator()
        fm.add_command(label="Sequence Settings…",              command=self._open_seq_settings)
        fm.add_separator()
        fm.add_command(label="Export Queue…",                   command=self.open_export_queue)
        fm.add_command(label="Render Project…",                 command=self.render_project)
        fm.add_separator()
        fm.add_command(label="Quit                     Ctrl+Q", command=self._on_close)
        bar.add_cascade(label="File", menu=fm)

        # ── Edit ──
        em = tk.Menu(bar, **_menu_kw)
        em.add_command(label="Undo                     Ctrl+Z",
                       command=self._undo, state="disabled")
        em.add_command(label="Redo               Ctrl+Shift+Z",
                       command=self._redo, state="disabled")
        em.add_separator()
        em.add_command(label="Delete Active Clip          Del", command=self._kb_delete)
        em.add_command(label="Split at Playhead             S",
                       command=lambda: self.split_clip_at_playhead(self.timeline.active_clip_id))
        em.add_command(label="Duplicate Clip                D",
                       command=lambda: self.duplicate_clip(self.timeline.active_clip_id))
        em.add_separator()
        em.add_command(label="Crop & Aspect Ratio…        C", command=self.open_crop_tool)
        bar.add_cascade(label="Edit", menu=em)
        self._edit_menu = em

        # ── AI ──
        ai_m = tk.Menu(bar, **_menu_kw)
        ai_m.add_command(label="Auto-Subtitle Active Clip…",
                         command=self._ai_subtitle)
        ai_m.add_command(label="Detect Scenes & Split…",
                         command=self._ai_scene_detect)
        ai_m.add_command(label="Smart Reframe (9:16)…",
                         command=self._ai_smart_reframe)
        ai_m.add_command(label="Crop & Aspect Ratio Tool…",
                         command=self.open_crop_tool)
        ai_m.add_separator()
        ai_m.add_command(label="Detect Beat Markers…",
                         command=self._ai_beat_markers)
        ai_m.add_command(label="AI Noise Reduction on Active Clip…",
                         command=self._ai_noise_reduce)
        ai_m.add_separator()
        ai_m.add_command(label="Colour Match to Active Clip…",
                         command=self._ai_color_match)
        bar.add_cascade(label="AI", menu=ai_m)

        # ── View ──
        vm = tk.Menu(bar, **_menu_kw)
        vm.add_command(label="Zoom In              =",  command=lambda: self.timeline._zoom_in())
        vm.add_command(label="Zoom Out             -",  command=lambda: self.timeline._zoom_out())
        vm.add_command(label="Zoom Reset",              command=lambda: self.timeline._zoom_reset() if hasattr(self.timeline, '_zoom_reset') else None)
        vm.add_separator()
        vm.add_command(label="Full-Screen Preview  ~",  command=self._toggle_fullscreen_preview)
        vm.add_separator()
        vm.add_command(label="Studio Video Scopes…",    command=self.open_video_scopes)
        vm.add_separator()
        vm.add_command(label="Preferences…      Ctrl+,", command=self._open_preferences)
        vm.add_command(label="Toggle Console",          command=lambda: toggle_console(True))
        bar.add_cascade(label="View", menu=vm)

        # ── Titles ──
        tm = tk.Menu(bar, **_menu_kw)
        tm.add_command(label="New Title Clip…", command=self._new_title_clip)
        bar.add_cascade(label="Titles", menu=tm)

        # ── Audio ──
        audio_m = tk.Menu(bar, **_menu_kw)
        audio_m.add_command(label="Studio Audio FX Rack & VSTs…", command=self.open_audio_rack)
        audio_m.add_separator()
        audio_m.add_command(label="AI Noise Reduction on Active Clip…", command=self._ai_noise_reduce)
        audio_m.add_command(label="Detect Beat Markers…", command=self._ai_beat_markers)
        bar.add_cascade(label="Audio", menu=audio_m)

        self.configure(menu=bar)

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ── Outer grid layout ──────────────────────────────────────────
        self.grid_columnconfigure(0, weight=3)  # Top-left: Preview viewer
        self.grid_columnconfigure(1, weight=1)  # Top-right: Tabbed panels
        self.grid_rowconfigure(0, weight=3)     # Top row (Preview + Tabs)
        self.grid_rowconfigure(1, weight=2)     # Bottom row (Timeline)
        self.grid_rowconfigure(2, weight=0)     # Footer (Status bar)

        self._build_preview()
        self._build_panels()
        self._build_timeline_area()
        self._build_statusbar()

    def _build_preview(self):
        pf = ctk.CTkFrame(self, fg_color="#0f1219", corner_radius=10, border_width=1, border_color="#21283b")
        pf.grid(row=0, column=0, padx=(10, 5), pady=(10, 5), sticky="nsew")
        pf.grid_rowconfigure(0, weight=1)
        pf.grid_columnconfigure(0, weight=1)
        self.preview_frame = pf

        self.lbl_preview = ctk.CTkLabel(pf, text="No Sequence Loaded", fg_color="black",
                                         font=("Segoe UI", 15, "bold"), text_color="#475569")
        self.lbl_preview.grid(row=0, column=0, padx=6, pady=(6, 2), sticky="nsew")

        # Top scrubber bar & timeline ribbon
        scrub_box = ctk.CTkFrame(pf, fg_color="#131722", corner_radius=6, border_width=1, border_color="#222a3d")
        scrub_box.grid(row=1, column=0, padx=6, pady=(3, 2), sticky="ew")
        scrub_box.grid_columnconfigure(0, weight=1)

        self.scrub_var = tk.DoubleVar(value=0.0)
        self.scrub_slider = ctk.CTkSlider(scrub_box, from_=0, to=1, variable=self.scrub_var,
                                           command=self._on_scrub_slider,
                                           height=16, button_color="#38bdf8", button_hover_color="#0ea5e9",
                                           progress_color="#0284c7", fg_color="#1e293b")
        self.scrub_slider.grid(row=0, column=0, padx=8, pady=4, sticky="ew")

        # Dynamic Modular Transport Bar Container
        self.transport_bar_frame = ctk.CTkFrame(pf, fg_color="transparent")
        self.transport_bar_frame.grid(row=2, column=0, padx=4, pady=(2, 6), sticky="ew")
        self.transport_bar_frame.bind("<Button-3>", lambda e: self._show_transport_context_menu(e))

        self._rebuild_transport_bar()

    def _rebuild_transport_bar(self):
        for child in self.transport_bar_frame.winfo_children():
            child.destroy()

        layout = getattr(self, "transport_layout", DEFAULT_TRANSPORT_LAYOUT)
        pod_order = layout.get("pod_order", ["transport", "timecode", "vumeter", "tools"])
        alignment = layout.get("alignment", "space_between")
        vu_w = layout.get("vu_width", 340)
        show_headers = layout.get("show_headers", True)
        visible = layout.get("visible_pods", {})

        # Dock container according to alignment
        if alignment == "center":
            dock = ctk.CTkFrame(self.transport_bar_frame, fg_color="transparent")
            dock.pack(anchor="center")
        elif alignment == "left":
            dock = ctk.CTkFrame(self.transport_bar_frame, fg_color="transparent")
            dock.pack(side="left")
        elif alignment == "right":
            dock = ctk.CTkFrame(self.transport_bar_frame, fg_color="transparent")
            dock.pack(side="right")
        else:  # space_between
            dock = self.transport_bar_frame

        dock.bind("<Button-3>", lambda e: self._show_transport_context_menu(e))

        def create_pod(pod_id, title_text):
            p_frame = ctk.CTkFrame(dock, fg_color="#131724", border_width=1, border_color="#252f44", corner_radius=8)
            p_frame.bind("<Button-3>", lambda e, pid=pod_id: self._show_transport_context_menu(e, pid))
            
            if show_headers:
                h_frame = ctk.CTkFrame(p_frame, fg_color="#192032", height=18, corner_radius=4)
                h_frame.pack(fill="x", padx=4, pady=(3, 2))
                h_frame.pack_propagate(False)
                h_frame.bind("<Button-3>", lambda e, pid=pod_id: self._show_transport_context_menu(e, pid))

                lbl = ctk.CTkLabel(h_frame, text=title_text, font=("Segoe UI", 8, "bold"), text_color="#64748b")
                lbl.pack(side="left", padx=4)
                lbl.bind("<Button-3>", lambda e, pid=pod_id: self._show_transport_context_menu(e, pid))

                btn_r = ctk.CTkButton(h_frame, text="▶", width=16, height=14, fg_color="transparent",
                                      hover_color="#29354e", text_color="#94a3b8", font=("Segoe UI", 7),
                                      command=lambda pid=pod_id: self._shift_pod(pid, 1))
                btn_r.pack(side="right", padx=1)
                _Tooltip(btn_r, "Shift Pod Right")

                btn_l = ctk.CTkButton(h_frame, text="◀", width=16, height=14, fg_color="transparent",
                                      hover_color="#29354e", text_color="#94a3b8", font=("Segoe UI", 7),
                                      command=lambda pid=pod_id: self._shift_pod(pid, -1))
                btn_l.pack(side="right", padx=1)
                _Tooltip(btn_l, "Shift Pod Left")

            c_frame = ctk.CTkFrame(p_frame, fg_color="transparent")
            c_frame.pack(padx=6, pady=(2 if show_headers else 6, 6), fill="both", expand=True)
            c_frame.bind("<Button-3>", lambda e, pid=pod_id: self._show_transport_context_menu(e, pid))
            return p_frame, c_frame

        # Render pods in configured order
        for pod_id in pod_order:
            if not visible.get(pod_id, True):
                continue

            if pod_id == "transport":
                pf, cf = create_pod("transport", "TRANSPORT")
                
                self.btn_rw = ctk.CTkButton(cf, text="⏮", width=36, height=34, fg_color="transparent",
                                            font=("Segoe UI", 13), hover_color="#222b3d", text_color="#cbd5e1",
                                            command=self._kb_goto_start)
                self.btn_rw.pack(side="left", padx=1)
                _Tooltip(self.btn_rw, "Go to Start (Home)")

                self.btn_shuttle_b = ctk.CTkButton(cf, text="⏪", width=36, height=34, fg_color="transparent",
                                                   font=("Segoe UI", 12), hover_color="#222b3d", text_color="#cbd5e1",
                                                   command=lambda: self._shuttle(-2))
                self.btn_shuttle_b.pack(side="left", padx=1)
                _Tooltip(self.btn_shuttle_b, "Shuttle Rewind (J)")

                self.btn_step_b = ctk.CTkButton(cf, text="◀|", width=36, height=34, fg_color="transparent",
                                                font=("Segoe UI", 12), hover_color="#222b3d", text_color="#cbd5e1",
                                                command=self._step_back)
                self.btn_step_b.pack(side="left", padx=1)
                _Tooltip(self.btn_step_b, "Step Back 1 Frame (←)")

                play_txt = "⏸ Pause" if self.is_playing else "▶ Play"
                self.btn_play = ctk.CTkButton(cf, text=play_txt, width=108, height=36,
                                              font=("Segoe UI", 13, "bold"),
                                              fg_color="#0d9488", hover_color="#0f766e", text_color="#ffffff",
                                              command=self.toggle_play)
                self.btn_play.pack(side="left", padx=4)
                _Tooltip(self.btn_play, "Play / Pause (Space / K)")

                self.btn_step_f = ctk.CTkButton(cf, text="|▶", width=36, height=34, fg_color="transparent",
                                                font=("Segoe UI", 12), hover_color="#222b3d", text_color="#cbd5e1",
                                                command=self._step_forward)
                self.btn_step_f.pack(side="left", padx=1)
                _Tooltip(self.btn_step_f, "Step Forward 1 Frame (→)")

                self.btn_shuttle_f = ctk.CTkButton(cf, text="⏩", width=36, height=34, fg_color="transparent",
                                                   font=("Segoe UI", 12), hover_color="#222b3d", text_color="#cbd5e1",
                                                   command=lambda: self._shuttle(2))
                self.btn_shuttle_f.pack(side="left", padx=1)
                _Tooltip(self.btn_shuttle_f, "Shuttle Forward (L)")

                self.btn_ff = ctk.CTkButton(cf, text="⏭", width=36, height=34, fg_color="transparent",
                                            font=("Segoe UI", 13), hover_color="#222b3d", text_color="#cbd5e1",
                                            command=self._kb_goto_end)
                self.btn_ff.pack(side="left", padx=1)
                _Tooltip(self.btn_ff, "Go to End (End)")

                loop_color = "#1e3a8a" if getattr(self, "is_looping", False) else "transparent"
                loop_tcolor = "#60a5fa" if getattr(self, "is_looping", False) else "#94a3b8"
                self.btn_loop = ctk.CTkButton(cf, text="🔁", width=34, height=34, fg_color=loop_color,
                                              hover_color="#1d4ed8", text_color=loop_tcolor,
                                              font=("Segoe UI", 12), command=self.toggle_loop)
                self.btn_loop.pack(side="left", padx=(3, 1))
                _Tooltip(self.btn_loop, "Toggle Playback Loop")

                if alignment == "space_between":
                    pf.pack(side="left", padx=4, pady=2, expand=True)
                else:
                    pf.pack(side="left", padx=4, pady=2)

            elif pod_id == "timecode":
                pf, cf = create_pod("timecode", "TIMECODE & SEQUENCE")
                
                # Digital LCD Bezel
                lcd_frame = ctk.CTkFrame(cf, fg_color="#080b12", border_width=1, border_color="#1a2233", corner_radius=6)
                lcd_frame.pack(fill="both", expand=True, padx=2, pady=1)

                top_row = ctk.CTkFrame(lcd_frame, fg_color="transparent")
                top_row.pack(fill="x", padx=6, pady=(3, 0))

                t = self.playhead_t
                h_part, rem = divmod(t, 3600)
                m_part, s_part = divmod(rem, 60)
                tc_str = f"{int(h_part):02d}:{int(m_part):02d}:{s_part:05.2f}"
                self.lbl_time = ctk.CTkLabel(top_row, text=tc_str,
                                             font=("Consolas", 15, "bold"), text_color="#38bdf8")
                self.lbl_time.pack(side="left", padx=(2, 6))
                _Tooltip(self.lbl_time, "Master Playhead Timecode")

                # Sequence info pill
                seq_text = f"{self.proj_w}×{self.proj_h}  {self.fps}fps"
                self.lbl_seq = ctk.CTkButton(top_row, text=seq_text,
                                             fg_color="#1e2638", hover_color="#2d374d",
                                             text_color="#94a3b8", font=("Segoe UI", 9, "bold"),
                                             height=22, corner_radius=11,
                                             command=self._open_seq_settings)
                self.lbl_seq.pack(side="left", padx=4)
                _Tooltip(self.lbl_seq, "Click to change Sequence Settings")

                # In/Out sub-label
                in_val = getattr(getattr(self, "timeline", None), "mark_in", None)
                out_val = getattr(getattr(self, "timeline", None), "mark_out", None)
                inout_txt = f"In: {in_val:.2f}s" if in_val is not None else "In: —"
                inout_txt += "   " + (f"Out: {out_val:.2f}s" if out_val is not None else "Out: —")
                
                self.lbl_inout = ctk.CTkLabel(lcd_frame, text=inout_txt,
                                              text_color="#64748b", font=("Segoe UI", 9, "bold"))
                self.lbl_inout.pack(anchor="w", padx=8, pady=(0, 2))
                _Tooltip(self.lbl_inout, "In / Out Trim Range Markers")

                if alignment == "space_between":
                    pf.pack(side="left", padx=4, pady=2, expand=True)
                else:
                    pf.pack(side="left", padx=4, pady=2)

            elif pod_id == "vumeter":
                pf, cf = create_pod("vumeter", "STEREO VU MONITOR")
                
                # Master Mute Button
                mute_txt = "🔇" if getattr(self, "is_muted", False) else "🔊"
                mute_tcolor = "#ef4444" if getattr(self, "is_muted", False) else "#94a3b8"
                self.btn_master_mute = ctk.CTkButton(cf, text=mute_txt, width=32, height=34,
                                                     fg_color="transparent", hover_color="#222b3d",
                                                     text_color=mute_tcolor, font=("Segoe UI", 12),
                                                     command=self.toggle_master_mute)
                self.btn_master_mute.pack(side="left", padx=(0, 4))
                _Tooltip(self.btn_master_mute, "Mute / Unmute Master Audio")

                self.vu_meter = StereoVUMeter(cf, width=vu_w, height=50)
                self.vu_meter.pack(side="left", padx=2)
                _Tooltip(self.vu_meter, "Master Stereo Volume Level Meter (L / R)")

                if alignment == "space_between":
                    pf.pack(side="left", padx=4, pady=2, expand=True)
                else:
                    pf.pack(side="left", padx=4, pady=2)

            elif pod_id == "tools":
                pf, cf = create_pod("tools", "STUDIO TOOLS")
                
                self.btn_fs = ctk.CTkButton(cf, text="⛶", width=34, height=34, fg_color="transparent",
                                            hover_color="#222b3d", text_color="#cbd5e1", font=("Segoe UI", 13),
                                            command=self._toggle_fullscreen_preview)
                self.btn_fs.pack(side="left", padx=1)
                _Tooltip(self.btn_fs, "Full-Screen Preview (F11 / ~)")

                self.btn_scopes = ctk.CTkButton(cf, text="📊", width=34, height=34, fg_color="transparent",
                                                hover_color="#222b3d", text_color="#cbd5e1", font=("Segoe UI", 12),
                                                command=self.open_video_scopes)
                self.btn_scopes.pack(side="left", padx=1)
                _Tooltip(self.btn_scopes, "Video Scopes & Vectorscope")

                self.btn_crop = ctk.CTkButton(cf, text="📐", width=34, height=34, fg_color="transparent",
                                              hover_color="#222b3d", text_color="#cbd5e1", font=("Segoe UI", 12),
                                              command=self.open_crop_tool)
                self.btn_crop.pack(side="left", padx=1)
                _Tooltip(self.btn_crop, "Crop & Aspect Ratio Tool (C)")

                self.btn_mixer = ctk.CTkButton(cf, text="🎛️", width=34, height=34, fg_color="transparent",
                                               hover_color="#222b3d", text_color="#cbd5e1", font=("Segoe UI", 12),
                                               command=lambda: self.tabs.set("Inspector"))
                self.btn_mixer.pack(side="left", padx=1)
                _Tooltip(self.btn_mixer, "Open Inspector / Audio Rack")

                self.btn_customizer = ctk.CTkButton(cf, text="⚙️", width=34, height=34, fg_color="transparent",
                                                    hover_color="#222b3d", text_color="#cbd5e1", font=("Segoe UI", 12),
                                                    command=self._open_transport_customizer)
                self.btn_customizer.pack(side="left", padx=1)
                _Tooltip(self.btn_customizer, "Customize Transport Bar Layout")

                if alignment == "space_between":
                    pf.pack(side="left", padx=4, pady=2, expand=True)
                else:
                    pf.pack(side="left", padx=4, pady=2)

    def _shift_pod(self, pod_id: str, direction: int):
        order = self.transport_layout.get("pod_order", ["transport", "timecode", "vumeter", "tools"])
        if pod_id not in order:
            return
        idx = order.index(pod_id)
        new_idx = idx + direction
        if 0 <= new_idx < len(order):
            order[idx], order[new_idx] = order[new_idx], order[idx]
            self.transport_layout["pod_order"] = order
            config_mgr.save_config("transport_layout", self.transport_layout)
            self._rebuild_transport_bar()

    def _show_transport_context_menu(self, event, target_pod=None):
        menu = tk.Menu(self, tearoff=0, bg="#1a202c", fg="#f1f5f9", activebackground="#2563eb", activeforeground="#ffffff")
        if target_pod:
            pod_names = {
                "transport": "Transport Controls",
                "timecode": "Timecode & Sequence",
                "vumeter": "Stereo VU Meter",
                "tools": "Studio Tools"
            }
            pod_title = pod_names.get(target_pod, target_pod.title())
            menu.add_command(label=f"◀ Move {pod_title} Left", command=lambda: self._shift_pod(target_pod, -1))
            menu.add_command(label=f"▶ Move {pod_title} Right", command=lambda: self._shift_pod(target_pod, 1))
            menu.add_separator()

        align_menu = tk.Menu(menu, tearoff=0, bg="#1a202c", fg="#f1f5f9", activebackground="#2563eb", activeforeground="#ffffff")
        align_menu.add_command(label="Distributed (Space Between)", command=lambda: self._set_transport_alignment("space_between"))
        align_menu.add_command(label="Center Aligned", command=lambda: self._set_transport_alignment("center"))
        align_menu.add_command(label="Left Aligned", command=lambda: self._set_transport_alignment("left"))
        align_menu.add_command(label="Right Aligned", command=lambda: self._set_transport_alignment("right"))
        menu.add_cascade(label="Layout Alignment", menu=align_menu)

        vu_menu = tk.Menu(menu, tearoff=0, bg="#1a202c", fg="#f1f5f9", activebackground="#2563eb", activeforeground="#ffffff")
        vu_menu.add_command(label="Compact (240px)", command=lambda: self._set_vu_width(240))
        vu_menu.add_command(label="Standard (340px)", command=lambda: self._set_vu_width(340))
        vu_menu.add_command(label="Large Studio (460px)", command=lambda: self._set_vu_width(460))
        menu.add_cascade(label="VU Meter Size", menu=vu_menu)

        menu.add_separator()
        menu.add_command(label="⚙️ Customize Transport Layout...", command=self._open_transport_customizer)
        menu.add_command(label="↺ Reset to Default Layout", command=self._reset_transport_layout)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _set_transport_alignment(self, align: str):
        self.transport_layout["alignment"] = align
        config_mgr.save_config("transport_layout", self.transport_layout)
        self._rebuild_transport_bar()

    def _set_vu_width(self, width: int):
        self.transport_layout["vu_width"] = width
        config_mgr.save_config("transport_layout", self.transport_layout)
        if hasattr(self, "vu_meter") and self.vu_meter:
            self.vu_meter.set_width(width)
        self._rebuild_transport_bar()

    def _reset_transport_layout(self):
        self.transport_layout = copy.deepcopy(DEFAULT_TRANSPORT_LAYOUT)
        config_mgr.save_config("transport_layout", self.transport_layout)
        self._rebuild_transport_bar()

    def toggle_loop(self):
        self.is_looping = not getattr(self, "is_looping", False)
        if hasattr(self, "btn_loop") and self.btn_loop and self.btn_loop.winfo_exists():
            if self.is_looping:
                self.btn_loop.configure(fg_color="#1e3a8a", text_color="#60a5fa")
            else:
                self.btn_loop.configure(fg_color="transparent", text_color="#94a3b8")

    def toggle_master_mute(self):
        self.is_muted = not getattr(self, "is_muted", False)
        if hasattr(self, "btn_master_mute") and self.btn_master_mute and self.btn_master_mute.winfo_exists():
            if self.is_muted:
                self.btn_master_mute.configure(text="🔇", text_color="#ef4444")
                if hasattr(self, "vu_meter") and self.vu_meter:
                    self.vu_meter.reset()
            else:
                self.btn_master_mute.configure(text="🔊", text_color="#94a3b8")

    def _open_transport_customizer(self):
        _TransportCustomizerDialog(self)

    def _build_panels(self):
        self.tabs = ctk.CTkTabview(
            self,
            fg_color="#131620",
            segmented_button_fg_color="#1a1e2b",
            segmented_button_selected_color="#1f6feb",
            segmented_button_selected_hover_color="#2b7de9",
            segmented_button_unselected_color="#1a1e2b",
            segmented_button_unselected_hover_color="#283042",
            text_color="#f1f5f9",
            corner_radius=8,
            border_width=1,
            border_color="#283246"
        )
        self.tabs.grid(row=0, column=1, padx=(5, 10), pady=(10, 5), sticky="nsew")
        
        # Enhanced modern Studio Tab Header styling
        if hasattr(self.tabs, "_segmented_button") and self.tabs._segmented_button:
            self.tabs._segmented_button.configure(
                font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
                height=38,
                corner_radius=6
            )

        self.tab_media = self.tabs.add("Media")
        self.tab_edit  = self.tabs.add("Inspector")
        self.tab_export= self.tabs.add("Export")
        self._build_media_panel()
        self._build_inspector_panel()
        self._build_export_panel()

    def _get_media_thumbnail(self, filepath):
        if not hasattr(self, "_thumbnail_cache"):
            self._thumbnail_cache = {}
        if filepath in self._thumbnail_cache:
            return self._thumbnail_cache[filepath]

        thumb_w, thumb_h = 136, 76
        ext = os.path.splitext(filepath)[1].lower()

        # Image thumbnail
        if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'):
            try:
                with Image.open(filepath) as im:
                    im = im.convert('RGBA')
                    im.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
                    bg = Image.new('RGBA', (thumb_w, thumb_h), (18, 22, 33, 255))
                    offset = ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2)
                    bg.paste(im, offset, im if im.mode == 'RGBA' else None)
                    ctk_img = ctk.CTkImage(light_image=bg, dark_image=bg, size=(thumb_w, thumb_h))
                    self._thumbnail_cache[filepath] = ctk_img
                    return ctk_img
            except Exception:
                pass

        # Video thumbnail (extract 1 frame)
        elif ext in ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.flv'):
            try:
                cap = cv2.VideoCapture(filepath)
                cap.set(cv2.CAP_PROP_POS_MSEC, 500)
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(rgb).convert('RGBA')
                    im.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
                    bg = Image.new('RGBA', (thumb_w, thumb_h), (18, 22, 33, 255))
                    offset = ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2)
                    bg.paste(im, offset)
                    ctk_img = ctk.CTkImage(light_image=bg, dark_image=bg, size=(thumb_w, thumb_h))
                    self._thumbnail_cache[filepath] = ctk_img
                    return ctk_img
            except Exception:
                pass

        # Audio thumbnail (stylish vinyl disc / waveform art)
        elif ext in ('.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'):
            im = Image.new('RGBA', (thumb_w, thumb_h), (15, 23, 42, 255))
            from PIL import ImageDraw
            import math
            draw = ImageDraw.Draw(im)
            cx, cy = thumb_w // 2, thumb_h // 2
            draw.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], outline=(56, 189, 248, 220), width=2, fill=(30, 41, 59, 255))
            draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(14, 165, 233, 255), width=2, fill=(15, 23, 42, 255))
            for i, x in enumerate(range(10, thumb_w - 10, 5)):
                h = (math.sin(i * 0.7) * 0.5 + 0.5) * 20 + 4
                draw.line([x, cy - h/2, x, cy + h/2], fill=(0, 210, 255, 170), width=2)
            ctk_img = ctk.CTkImage(light_image=im, dark_image=im, size=(thumb_w, thumb_h))
            self._thumbnail_cache[filepath] = ctk_img
            return ctk_img

        # Generic thumbnail fallback
        im = Image.new('RGBA', (thumb_w, thumb_h), (20, 25, 38, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(im)
        draw.rectangle([10, 10, thumb_w - 10, thumb_h - 10], outline=(71, 85, 105, 200), width=1)
        ctk_img = ctk.CTkImage(light_image=im, dark_image=im, size=(thumb_w, thumb_h))
        self._thumbnail_cache[filepath] = ctk_img
        return ctk_img

    def _refresh_media_bin_ui(self):
        if not hasattr(self, "media_scroll") or not self.media_scroll.winfo_exists():
            return

        for child in self.media_scroll.winfo_children():
            child.destroy()

        if not self.media_bin:
            empty_frame = ctk.CTkFrame(self.media_scroll, fg_color="transparent")
            empty_frame.pack(fill="both", expand=True, pady=40)
            ctk.CTkLabel(empty_frame, text="📁", font=("Segoe UI", 28), text_color="#475569").pack(pady=(0, 4))
            ctk.CTkLabel(empty_frame, text="No Media Imported", font=("Segoe UI", 12, "bold"), text_color="#64748b").pack()
            ctk.CTkLabel(empty_frame, text="Click '+ Import' or drag files here", font=("Segoe UI", 10), text_color="#475569").pack(pady=(2, 0))
            return

        # 2-column grid configuration
        self.media_scroll.grid_columnconfigure(0, weight=1)
        self.media_scroll.grid_columnconfigure(1, weight=1)

        for idx, filepath in enumerate(self.media_bin):
            row = idx // 2
            col = idx % 2

            is_selected = (idx == self.selected_media_index)
            border_col = "#1f6feb" if is_selected else "#242c3f"
            card_bg = "#182238" if is_selected else "#121622"
            border_w = 2 if is_selected else 1

            card = ctk.CTkFrame(self.media_scroll, fg_color=card_bg,
                                border_width=border_w, border_color=border_col,
                                corner_radius=6)
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

            thumb_img = self._get_media_thumbnail(filepath)
            lbl_thumb = ctk.CTkLabel(card, image=thumb_img, text="", corner_radius=4)
            lbl_thumb.pack(padx=3, pady=(3, 2), fill="x")

            # Filename
            fname = os.path.basename(filepath)
            short_fname = fname if len(fname) <= 18 else fname[:15] + "..."
            lbl_name = ctk.CTkLabel(card, text=short_fname, font=("Segoe UI", 10, "bold" if is_selected else "normal"),
                                    text_color="#f1f5f9" if is_selected else "#cbd5e1")
            lbl_name.pack(padx=4, pady=(0, 4))
            _Tooltip(card, f"{fname}\nClick to select, Double-click to insert")

            def make_callbacks(i):
                def on_click(event):
                    self.selected_media_index = i
                    self._refresh_media_bin_ui()
                    self._on_media_select()

                def on_double_click(event):
                    self.selected_media_index = i
                    self.insert_to_timeline()

                return on_click, on_double_click

            click_fn, dclick_fn = make_callbacks(idx)
            for w in (card, lbl_thumb, lbl_name):
                w.bind("<Button-1>", click_fn)
                w.bind("<Double-Button-1>", dclick_fn)

    def _build_media_panel(self):
        self.media_listbox = _MediaListboxCompat(self)

        btn_row = ctk.CTkFrame(self.tab_media, fg_color="transparent")
        btn_row.pack(fill="x", padx=6, pady=(8, 6))
        ctk.CTkButton(btn_row, text="➕ Import", command=self.import_media,
                       font=("Segoe UI", 11, "bold"), height=34, width=105,
                       fg_color="#1f6feb", hover_color="#2b7de9").pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="✕ Remove", command=self._remove_media_item,
                       font=("Segoe UI", 11, "bold"), height=34,
                       fg_color="#363f50", hover_color="#48536a", text_color="#e2e8f0", width=100).pack(side="right", padx=2)

        # Media Visual Thumbnail Scroll Container
        self.media_scroll = ctk.CTkScrollableFrame(
            self.tab_media, fg_color="#0b0e14", corner_radius=8,
            border_width=1, border_color="#21283a"
        )
        self.media_scroll.pack(fill="both", expand=True, padx=6, pady=4)
        self._refresh_media_bin_ui()

        # Media Information Card (with polished card styling and vibrant accents)
        self.media_info_frame = ctk.CTkFrame(self.tab_media, fg_color="#171b26", corner_radius=8, border_width=1, border_color="#2c374d")
        self.media_info_frame.pack(fill="x", padx=6, pady=(6, 6))

        self.lbl_media_info_title = ctk.CTkLabel(
            self.media_info_frame, 
            text="📁 Media Properties", 
            font=("Segoe UI", 13, "bold"), 
            text_color="#00d2ff"
        )
        self.lbl_media_info_title.pack(anchor="w", padx=12, pady=(8, 3))

        self.lbl_media_info_desc = ctk.CTkLabel(
            self.media_info_frame, 
            text="Select a file to inspect resolution, codec, and audio properties.",
            font=("Segoe UI", 11, "bold"), 
            text_color="#e2e8f0",
            justify="left",
            anchor="w"
        )
        self.lbl_media_info_desc.pack(anchor="w", padx=12, pady=(0, 10), fill="x")

        # Drag-and-drop files onto the media bin
        if _HAS_DND:
            self.media_scroll.drop_target_register(DND_FILES)
            self.media_scroll.dnd_bind('<<Drop>>', self._on_dnd_drop_media)

        ins_row = ctk.CTkFrame(self.tab_media, fg_color="transparent")
        ins_row.pack(fill="x", padx=6, pady=(4, 8))
        ctk.CTkLabel(ins_row, text="Track:", font=("Segoe UI", 11, "bold"), text_color="#94a3b8").pack(side="left", padx=(2, 4))
        self.ins_trk_var = ctk.StringVar(value="V1")
        self.ins_trk_combo = ctk.CTkComboBox(ins_row, variable=self.ins_trk_var,
                                             values=["V2", "V1", "A1", "A2"], width=75, height=34,
                                             font=("Segoe UI", 11, "bold"), fg_color="#1c212d", border_color="#333d52")
        self.ins_trk_combo.pack(side="left", padx=4)
        ctk.CTkButton(ins_row, text="⬇ Insert to Timeline", height=34,
                       font=("Segoe UI", 11, "bold"), fg_color="#0e8a7d", hover_color="#12a899",
                       command=self.insert_to_timeline).pack(side="right", padx=2)

    def _on_media_select(self, event=None):
        """Update Media Properties card when a media item is clicked in the media bin."""
        if self.selected_media_index is None or not (0 <= self.selected_media_index < len(self.media_bin)):
            self.lbl_media_info_desc.configure(
                text="Select a file to inspect resolution, codec, and audio properties."
            )
            return
        path = self.media_bin[self.selected_media_index]
        info = get_media_info_summary(path)
        if not info:
            return

        lines = [
            f"• File:  {info['filename']}  ({info['type']})",
            f"• Size & Duration:  {info['size']}  |  {info['duration']}",
        ]
        if info['resolution'] != '—':
            lines.append(f"• Format:  {info['resolution']} @ {info['fps']}")
        if info['audio'] != '—':
            lines.append(f"• Audio:  {info['audio']}")

        self.lbl_media_info_desc.configure(text="\n".join(lines))

    def refresh_track_selectors(self):
        """Update track combo boxes to reflect current timeline tracks."""
        if hasattr(self, 'ins_trk_combo') and hasattr(self, 'timeline'):
            self.ins_trk_combo.configure(values=list(self.timeline.track_keys))
            if self.ins_trk_var.get() not in self.timeline.track_keys and self.timeline.track_keys:
                self.ins_trk_var.set(self.timeline.track_keys[0])

    def _build_inspector_panel(self):
        self.insp_lbl = ctk.CTkLabel(self.tab_edit, text="Select a clip on the timeline.",
                                      text_color="#888888", font=("Segoe UI", 12, "bold"))
        self.insp_lbl.pack(pady=30)

        self.insp_c = ctk.CTkScrollableFrame(self.tab_edit, fg_color="transparent")

        def mk_row(lbl, widget_cls, **kwargs):
            f = ctk.CTkFrame(self.insp_c, fg_color="transparent")
            f.pack(fill="x", pady=3)
            if lbl:
                ctk.CTkLabel(f, text=lbl, width=78, anchor="w",
                              font=("Segoe UI", 10, "bold")).pack(side="left")
            w = widget_cls(f, **kwargs)
            w.pack(side="right", fill="x", expand=True)
            return w

        def section(txt):
            ctk.CTkLabel(self.insp_c, text=txt,
                          font=("Segoe UI", 12, "bold"),
                          text_color="#00d2be").pack(anchor="w", pady=(12, 2))
            ctk.CTkFrame(self.insp_c, height=1, fg_color="#3a4254").pack(fill="x", pady=(0, 4))

        # Variables
        self.var_tin, self.var_tout = ctk.StringVar(), ctk.StringVar()
        self.var_scale = ctk.DoubleVar(value=1.0)
        self.var_vol   = ctk.DoubleVar(value=1.0)
        self.var_x, self.var_y      = ctk.StringVar(), ctk.StringVar()
        self.var_aspect, self.var_rot = ctk.StringVar(), ctk.StringVar()
        self.var_speed = ctk.DoubleVar(value=1.0)
        self.var_wm, self.var_wmpos = ctk.StringVar(), ctk.StringVar()
        self.var_rmbg  = ctk.BooleanVar()
        self.var_mute  = ctk.BooleanVar()
        self.var_transition = ctk.StringVar(value="None")
        self.var_trans_dur  = ctk.StringVar(value="0.5")
        self.var_notes      = ctk.StringVar()

        # ── Trim ──
        section("Trim")
        tr = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        tr.pack(fill="x", pady=2)
        ctk.CTkEntry(tr, textvariable=self.var_tin, width=65, height=30, font=("Segoe UI", 10)).pack(side="left", padx=2)
        ctk.CTkButton(tr, text="[  In", width=52, height=30, font=("Segoe UI", 10, "bold"),
                       command=lambda: self.set_trim('in')).pack(side="left", padx=2)
        ctk.CTkButton(tr, text="Out  ]", width=52, height=30, font=("Segoe UI", 10, "bold"),
                       command=lambda: self.set_trim('out')).pack(side="right", padx=2)
        ctk.CTkEntry(tr, textvariable=self.var_tout, width=65, height=30, font=("Segoe UI", 10)).pack(side="right", padx=2)

        # ── Transition ──
        section("Transition (into next clip)")
        mk_row("Type:", ctk.CTkComboBox, variable=self.var_transition, height=30, font=("Segoe UI", 10),
               values=TRANSITION_TYPES)
        mk_row("Duration s:", ctk.CTkEntry, textvariable=self.var_trans_dur, height=30, font=("Segoe UI", 10))

        # ── Transform ──
        section("Transform")
        self.btn_open_crop = ctk.CTkButton(
            self.insp_c,
            text="📐 Visual Crop & Aspect Tool…",
            height=32,
            font=("Segoe UI", 10, "bold"),
            fg_color="#1d4ed8",
            hover_color="#1e40af",
            command=self.open_crop_tool,
        )
        self.btn_open_crop.pack(fill="x", pady=(2, 4))

        crop_stat_row = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        crop_stat_row.pack(fill="x", pady=2)
        ctk.CTkLabel(crop_stat_row, text="Crop Status:", width=78, anchor="w",
                      font=("Segoe UI", 10, "bold")).pack(side="left")
        self.lbl_crop_status = ctk.CTkLabel(crop_stat_row, text="Full Frame", font=("Segoe UI", 10), text_color="#94a3b8")
        self.lbl_crop_status.pack(side="left", padx=4)
        self.btn_reset_crop = ctk.CTkButton(
            crop_stat_row,
            text="Reset",
            width=52,
            height=22,
            font=("Segoe UI", 9),
            fg_color="#334155",
            hover_color="#475569",
            command=self._reset_clip_crop,
        )
        self.btn_reset_crop.pack(side="right")

        mk_row("Aspect:", ctk.CTkComboBox, variable=self.var_aspect, height=30, font=("Segoe UI", 10),
               values=["Original", "16:9", "9:16", "1:1", "4:3", "21:9", "4:5", "2:3"])
        mk_row("Rotate:", ctk.CTkComboBox, variable=self.var_rot, height=30, font=("Segoe UI", 10),
               values=["0°", "90°", "180°", "270°"])

        sc_row = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        sc_row.pack(fill="x", pady=3)
        ctk.CTkLabel(sc_row, text="Scale:", width=78, anchor="w",
                      font=("Segoe UI", 10, "bold")).pack(side="left")
        self.lbl_sc = ctk.CTkLabel(sc_row, text="1.0×", width=42,
                                    font=("Consolas", 11, "bold"))
        self.lbl_sc.pack(side="right")
        ctk.CTkSlider(sc_row, variable=self.var_scale, from_=0.1, to=3.0, height=18,
                       command=lambda v: self.lbl_sc.configure(text=f"{v:.2f}×")
                       ).pack(side="right", fill="x", expand=True, padx=4)

        pos_row = mk_row("Pos X / Y:", ctk.CTkEntry, textvariable=self.var_x, width=65, height=30, font=("Segoe UI", 10))
        ctk.CTkEntry(pos_row.master, textvariable=self.var_y, width=65, height=30, font=("Segoe UI", 10)).pack(side="right", padx=2)

        # ── Audio ──
        section("Audio")
        vol_row = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        vol_row.pack(fill="x", pady=3)
        ctk.CTkLabel(vol_row, text="Volume:", width=78, anchor="w",
                      font=("Segoe UI", 10, "bold")).pack(side="left")
        self.lbl_vol = ctk.CTkLabel(vol_row, text="100%", width=42,
                                     font=("Consolas", 11, "bold"))
        self.lbl_vol.pack(side="right")
        ctk.CTkSlider(vol_row, variable=self.var_vol, from_=0.0, to=2.0, height=18,
                       command=lambda v: self.lbl_vol.configure(text=f"{int(v*100)}%")
                       ).pack(side="right", fill="x", expand=True, padx=4)
        mk_row("", ctk.CTkCheckBox, text="Mute Track / Clip", variable=self.var_mute, font=("Segoe UI", 10, "bold"))

        # ── Effects ──
        section("Effects")
        sp_row = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        sp_row.pack(fill="x", pady=3)
        ctk.CTkLabel(sp_row, text="Speed:", width=78, anchor="w",
                      font=("Segoe UI", 10, "bold")).pack(side="left")
        self.lbl_sp = ctk.CTkLabel(sp_row, text="1.00×", width=42,
                                    font=("Consolas", 11, "bold"))
        self.lbl_sp.pack(side="right")

        def _on_speed_slider_drag(v):
            spd = round(float(v), 2)
            self.lbl_sp.configure(text=f"{spd:.2f}×")
            cid = getattr(self.timeline, 'active_clip_id', None)
            if cid:
                c = next((x for x in self.project_clips if x['id'] == cid), None)
                if c:
                    c['speed'] = spd
                    self.timeline._needs_full_redraw = True
                    self.timeline.draw()
            if hasattr(self, '_update_export_size_calculator'):
                self._update_export_size_calculator('bitrate')

        self.slider_speed = ctk.CTkSlider(sp_row, variable=self.var_speed, from_=0.25, to=4.0, height=18,
                                           command=_on_speed_slider_drag)
        self.slider_speed.pack(side="right", fill="x", expand=True, padx=4)
        self.slider_speed.bind("<ButtonRelease-1>", lambda e: self.rebuild_proxy_cache())

        mk_row("Watermark:", ctk.CTkEntry, textvariable=self.var_wm, height=30, font=("Segoe UI", 10))
        mk_row("WM Pos:", ctk.CTkComboBox, variable=self.var_wmpos, height=30, font=("Segoe UI", 10),
               values=["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"])

        # ── AI / Background ──
        section("AI / Background")
        mk_row("", ctk.CTkCheckBox, text="Remove Background (AI)", variable=self.var_rmbg, font=("Segoe UI", 10, "bold"))
        ai_f = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        ai_f.pack(fill="x", pady=4)
        ctk.CTkButton(ai_f, text="👁 Preview Mask", width=120, height=32, font=("Segoe UI", 10, "bold"),
                       command=self.preview_mask).pack(side="left", padx=2)
        ctk.CTkButton(ai_f, text="🖌 Touchup", width=100, height=32, font=("Segoe UI", 10, "bold"),
                       command=self.open_touchup).pack(side="right", padx=2)

        # ── Colour Grading ──
        section("Colour Grading")
        if _HAS_COLOR_GRADING:
            self.color_grading_panel = ColorGradingPanel(self.insp_c)
            self.color_grading_panel.pack(fill="x", pady=2)
        else:
            self.color_grading_panel = None
            ctk.CTkLabel(self.insp_c, text="(color_grading module not found)",
                          text_color="#555555", font=("Segoe UI", 9)).pack(anchor="w")

        # ── Effects Registry ─────────────────────────────────────────────
        section("Video Effects")
        if _HAS_EFFECTS:
            self._effects_widgets: dict = {}   # effect_name → {enabled_var, param_vars}
            self._effects_frame = ctk.CTkScrollableFrame(
                self.insp_c, height=160, fg_color="#1a1a1a", corner_radius=6)
            self._effects_frame.pack(fill="x", pady=4)
            self._build_effects_registry_panel()
        else:
            ctk.CTkLabel(self.insp_c, text="(effects_registry module not found)",
                          text_color="#555555", font=("Segoe UI", 9)).pack(anchor="w")

        ctk.CTkButton(self.insp_c, text="✔ Apply Changes", height=40,
                       font=("Segoe UI", 12, "bold"),
                       command=self.apply_inspector_changes).pack(pady=12, fill="x")

        # Keyframes button
        ctk.CTkButton(self.insp_c, text="⏱ Edit Keyframes", height=32,
                       font=("Segoe UI", 10, "bold"),
                       fg_color="#333333",
                       command=self._open_keyframe_editor).pack(pady=4, fill="x")

        # ── Clip Tools ──
        section("Clip Tools")
        ctk.CTkButton(self.insp_c, text="Export Transparent GIF", fg_color="teal",
                       hover_color="#006666", height=34, font=("Segoe UI", 10, "bold"),
                       command=lambda: self.export_clip(True)).pack(pady=2, fill="x")
        ctk.CTkButton(self.insp_c, text="Extract PNG Frames", fg_color="teal",
                       hover_color="#006666", height=34, font=("Segoe UI", 10, "bold"),
                       command=lambda: self.export_clip(False)).pack(pady=2, fill="x")
        ctk.CTkButton(self.insp_c, text="🗑 Delete Clip", height=34, font=("Segoe UI", 10, "bold"),
                       fg_color="#6b1111", hover_color="#4a0a0a",
                       command=self._kb_delete).pack(pady=(10, 4), fill="x")

        # ── Notes ──
        section("Notes")
        ctk.CTkEntry(self.insp_c, textvariable=self.var_notes, height=30, font=("Segoe UI", 10),
                     placeholder_text="Clip notes…").pack(fill="x", pady=2)

    def _get_sequence_total_duration(self) -> float:
        """Return total sequence duration in seconds accounting for track & clip speeds."""
        if not self.project_clips:
            return 1.0
        if hasattr(self, 'timeline') and hasattr(self.timeline, 'get_effective_clip_duration'):
            return max([c['start'] + self.timeline.get_effective_clip_duration(c) for c in self.project_clips] or [1.0])
        return max([c['start'] + (c['trim_out'] - c['trim_in']) for c in self.project_clips] or [1.0])

    def _build_export_panel(self):
        # ── Format & Encoding Options ──
        def row(lbl, widget_cls, **kw):
            f = ctk.CTkFrame(self.tab_export, fg_color="transparent")
            f.pack(fill="x", pady=4, padx=8)
            ctk.CTkLabel(f, text=lbl, width=80, anchor="w", font=("Segoe UI", 11, "bold")).pack(side="left")
            w = widget_cls(f, height=32, font=("Segoe UI", 10), **kw)
            w.pack(side="right", fill="x", expand=True)
            return w

        self.exp_fmt   = row("Format:", ctk.CTkComboBox,
                              values=["mp4", "avi", "mkv", "webm", "mov"])
        self.exp_codec = row("Codec:", ctk.CTkComboBox,
                              values=["libx264", "h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_amf", "libvpx-vp9", "libx265"])
        self.exp_fps   = row("FPS:", ctk.CTkComboBox,
                              values=["23.976", "24", "25", "29.97", "30", "50", "60"])
        self.exp_res   = row("Resolution:", ctk.CTkComboBox,
                              values=["1920x1080", "3840x2160", "2560x1440",
                                      "1280x720", "854x480", "640x360"])

        # ── Quality & Target File Size Optimizer Card ─────────────────────
        card = ctk.CTkFrame(self.tab_export, fg_color="#181a20", corner_radius=10,
                            border_width=1, border_color="#2d3548")
        card.pack(fill="x", padx=8, pady=(10, 6))

        # Title bar of optimizer
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(hdr, text="🎛 Quality & Target File Size Optimizer",
                      font=("Segoe UI", 12, "bold"), text_color="#00d2be").pack(side="left")

        self.lbl_exp_dur = ctk.CTkLabel(hdr, text="Duration: 00:00 (0.0s)",
                                         font=("Consolas", 10, "bold"), text_color="#a0aec0")
        self.lbl_exp_dur.pack(side="right")

        # Variables for calculator
        self.var_quality_preset = ctk.StringVar(value="Medium (6 Mbps)")
        self.var_exp_bitrate    = ctk.DoubleVar(value=6000.0)  # in kbps
        self.var_target_size    = ctk.StringVar(value="50.0")
        self.var_target_unit    = ctk.StringVar(value="MB")

        # 1. Preset dropdown row
        pr_row = ctk.CTkFrame(card, fg_color="transparent")
        pr_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(pr_row, text="Quality Preset:", width=105, anchor="w",
                      font=("Segoe UI", 10, "bold")).pack(side="left")
        self.combo_quality_preset = ctk.CTkComboBox(
            pr_row, variable=self.var_quality_preset, height=30, font=("Segoe UI", 10),
            values=[
                "Auto (Target File Size)",
                "Ultra (20 Mbps)",
                "High (12 Mbps)",
                "Medium (6 Mbps)",
                "Low (2.5 Mbps)",
                "Web (1 Mbps)",
                "Custom Bitrate",
            ],
            command=lambda v: self._update_export_size_calculator('preset')
        )
        self.combo_quality_preset.pack(side="right", fill="x", expand=True)

        # 2. Target File Size (MB / GB) entry and unit row
        ts_row = ctk.CTkFrame(card, fg_color="transparent")
        ts_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(ts_row, text="Target File Size:", width=105, anchor="w",
                      font=("Segoe UI", 10, "bold")).pack(side="left")

        self.unit_combo = ctk.CTkComboBox(
            ts_row, variable=self.var_target_unit, values=["MB", "GB"], width=70, height=30,
            font=("Segoe UI", 10, "bold"),
            command=lambda u: self._update_export_size_calculator('unit_change')
        )
        self.unit_combo.pack(side="right", padx=(4, 0))

        self.ent_target_size = ctk.CTkEntry(
            ts_row, textvariable=self.var_target_size, height=30, font=("Consolas", 11, "bold"), width=90
        )
        self.ent_target_size.pack(side="right", fill="x", expand=True)
        self.ent_target_size.bind("<KeyRelease>", lambda e: self._update_export_size_calculator('target'))

        # 3. Video Bitrate Slider row
        br_row = ctk.CTkFrame(card, fg_color="transparent")
        br_row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(br_row, text="Video Bitrate:", width=105, anchor="w",
                      font=("Segoe UI", 10, "bold")).pack(side="left")
        self.lbl_bitrate_val = ctk.CTkLabel(br_row, text="6000 kbps (6.0 Mbps)", width=135,
                                             anchor="e", font=("Consolas", 10, "bold"), text_color="#38bdf8")
        self.lbl_bitrate_val.pack(side="right")
        self.slider_bitrate = ctk.CTkSlider(
            br_row, variable=self.var_exp_bitrate, from_=300.0, to=30000.0, height=18,
            command=lambda v: self._update_export_size_calculator('bitrate')
        )
        self.slider_bitrate.pack(side="right", fill="x", expand=True, padx=(4, 8))

        # 4. Live Estimated Size Banner
        est_frame = ctk.CTkFrame(card, fg_color="#101318", corner_radius=6, border_width=1, border_color="#232a3b")
        est_frame.pack(fill="x", padx=12, pady=(6, 12))
        self.lbl_exp_est_size = ctk.CTkLabel(
            est_frame,
            text="Estimated File Size: ~50.0 MB  |  Video Bitrate: 6,000 kbps",
            font=("Segoe UI", 11, "bold"),
            text_color="#facc15"
        )
        self.lbl_exp_est_size.pack(pady=8, padx=10)

        # Initial calculator calculation
        self._update_export_size_calculator('preset')

        ctk.CTkButton(self.tab_export, text="⬛  Render Project", height=50,
                       fg_color="#1a6e1a", hover_color="#145414",
                       font=("Segoe UI", 13, "bold"),
                       command=self.render_project).pack(pady=16, padx=8, fill="x")

    def _update_export_size_calculator(self, source='preset'):
        """Bidirectionally synchronize Target File Size (MB/GB), Bitrate, Quality presets, and sequence duration."""
        if not hasattr(self, 'lbl_exp_est_size'):
            return

        T = max(1.0, self._get_sequence_total_duration())
        mins, secs = divmod(T, 60)
        hours, mins = divmod(mins, 60)
        if hours >= 1:
            dur_str = f"{int(hours)}:{int(mins):02d}:{secs:02.0f}"
        else:
            dur_str = f"{int(mins):02d}:{secs:04.1f}"
        self.lbl_exp_dur.configure(text=f"Duration: {dur_str} ({T:.1f}s)")

        audio_bitrate_kbps = 192.0  # standard high quality audio
        preset_map = {
            "Ultra (20 Mbps)": 20000.0,
            "High (12 Mbps)": 12000.0,
            "Medium (6 Mbps)": 6000.0,
            "Low (2.5 Mbps)": 2500.0,
            "Web (1 Mbps)": 1000.0,
        }

        unit = self.var_target_unit.get().strip().upper() or "MB"

        if source == 'preset':
            preset = self.var_quality_preset.get()
            if preset in preset_map:
                v_bitrate = preset_map[preset]
                self.var_exp_bitrate.set(v_bitrate)
                # Compute resulting size
                total_kbps = v_bitrate + audio_bitrate_kbps
                size_mb = (total_kbps * T) / 8192.0
                if unit == "GB":
                    self.var_target_size.set(f"{size_mb / 1024.0:.3f}")
                else:
                    self.var_target_size.set(f"{size_mb:.1f}")

        elif source == 'unit_change':
            # Preserve the current byte size but show in new unit
            v_bitrate = self.var_exp_bitrate.get()
            total_kbps = v_bitrate + audio_bitrate_kbps
            size_mb = (total_kbps * T) / 8192.0
            if unit == "GB":
                self.var_target_size.set(f"{size_mb / 1024.0:.3f}")
            else:
                self.var_target_size.set(f"{size_mb:.1f}")

        elif source == 'target':
            try:
                val = float(self.var_target_size.get().strip() or "0")
                if val > 0:
                    size_mb = val * 1024.0 if unit == "GB" else val
                    total_kbps_needed = (size_mb * 8192.0) / T
                    v_bitrate = max(200.0, min(80000.0, total_kbps_needed - audio_bitrate_kbps))
                    self.var_exp_bitrate.set(round(v_bitrate))
                    self.var_quality_preset.set("Auto (Target File Size)")
            except ValueError:
                pass

        elif source == 'bitrate':
            v_bitrate = self.var_exp_bitrate.get()
            total_kbps = v_bitrate + audio_bitrate_kbps
            size_mb = (total_kbps * T) / 8192.0
            if unit == "GB":
                self.var_target_size.set(f"{size_mb / 1024.0:.3f}")
            else:
                self.var_target_size.set(f"{size_mb:.1f}")
            # Find closest preset or mark Custom
            matched_preset = "Custom Bitrate"
            for p_name, p_rate in preset_map.items():
                if abs(v_bitrate - p_rate) < 50.0:
                    matched_preset = p_name
                    break
            self.var_quality_preset.set(matched_preset)

        # Update display labels
        v_bitrate = self.var_exp_bitrate.get()
        if v_bitrate >= 1000:
            self.lbl_bitrate_val.configure(text=f"{int(v_bitrate):,} kbps ({v_bitrate/1000:.1f} Mbps)")
        else:
            self.lbl_bitrate_val.configure(text=f"{int(v_bitrate)} kbps")

        total_kbps = v_bitrate + audio_bitrate_kbps
        est_mb = (total_kbps * T) / 8192.0
        if est_mb >= 1024:
            self.lbl_exp_est_size.configure(
                text=f"Estimated File Size: ~{est_mb/1024.0:.2f} GB ({est_mb:.1f} MB)  |  Video: {int(v_bitrate):,} kbps  |  Audio: 192 kbps"
            )
        else:
            self.lbl_exp_est_size.configure(
                text=f"Estimated File Size: ~{est_mb:.1f} MB  |  Video: {int(v_bitrate):,} kbps  |  Audio: 192 kbps"
            )

    def _build_timeline_area(self):
        tf = ctk.CTkFrame(self)
        tf.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        tf.grid_rowconfigure(0, weight=1)
        tf.grid_columnconfigure(0, weight=1)
        self._timeline_container = tf
        self.timeline = MultiTrackTimeline(tf, self)
        self.timeline.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Wire drag-and-drop onto the timeline canvas
        if _HAS_DND:
            self.timeline.canvas.drop_target_register(DND_FILES)
            self.timeline.canvas.dnd_bind('<<Drop>>', self._on_dnd_drop_timeline)

    def _build_statusbar(self):
        sb = ctk.CTkFrame(self, height=36, corner_radius=0)
        sb.grid(row=2, column=0, columnspan=2, padx=0, pady=0, sticky="ew")
        sb.grid_columnconfigure(1, weight=1)

        self.lbl_status = ctk.CTkLabel(sb, text="Ready.", anchor="w",
                                        font=("Segoe UI", 10, "bold"))
        self.lbl_status.grid(row=0, column=1, padx=12, sticky="w")

        self.prog_bar = ctk.CTkProgressBar(sb, mode="determinate", width=250, height=10)
        self.prog_bar.grid(row=0, column=2, padx=8, pady=8)
        self.prog_bar.set(0)

        self.btn_cancel = ctk.CTkButton(sb, text="✕ Cancel", width=90, height=26,
                                         fg_color="#7a2020", hover_color="#5c1515",
                                         font=("Segoe UI", 10, "bold"),
                                         command=self.cancel_task, state="disabled")
        self.btn_cancel.grid(row=0, column=3, padx=(0, 8))

        ctk.CTkLabel(sb, text="Space=Play  I/O=Trim  Del=Delete  S=Split  D=Dup  Z=Undo  Ctrl+Scroll=Zoom",
                      text_color="#666666", font=("Segoe UI", 9, "bold")
                      ).grid(row=0, column=4, padx=12)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------
    def _is_text_focused(self) -> bool:
        """Check if an Entry, Textbox, or input widget currently has keyboard focus."""
        f = self.focus_get()
        if f is None:
            return False
        cls_name = f.__class__.__name__.lower()
        return any(k in cls_name for k in ("entry", "text", "spinbox"))

    def _bind_keys(self):
        # Spacebar Play/Pause toggle
        self.bind("<space>",          lambda e: not self._is_text_focused() and self.toggle_play())
        # Delete / Ripple Delete
        self.bind("<Delete>",         lambda e: not self._is_text_focused() and self._kb_delete())
        self.bind("<BackSpace>",      lambda e: not self._is_text_focused() and self._kb_delete())
        self.bind("<Shift-Delete>",   lambda e: not self._is_text_focused() and self._kb_ripple_delete())
        self.bind("<Shift-BackSpace>",lambda e: not self._is_text_focused() and self._kb_ripple_delete())
        # Playhead Navigation: Single Frame step
        self.bind("<Left>",           lambda e: not self._is_text_focused() and self._step_back())
        self.bind("<Right>",          lambda e: not self._is_text_focused() and self._step_forward())
        # Playhead Navigation: 1-second Jump
        self.bind("<Shift-Left>",     lambda e: not self._is_text_focused() and self.scrub_to(max(0.0, self.playhead_t - 1.0)))
        self.bind("<Shift-Right>",    lambda e: not self._is_text_focused() and self.scrub_to(self.playhead_t + 1.0))
        # Home / End
        self.bind("<Home>",           lambda e: not self._is_text_focused() and self._kb_goto_start())
        self.bind("<End>",            lambda e: not self._is_text_focused() and self._kb_goto_end())
        # In / Out trim marks
        self.bind("i",                lambda e: not self._is_text_focused() and self.set_trim('in'))
        self.bind("o",                lambda e: not self._is_text_focused() and self.set_trim('out'))
        # Split & Duplicate
        self.bind("s",                lambda e: not self._is_text_focused() and self.split_clip_at_playhead(self.timeline.active_clip_id))
        self.bind("<Control-k>",      lambda e: not self._is_text_focused() and self.split_clip_at_playhead(self.timeline.active_clip_id))
        self.bind("d",                lambda e: not self._is_text_focused() and self.duplicate_clip(self.timeline.active_clip_id))
        self.bind("<Control-d>",      lambda e: not self._is_text_focused() and self.duplicate_clip(self.timeline.active_clip_id))
        # Crop & Aspect Ratio Tool
        self.bind("c",                lambda e: not self._is_text_focused() and self.open_crop_tool())
        self.bind("C",                lambda e: not self._is_text_focused() and self.open_crop_tool())
        # J / K / L Shuttle
        self.bind("j",                lambda e: not self._is_text_focused() and self._shuttle(-2))
        self.bind("k",                lambda e: not self._is_text_focused() and self._kb_pause())
        self.bind("l",                lambda e: not self._is_text_focused() and self._shuttle(2))
        # Markers & Tools
        self.bind("m",                lambda e: not self._is_text_focused() and self._add_marker_at_playhead())
        # File & Edit controls
        self.bind("<Control-i>",      lambda e: self.import_media())
        self.bind("<Control-q>",      lambda e: self._on_close())
        self.bind("<Control-n>",      lambda e: self.new_project())
        self.bind("<Control-o>",      lambda e: self.open_project())
        self.bind("<Control-s>",      lambda e: self.save_project())
        self.bind("<Control-S>",      lambda e: self.save_project_as())  # Ctrl+Shift+S
        self.bind("<Control-z>",      lambda e: self._undo())
        self.bind("<Control-Z>",      lambda e: self._redo())  # Ctrl+Shift+Z
        self.bind("<Control-y>",      lambda e: self._redo())
        self.bind("`",                lambda e: self._toggle_fullscreen_preview())
        self.bind("<F11>",            lambda e: self._toggle_fullscreen_preview())
        # Zoom
        self.bind("<equal>",          lambda e: not self._is_text_focused() and self.timeline._zoom_in())
        self.bind("<minus>",          lambda e: not self._is_text_focused() and self.timeline._zoom_out())
        # Preferences
        self.bind("<Control-comma>",  lambda e: self._open_preferences())

    def _kb_delete(self):
        cid = self.timeline.active_clip_id
        if cid:
            self.delete_clip(cid)

    def _kb_ripple_delete(self):
        cid = self.timeline.active_clip_id
        if cid:
            self.ripple_delete_clip(cid)

    def ripple_delete_clip(self, clip_id: str):
        """Delete a clip and close the gap by shifting downstream clips."""
        if not clip_id:
            return
        c = next((x for x in self.project_clips if x['id'] == clip_id), None)
        if not c:
            return
        if _HAS_COMMANDS and self.cmd_stack is not None:
            self.cmd_stack.push(RippleDeleteClipCommand(self, c))
        else:
            dur = c['trim_out'] - c['trim_in']
            st = c['start']
            trk = c.get('track')
            self.project_clips = [x for x in self.project_clips if x['id'] != clip_id]
            for x in self.project_clips:
                if x.get('track') == trk and x['start'] >= st + dur - 0.001:
                    x['start'] = max(0.0, x['start'] - dur)
            self.timeline.active_clip_id = None
            self.update_inspector()
            self.rebuild_proxy_cache()
            self.timeline.draw()

    def _kb_goto_start(self):
        self.scrub_to(0.0)

    def _kb_goto_end(self):
        if self.master_proxy_clip:
            self.scrub_to(self.master_proxy_clip.duration)

    def _step_back(self):
        self.scrub_to(max(0.0, self.playhead_t - 1.0 / self.fps))

    def _step_forward(self):
        self.scrub_to(self.playhead_t + 1.0 / self.fps)

    def _shuttle(self, direction):
        """J/L shuttle: nudge playhead by 5 frames per press."""
        self.scrub_to(max(0.0, self.playhead_t + direction * 5.0 / self.fps))

    def _kb_pause(self):
        if self.is_playing:
            self.toggle_play()

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------
    def _undo(self):
        if self.cmd_stack and self.cmd_stack.can_undo:
            self.cmd_stack.undo()

    def _redo(self):
        if self.cmd_stack and self.cmd_stack.can_redo:
            self.cmd_stack.redo()

    # ------------------------------------------------------------------
    # Sequence settings
    # ------------------------------------------------------------------
    def _open_seq_settings(self):
        _SequenceSettingsDialog(self)

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------
    def _post_startup(self):
        """Tasks deferred until after the UI is fully drawn."""
        # Crash reporter
        if _HAS_CRASH_REPORTER:
            install_crash_handler(self)
        # Version check (non-blocking background thread)
        if _HAS_VERSION_CHECK:
            check_for_updates(self, getattr(self, 'lbl_status', None))
        # Autosave recovery check
        if _HAS_AUTOSAVE:
            check_autosave_recovery(self)

    def _on_close(self):
        config_mgr.save_config("geometry", self.geometry())
        if self.is_playing:
            self.is_playing = False
        if self._frame_queue is not None:
            self._frame_queue.stop()
        if self._autosaver:
            self._autosaver.stop()
        if _HAS_AUTOSAVE:
            clear_autosave()
        for c in self.open_proxy_clips:
            try:
                c.close()
            except Exception:
                pass
        self.destroy()

    # ------------------------------------------------------------------
    # Full-screen preview
    # ------------------------------------------------------------------
    def _toggle_fullscreen_preview(self):
        """Open/close a Toplevel that fills the screen with the current preview frame."""
        if hasattr(self, '_fs_window') and self._fs_window and self._fs_window.winfo_exists():
            self._fs_window.destroy()
            self._fs_window = None
            return
        self._fs_window = tk.Toplevel(self)
        self._fs_window.attributes('-fullscreen', True)
        self._fs_window.configure(bg='black')
        lbl = tk.Label(self._fs_window, bg='black')
        lbl.pack(fill='both', expand=True)
        self._fs_window.bind('<Escape>', lambda e: self._toggle_fullscreen_preview())
        self._fs_window.bind('`',        lambda e: self._toggle_fullscreen_preview())
        if self.master_proxy_clip:
            try:
                frame = self.master_proxy_clip.get_frame(self.playhead_t)
                pim = Image.fromarray(frame)
                sw = self._fs_window.winfo_screenwidth()
                sh = self._fs_window.winfo_screenheight()
                pim.thumbnail((sw, sh), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(pim)
                lbl.configure(image=tk_img)
                lbl._image = tk_img
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Media bin
    # ------------------------------------------------------------------
    def import_media(self):
        paths = filedialog.askopenfilenames(
            title="Import Media",
            filetypes=[("Media Files", "*.mp4 *.avi *.mkv *.mov *.webm *.mp3 *.wav *.aac *.flac *.png *.jpg *.jpeg *.bmp *.webp"),
                       ("Video", "*.mp4 *.avi *.mkv *.mov *.webm"),
                       ("Audio", "*.mp3 *.wav *.aac *.flac"),
                       ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"),
                       ("All Files", "*.*")]
        )
        last_idx = None
        for p in paths:
            if p not in self.media_bin:
                self.media_bin.append(p)
                last_idx = len(self.media_bin) - 1
        if last_idx is not None:
            self.selected_media_index = last_idx
            self._refresh_media_bin_ui()
            self._on_media_select()
        log.info("Imported %d file(s)", len(paths))

    def _remove_media_item(self):
        if self.selected_media_index is None or not (0 <= self.selected_media_index < len(self.media_bin)):
            return
        idx = self.selected_media_index
        self.media_bin.pop(idx)
        if self.media_bin:
            self.selected_media_index = min(idx, len(self.media_bin) - 1)
        else:
            self.selected_media_index = None
        self._refresh_media_bin_ui()
        self._on_media_select()

    def _media_double_click(self, event):
        self.insert_to_timeline()

    def insert_to_timeline(self):
        if self.selected_media_index is None or not (0 <= self.selected_media_index < len(self.media_bin)):
            return
        path = self.media_bin[self.selected_media_index]
        trk  = self.ins_trk_var.get()
        try:
            if is_image_file(path) or trk.startswith('I'):
                dur = 5.0
            elif trk.startswith('A'):
                clip = AudioFileClip(path)
                dur = clip.duration
                clip.close()
            else:
                clip = VideoFileClip(path)
                dur = clip.duration
                clip.close()
            trk_clips = [c for c in self.project_clips if c['track'] == trk]
            start_t   = (max(c['start'] + (c['trim_out'] - c['trim_in'])
                              for c in trk_clips)
                          if trk_clips else 0.0)
            new_clip = {
                'id':           str(uuid.uuid4()),
                'path':         path,
                'track':        trk,
                'start':        start_t,
                'trim_in':      0.0,
                'trim_out':     dur,
                'scale':        1.0,
                'x':            'center',
                'y':            'center',
                'volume':       1.0,
                'speed':        1.0,
                'mute_audio':   False,
                'rm_bg':        False,
                'custom_masks': {},
                'aspect':       'Original',
                'rot':          '0°',
                'wm_text':      '',
                'wm_pos':       'Bottom Right',
                'transition_type': 'None',
                'transition_dur':  0.5,
                'notes':        '',
            }
            if _HAS_COMMANDS and self.cmd_stack is not None:
                self.cmd_stack.push(AddClipCommand(self, new_clip))
            else:
                self.project_clips.append(new_clip)
                self.rebuild_proxy_cache()
                self.timeline.draw()
            log.info("Inserted '%s' -> %s at t=%.2f", os.path.basename(path), trk, start_t)
        except Exception as e:
            messagebox.showerror("Import Error", str(e))
            log.error("insert_to_timeline failed: %s", e)

    # ------------------------------------------------------------------
    # Inspector
    # ------------------------------------------------------------------
    def update_inspector(self):
        cid = self.timeline.active_clip_id
        if not cid:
            self.insp_c.pack_forget()
            self.insp_lbl.pack(pady=30)
            return
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            return
        self.insp_lbl.pack_forget()
        self.insp_c.pack(fill="both", expand=True)

        self.var_tin.set(f"{c['trim_in']:.3f}")
        self.var_tout.set(f"{c['trim_out']:.3f}")
        scale = c.get('scale', 1.0)
        self.var_scale.set(scale)
        self.lbl_sc.configure(text=f"{scale:.2f}×")
        vol = c.get('volume', 1.0)
        self.var_vol.set(vol)
        self.lbl_vol.configure(text=f"{int(vol * 100)}%")
        spd = c.get('speed', 1.0)
        self.var_speed.set(spd)
        self.lbl_sp.configure(text=f"{spd:.2f}×")
        self.var_x.set(str(c.get('x', 'center')))
        self.var_y.set(str(c.get('y', 'center')))
        self.var_aspect.set(c.get('aspect', 'Original'))
        crop_box = c.get('crop')
        if hasattr(self, 'lbl_crop_status'):
            if crop_box and isinstance(crop_box, dict):
                self.lbl_crop_status.configure(text="Custom Box", text_color="#38bdf8")
                self.btn_reset_crop.configure(state="normal")
            else:
                self.lbl_crop_status.configure(text="Full Frame", text_color="#94a3b8")
                self.btn_reset_crop.configure(state="disabled")
        self.var_rot.set(c.get('rot', '0°'))
        self.var_wm.set(c.get('wm_text', ''))
        self.var_wmpos.set(c.get('wm_pos', 'Bottom Right'))
        self.var_rmbg.set(c.get('rm_bg', False))
        self.var_mute.set(c.get('mute_audio', False))
        self.var_transition.set(c.get('transition_type', 'None'))
        self.var_trans_dur.set(str(c.get('transition_dur', 0.5)))
        self.var_notes.set(c.get('notes', ''))

        # In/Out marker labels
        tin  = c['trim_in']
        tout = c['trim_out']
        self.lbl_inout.configure(text=f"In: {tin:.3f}s   Out: {tout:.3f}s")

        # Colour grading
        if self.color_grading_panel is not None:
            self.color_grading_panel.load_from_clip(c)

        # Effects registry — restore per-effect enable + param state
        if _HAS_EFFECTS and hasattr(self, '_effects_widgets'):
            clip_effects = c.get('effects', {})
            for eff_name, eff_widgets in self._effects_widgets.items():
                saved = clip_effects.get(eff_name, {})
                eff_widgets['enabled_var'].set(saved.get('enabled', False))
                for p_name, p_var in eff_widgets.get('param_vars', {}).items():
                    p_var.set(saved.get(p_name, p_var.get()))

    def set_trim(self, mode):
        cid = self.timeline.active_clip_id
        c   = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            return
        t_local = self.playhead_t - c['start'] + c['trim_in']
        if mode == 'in':
            self.var_tin.set(f"{max(0.0, t_local):.3f}")
        else:
            self.var_tout.set(f"{t_local:.3f}")

    def apply_inspector_changes(self):
        cid = self.timeline.active_clip_id
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            self.show_toast("Changes failed: No clip selected.", success=False)
            messagebox.showwarning("No Clip Selected", "Please select a clip on the timeline first.")
            return
        try:
            tin_str = self.var_tin.get().strip()
            tout_str = self.var_tout.get().strip()
            tin  = float(tin_str) if tin_str else 0.0
            tout = float(tout_str) if tout_str else float(c.get('trim_out', 0.0))
            if tout <= tin:
                self.show_toast("Changes failed: Out point must be after In point.", success=False)
                messagebox.showerror("Invalid Trim", "Out point must be after In point.")
                return

            # Capture old props for undo
            old_props = copy.deepcopy(c)

            c['trim_in']  = tin
            c['trim_out'] = tout
            c['scale']    = float(self.var_scale.get())
            c['speed']    = float(self.var_speed.get())
            c['volume']   = float(self.var_vol.get())
            c['mute_audio'] = bool(self.var_mute.get())
            x = self.var_x.get().strip()
            y = self.var_y.get().strip()
            c['x'] = int(x) if (x.startswith('-') and x[1:].isdigit()) or x.isdigit() else x
            c['y'] = int(y) if (y.startswith('-') and y[1:].isdigit()) or y.isdigit() else y
            c['aspect']   = self.var_aspect.get()
            c['rot']      = self.var_rot.get()
            c['wm_text']  = self.var_wm.get()
            c['wm_pos']   = self.var_wmpos.get()
            c['rm_bg']    = bool(self.var_rmbg.get())
            c['transition_type'] = self.var_transition.get()
            try:
                c['transition_dur'] = float(self.var_trans_dur.get())
            except ValueError:
                c['transition_dur'] = 0.5
            c['notes'] = self.var_notes.get()

            if self.color_grading_panel is not None:
                self.color_grading_panel.save_to_clip(c)

            # Save effects registry state into clip dict
            if _HAS_EFFECTS and hasattr(self, '_effects_widgets'):
                effects_out = {}
                for eff_name, eff_widgets in self._effects_widgets.items():
                    enabled = eff_widgets['enabled_var'].get()
                    entry = {'enabled': enabled}
                    for p_name, p_var in eff_widgets.get('param_vars', {}).items():
                        entry[p_name] = p_var.get()
                    effects_out[eff_name] = entry
                c['effects'] = effects_out

            new_props = copy.deepcopy(c)

            if _HAS_COMMANDS and self.cmd_stack is not None:
                self.cmd_stack.push(InspectorChangeCommand(self, cid, old_props, new_props))
            else:
                self.rebuild_proxy_cache()
                self.timeline.draw()

            self.show_toast("Changes applied successfully!", success=True)
            if hasattr(self, 'lbl_status'):
                clip_label = c.get('label') or os.path.basename(c.get('path', '')) or 'Clip'
                self.lbl_status.configure(text=f"Changes applied to: {clip_label}")
        except ValueError as e:
            self.show_toast(f"Changes failed to apply: {e}", success=False)
            messagebox.showerror("Invalid Value", f"Failed to parse property value: {e}")
        except Exception as e:
            self.show_toast(f"Changes failed to apply: {e}", success=False)
            messagebox.showerror("Error", f"Failed to apply changes: {e}")

    def _reset_clip_crop(self):
        """Reset crop for the active clip to full frame (None)."""
        cid = getattr(self.timeline, 'active_clip_id', None)
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            self.show_toast("No clip selected.", success=False)
            return
        old_props = copy.deepcopy(c)
        c['crop'] = None
        c['aspect'] = 'Original'
        self.var_aspect.set('Original')
        new_props = copy.deepcopy(c)
        if _HAS_COMMANDS and self.cmd_stack is not None:
            self.cmd_stack.push(InspectorChangeCommand(self, cid, old_props, new_props))
        else:
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()
        self.show_toast("Crop reset to full frame.", success=True)

    def open_crop_tool(self):
        """Launch the interactive visual Crop & Aspect Ratio dialog for the active clip."""
        cid = getattr(self.timeline, 'active_clip_id', None)
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            self.show_toast("Please select a video or image clip first.", success=False)
            messagebox.showinfo("Crop & Aspect Ratio Tool", "Please select a clip on the timeline first.")
            return

        trk = c.get('track', '')
        if trk.startswith('A'):
            self.show_toast("Audio clips cannot be cropped.", success=False)
            messagebox.showinfo("Crop & Aspect Ratio Tool", "Audio clips have no visual frames to crop.")
            return

        path = c.get('path', '')
        frame = None
        if path == '__title__' or 'title_config' in c or 'title_text' in c:
            try:
                from title_generator import TitleClip
                if 'title_config' in c:
                    tc = TitleClip.from_config(c['title_config'])
                else:
                    tc = TitleClip(
                        text=c.get('title_text', 'Title'),
                        duration=1.0,
                        fps=30.0,
                        width=self.proj_w,
                        height=self.proj_h,
                    )
                mpy = tc.to_moviepy_clip()
                frame = mpy.get_frame(0.0)
                mpy.close()
            except Exception as exc:
                log.warning("Could not extract title frame for crop: %s", exc)
        elif is_image_file(path) or trk.startswith('I'):
            try:
                bgr = cv2.imread(path)
                if bgr is not None:
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                log.warning("Could not read image for crop: %s", exc)
        else:
            try:
                cap = cv2.VideoCapture(path)
                t_local = max(0.0, self.playhead_t - c.get('start', 0.0) + c.get('trim_in', 0.0))
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_idx = int(t_local * src_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, bgr = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(c.get('trim_in', 0.0) * src_fps))
                    ret, bgr = cap.read()
                cap.release()
                if ret and bgr is not None:
                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                log.warning("Could not extract video frame for crop: %s", exc)

        if frame is None:
            frame = np.zeros((self.proj_h, self.proj_w, 3), dtype=np.uint8)
            cv2.putText(frame, "Clip Preview Frame", (self.proj_w // 4, self.proj_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)

        def _on_crop_applied(crop_box, aspect_str):
            old_props = copy.deepcopy(c)
            c['crop'] = crop_box
            c['aspect'] = aspect_str
            self.var_aspect.set(aspect_str)
            new_props = copy.deepcopy(c)
            if _HAS_COMMANDS and self.cmd_stack is not None:
                self.cmd_stack.push(InspectorChangeCommand(self, cid, old_props, new_props))
            else:
                self.rebuild_proxy_cache()
                self.timeline.draw()
                self.update_inspector()
            self.show_toast(f"Crop & Aspect ({aspect_str}) applied!", success=True)

        if not _HAS_CROP_TOOL:
            messagebox.showerror("Error", "crop_tool module not available.")
            return

        CropAspectDialog(self, clip_data=c, frame_rgb=frame, on_apply=_on_crop_applied)

    # ------------------------------------------------------------------
    # Keyframe editor
    def _build_effects_registry_panel(self):
        """
        Populate self._effects_frame with one collapsible row per registered effect.
        Each row has: [✓ Enable checkbox] [Effect Label] [param sliders/entries].
        """
        if not _HAS_EFFECTS:
            return
        parent = self._effects_frame

        # Group by category
        by_cat = _effects_registry.by_category()
        for cat_name, effects in by_cat.items():
            # Category label
            ctk.CTkLabel(parent, text=cat_name.upper(),
                          font=("Segoe UI", 8, "bold"),
                          text_color="#888888").pack(anchor="w", padx=6, pady=(6, 1))

            for eff in effects:
                eff_frame = ctk.CTkFrame(parent, fg_color="#222222", corner_radius=4)
                eff_frame.pack(fill="x", padx=4, pady=2)

                # Enable toggle
                enabled_var = ctk.BooleanVar(value=False)
                ctk.CTkCheckBox(eff_frame, text=eff.label, variable=enabled_var,
                                font=("Segoe UI", 9), width=170
                                ).pack(side="left", padx=6, pady=3)

                param_vars: dict = {}
                # Build compact param controls
                for param in eff.params:
                    if param.type == "bool":
                        # bool params are represented by the enabled checkbox itself; skip
                        continue
                    p_frame = ctk.CTkFrame(eff_frame, fg_color="transparent")
                    p_frame.pack(side="left", padx=4)
                    ctk.CTkLabel(p_frame, text=param.label,
                                  font=("Segoe UI", 7), text_color="#777777",
                                  width=44, anchor="w").pack(side="left")
                    if param.type in ("float", "int"):
                        p_var = ctk.DoubleVar(value=float(param.default))
                        steps = max(1, int((param.max_val - param.min_val) / max(param.step, 1e-9)))
                        slider = ctk.CTkSlider(p_frame,
                                               from_=param.min_val, to=param.max_val,
                                               variable=p_var,
                                               number_of_steps=min(steps, 200),
                                               width=80, height=14)
                        slider.pack(side="left", padx=2)
                    elif param.type == "choice":
                        p_var = ctk.StringVar(value=str(param.default))
                        ctk.CTkComboBox(p_frame, variable=p_var,
                                         values=param.choices, width=80).pack(side="left", padx=2)
                    else:
                        p_var = ctk.StringVar(value=str(param.default))
                        ctk.CTkEntry(p_frame, textvariable=p_var, width=60).pack(side="left", padx=2)
                    param_vars[param.name] = p_var

                self._effects_widgets[eff.name] = {
                    'enabled_var': enabled_var,
                    'param_vars':  param_vars,
                }

    # ------------------------------------------------------------------
    def _open_keyframe_editor(self):
        c = next((x for x in self.project_clips if x['id'] == self.timeline.active_clip_id), None)
        if c:
            if _HAS_KEYFRAMES:
                KeyframeEditor(self, c)
            else:
                messagebox.showinfo("Keyframes", "keyframes module not found.")

    # ------------------------------------------------------------------
    # Clip label
    # ------------------------------------------------------------------
    def set_clip_label(self, clip_id: str, label_name):
        c = next((x for x in self.project_clips if x['id'] == clip_id), None)
        if c:
            c['label'] = label_name
            self.timeline.draw()

    # ------------------------------------------------------------------
    # Clip operations (also called from timeline context menu)
    # ------------------------------------------------------------------
    def delete_clip(self, clip_id):
        c = next((x for x in self.project_clips if x['id'] == clip_id), None)
        if not c:
            return
        if _HAS_COMMANDS and self.cmd_stack is not None:
            self.cmd_stack.push(DeleteClipCommand(self, c))
        else:
            self.project_clips = [x for x in self.project_clips if x['id'] != clip_id]
            if self.timeline.active_clip_id == clip_id:
                self.timeline.active_clip_id = None
            self.update_inspector()
            self.rebuild_proxy_cache()
            self.timeline.draw()

    def duplicate_clip(self, clip_id):
        if not clip_id:
            return
        src = next((c for c in self.project_clips if c['id'] == clip_id), None)
        if not src:
            return
        if _HAS_COMMANDS and self.cmd_stack is not None:
            self.cmd_stack.push(DuplicateClipCommand(self, clip_id, str(uuid.uuid4())))
        else:
            dup = copy.deepcopy(src)
            dup['id']    = str(uuid.uuid4())
            dup['start'] = src['start'] + (src['trim_out'] - src['trim_in'])
            self.project_clips.append(dup)
            self.timeline.active_clip_id = dup['id']
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()

    def split_clip_at_playhead(self, clip_id):
        if not clip_id:
            return
        c = next((x for x in self.project_clips if x['id'] == clip_id), None)
        if not c:
            return
        t_local = self.playhead_t - c['start'] + c['trim_in']
        if not (c['trim_in'] < t_local < c['trim_out']):
            messagebox.showinfo("Split", "Playhead must be inside the clip to split.")
            return
        if _HAS_COMMANDS and self.cmd_stack is not None:
            right_id = str(uuid.uuid4())
            self.cmd_stack.push(SplitClipCommand(self, c, clip_id, right_id, t_local))
        else:
            right = copy.deepcopy(c)
            right['id']       = str(uuid.uuid4())
            right['trim_in']  = t_local
            right['start']    = self.playhead_t
            c['trim_out']     = t_local
            self.project_clips.append(right)
            self.rebuild_proxy_cache()
            self.timeline.draw()

    def separate_audio(self, clip_id: str):
        """Extract audio from video clip into an audio track and link them."""
        clip = next((x for x in self.project_clips if x['id'] == clip_id), None)
        if not clip:
            return

        # Find target audio track
        a_tracks = [k for k in self.timeline.track_keys if k.startswith('A')]
        if not a_tracks:
            self.timeline.add_audio_track()
            a_tracks = [k for k in self.timeline.track_keys if k.startswith('A')]
        target_track = a_tracks[0]

        audio_id = str(uuid.uuid4())
        audio_clip = make_clip(
            path=clip['path'],
            track=target_track,
            start=clip['start'],
            trim_in=clip['trim_in'],
            trim_out=clip['trim_out'],
            volume=clip.get('volume', 1.0),
            mute_audio=False,
            linked_clip_id=clip['id'],
        )
        audio_clip['id'] = audio_id

        if _HAS_COMMANDS and self.cmd_stack is not None:
            cmd = SeparateAudioCommand(self, clip['id'], audio_clip)
            self.cmd_stack.push(cmd)
        else:
            clip['mute_audio'] = True
            clip['linked_clip_id'] = audio_id
            self.project_clips.append(audio_clip)
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()

    def toggle_clip_link(self, clip_id: str):
        """Toggle link/unlink state for a clip and its companion."""
        clip = next((x for x in self.project_clips if x['id'] == clip_id), None)
        if not clip:
            return
        linked_id = clip.get('linked_clip_id')
        if linked_id:
            if _HAS_COMMANDS and self.cmd_stack is not None:
                self.cmd_stack.push(ToggleLinkCommand(self, clip_id, linked_id))
            else:
                clip['linked_clip_id'] = None
                companion = next((x for x in self.project_clips if x['id'] == linked_id), None)
                if companion:
                    companion['linked_clip_id'] = None
                self.timeline.draw()
        else:
            other_ids = [cid for cid in self.timeline.selected_clip_ids if cid != clip_id]
            if other_ids:
                self.link_clips(clip_id, other_ids[0])

    def link_clips(self, clip1_id: str, clip2_id: str):
        """Link two clips together so their move/trim/split operations stay synchronized."""
        if _HAS_COMMANDS and self.cmd_stack is not None:
            self.cmd_stack.push(ToggleLinkCommand(self, clip1_id, clip2_id))
        else:
            c1 = next((x for x in self.project_clips if x['id'] == clip1_id), None)
            c2 = next((x for x in self.project_clips if x['id'] == clip2_id), None)
            if c1 and c2:
                c1['linked_clip_id'] = clip2_id
                c2['linked_clip_id'] = clip1_id
                self.timeline.draw()

    def slip_clip(self, clip_id: str, delta: float):
        """Slip clip: shift source in and out points by delta."""
        if not clip_id:
            return
        c = next((x for x in self.project_clips if x['id'] == clip_id), None)
        if not c:
            return
        old_in = c['trim_in']
        old_out = c['trim_out']
        dur = old_out - old_in
        new_in = max(0.0, old_in + delta)
        new_out = new_in + dur
        if _HAS_COMMANDS and self.cmd_stack is not None:
            self.cmd_stack.push(SlipClipCommand(self, clip_id, old_in, old_out, new_in, new_out))
        else:
            c['trim_in'] = new_in
            c['trim_out'] = new_out
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()

    def slide_clip(self, clip_id: str, delta: float):
        """Slide clip: shift clip start along timeline and trim adjacent neighbors."""
        if not clip_id:
            return
        c = next((x for x in self.project_clips if x['id'] == clip_id), None)
        if not c:
            return
        trk = c.get('track')
        same_trk = sorted([x for x in self.project_clips if x.get('track') == trk], key=lambda x: x['start'])
        idx = next((i for i, x in enumerate(same_trk) if x['id'] == clip_id), None)
        prev_id = same_trk[idx - 1]['id'] if (idx is not None and idx > 0) else None
        next_id = same_trk[idx + 1]['id'] if (idx is not None and idx < len(same_trk) - 1) else None

        if _HAS_COMMANDS and self.cmd_stack is not None:
            self.cmd_stack.push(SlideClipCommand(self, clip_id, delta, prev_id, next_id))
        else:
            c['start'] = max(0.0, c['start'] + delta)
            if prev_id:
                prev_c = next((x for x in self.project_clips if x['id'] == prev_id), None)
                if prev_c:
                    prev_c['trim_out'] = max(prev_c['trim_in'] + 0.1, prev_c['trim_out'] + delta)
            if next_id:
                next_c = next((x for x in self.project_clips if x['id'] == next_id), None)
                if next_c:
                    next_c['start'] += delta
                    next_c['trim_in'] = max(0.0, next_c['trim_in'] + delta)
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()

    # ------------------------------------------------------------------
    # Proxy / preview
    # ------------------------------------------------------------------
    def rebuild_proxy_cache(self):
        if self.is_playing:
            self.toggle_play()

        # Push beat markers to timeline
        if hasattr(self, 'timeline') and hasattr(self.timeline, 'beat_markers'):
            self.timeline.beat_markers = list(self._beat_markers)

        if not self.project_clips:
            # Close existing clips
            for c in self.open_proxy_clips:
                try: c.close()
                except Exception: pass
            self.open_proxy_clips = []
            self.master_proxy_clip = None
            self.lbl_preview.configure(image=None, text="No Sequence")
            self._update_scrub_range(1.0)
            return

        if _HAS_PROXY_WORKER:
            # Async path — offload to background thread
            self._proxy_building = True
            self.lbl_status.configure(text="Building preview…")
            t_vols = getattr(self.timeline, 'track_volumes', None) if hasattr(self, 'timeline') else None
            t_pans = getattr(self.timeline, 'track_pans', None) if hasattr(self, 'timeline') else None
            t_mute = getattr(self.timeline, 'track_muted', None) if hasattr(self, 'timeline') else None
            t_solo = getattr(self.timeline, 'track_solo', None) if hasattr(self, 'timeline') else None
            t_spds = getattr(self.timeline, 'track_speeds', None) if hasattr(self, 'timeline') else None
            _proxy_worker.request(
                self.project_clips,
                self.proxy_w, self.proxy_h,
                self.proj_w,  self.proj_h,
                self,
                self._on_proxy_ready,
                track_volumes=t_vols,
                track_pans=t_pans,
                track_muted=t_mute,
                track_solo=t_solo,
                track_speeds=t_spds,
            )
        else:
            # Synchronous fallback (blocks UI — kept for compatibility)
            for c in self.open_proxy_clips:
                try: c.close()
                except Exception: pass
            self.open_proxy_clips = []
            max_dur = self._get_sequence_total_duration()
            v_layers = [ColorClip(size=(self.proxy_w, self.proxy_h),
                                   color=(0, 0, 0), duration=max_dur)]
            v_data = sorted(
                [c for c in self.project_clips if c['track'] in ('V1', 'V2')],
                key=lambda x: 0 if x['track'] == 'V1' else 1
            )
            for d in v_data:
                try:
                    clip = (VideoFileClip(d['path'])
                            .subclip(d['trim_in'], d['trim_out'])
                            .resize(height=self.proxy_h))
                    # Custom Crop Box (normalized 0.0 - 1.0: x1, y1, x2, y2)
                    crop_box = d.get('crop')
                    if crop_box and isinstance(crop_box, dict):
                        cw, ch = clip.size
                        cx1 = int(round(float(crop_box.get('x1', 0.0)) * cw))
                        cy1 = int(round(float(crop_box.get('y1', 0.0)) * ch))
                        cx2 = int(round(float(crop_box.get('x2', 1.0)) * cw))
                        cy2 = int(round(float(crop_box.get('y2', 1.0)) * ch))
                        cx1 = max(0, min(cx1, cw - 2))
                        cy1 = max(0, min(cy1, ch - 2))
                        cx2 = max(cx1 + 2, min(cx2, cw))
                        cy2 = max(cy1 + 2, min(cy2, ch))
                        clip = clip.crop(x1=cx1, y1=cy1, x2=cx2, y2=cy2)

                    # Aspect ratio crop
                    aspect = d.get('aspect', 'Original')
                    if aspect and aspect != 'Original':
                        cw, ch = clip.size
                        ratio_map = {
                            '16:9': 16/9, '9:16': 9/16, '1:1': 1.0, '4:3': 4/3,
                            '21:9': 21/9, '4:5': 4/5, '2:3': 2/3
                        }
                        tr = ratio_map.get(aspect, cw / ch)
                        cr = cw / ch
                        if cr > tr + 0.005:
                            new_w = max(2, int(round(ch * tr)))
                            clip = clip.crop(x1=(cw - new_w) // 2, y1=0, width=new_w, height=ch)
                        elif cr < tr - 0.005:
                            new_h = max(2, int(round(cw / tr)))
                            clip = clip.crop(x1=0, y1=(ch - new_h) // 2, width=cw, height=new_h)

                    if d.get('scale', 1.0) != 1.0:
                        clip = clip.resize(d['scale'])
                    px = (d.get('x') if isinstance(d.get('x'), str)
                          else int(d['x'] * self.proxy_w / self.proj_w))
                    py = (d.get('y') if isinstance(d.get('y'), str)
                          else int(d['y'] * self.proxy_h / self.proj_h))
                    clip = clip.set_pos((px, py)).set_start(d['start'])
                    v_layers.append(clip)
                except Exception as exc:
                    log.warning("Proxy build skipped '%s': %s", d.get('path', ''), exc)
            self.master_proxy_clip = (
                CompositeVideoClip(v_layers, size=(self.proxy_w, self.proxy_h))
                .set_duration(max_dur)
            )
            self._update_scrub_range(max_dur)
            self.scrub_to(min(self.playhead_t, max_dur))

    def _on_proxy_ready(self, master_clip, open_clips):
        """Called on the UI thread when the async proxy build finishes."""
        # Close old clips
        for c in self.open_proxy_clips:
            try: c.close()
            except Exception: pass
        self.open_proxy_clips = open_clips or []
        self.master_proxy_clip = master_clip
        self._proxy_building = False
        if master_clip:
            self._update_scrub_range(master_clip.duration)
            self.scrub_to(min(self.playhead_t, master_clip.duration))
            self.lbl_status.configure(text="Ready.")
        else:
            self.lbl_preview.configure(image=None, text="No Sequence")
            self.lbl_status.configure(text="Preview unavailable.")
        self.timeline.draw()

    def _update_scrub_range(self, duration):
        self.scrub_slider.configure(to=max(duration, 0.01))

    def _on_scrub_slider(self, value):
        self.scrub_to(float(value))

    def scrub_to(self, t):
        self.playhead_t = max(0.0, t)
        self.scrub_var.set(self.playhead_t)
        self.timeline.set_playhead(self.playhead_t)

        total = self.playhead_t
        h_part, remainder = divmod(total, 3600)
        m_part, s_part    = divmod(remainder, 60)
        self.lbl_time.configure(
            text=f"{int(h_part):02d}:{int(m_part):02d}:{s_part:06.3f}"
        )

        if not self.master_proxy_clip:
            return

        if self.is_playing:
            if self._frame_queue is not None:
                self._frame_queue.start_playback(
                    self.master_proxy_clip,
                    self.fps,
                    self.playhead_t,
                    self,
                    self._on_playback_frame,
                    self._on_playback_end,
                )
            return

        try:
            t_clamped = min(self.playhead_t,
                            self.master_proxy_clip.duration - 0.001)
            frame = self.master_proxy_clip.get_frame(max(0.0, t_clamped))
            pim   = Image.fromarray(frame)
            pw    = self.preview_frame.winfo_width() - 10
            ph    = self.preview_frame.winfo_height() - 68
            if pw > 10 and ph > 10:
                pim.thumbnail((pw, ph), Image.Resampling.BILINEAR)
            cim = ctk.CTkImage(light_image=pim, dark_image=pim, size=pim.size)
            self.lbl_preview.configure(image=cim, text="")
            self.lbl_preview._image = cim
            if hasattr(self, '_scopes_window') and self._scopes_window and self._scopes_window.winfo_exists():
                self._scopes_window.update_scope(frame)
        except Exception as exc:
            log.debug("scrub_to frame error: %s", exc)

    def open_video_scopes(self):
        """Open or focus the floating Studio Video Scopes window."""
        if not _HAS_VIDEO_SCOPES:
            messagebox.showinfo("Video Scopes", "video_scopes module not found.")
            return
        if self._scopes_window is None or not self._scopes_window.winfo_exists():
            self._scopes_window = VideoScopesWindow(self)
        else:
            self._scopes_window.lift()
            self._scopes_window.focus_force()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def toggle_play(self):
        if not self.master_proxy_clip:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.configure(text="⏸ Pause")
            if self._frame_queue is not None:
                # Smooth wall-clock-accurate playback via FrameQueue
                self._frame_queue.start_playback(
                    self.master_proxy_clip,
                    self.fps,
                    self.playhead_t,
                    self,
                    self._on_playback_frame,
                    self._on_playback_end,
                )
            else:
                self._play_loop()
        else:
            self.btn_play.configure(text="▶ Play")
            if hasattr(self, "vu_meter") and self.vu_meter:
                self.vu_meter.reset()
            if self._frame_queue is not None:
                self._frame_queue.stop()
            if self._play_after_id:
                self.after_cancel(self._play_after_id)
                self._play_after_id = None

    def _on_playback_frame(self, frame, t: float):
        """Receive a decoded frame from FrameQueue on the UI thread."""
        if not self.is_playing:
            return
        self.playhead_t = t
        self.scrub_var.set(t)
        self.timeline.set_playhead(t)
        # Update timecode
        h_part, rem = divmod(t, 3600)
        m_part, s_part = divmod(rem, 60)
        self.lbl_time.configure(
            text=f"{int(h_part):02d}:{int(m_part):02d}:{s_part:06.3f}"
        )
        # Fallback audio meter update if direct sounddevice callback is inactive
        if hasattr(self, "vu_meter") and self.vu_meter:
            if getattr(self, "is_muted", False):
                self.vu_meter.reset()
            elif getattr(self.master_proxy_clip, "audio", None):
                try:
                    if not hasattr(self, '_frame_queue') or self._frame_queue is None or getattr(self._frame_queue, '_audio_stream', None) is None:
                        a_frame = self.master_proxy_clip.audio.get_frame(t)
                        if a_frame is not None:
                            a_arr = np.asarray(a_frame)
                            if a_arr.ndim >= 1 and a_arr.size > 0:
                                if a_arr.ndim == 1:
                                    pk = float(np.max(np.abs(a_arr)))
                                    self.vu_meter.set_levels(pk, pk)
                                else:
                                    self.vu_meter.set_levels(float(np.max(np.abs(a_arr[:, 0]))), float(np.max(np.abs(a_arr[:, 1]))))
                except Exception:
                    pass
        # Render frame to preview
        try:
            pim = Image.fromarray(frame)
            pw  = self.preview_frame.winfo_width() - 10
            ph  = self.preview_frame.winfo_height() - 68
            if pw > 10 and ph > 10:
                pim.thumbnail((pw, ph), Image.Resampling.BILINEAR)
            cim = ctk.CTkImage(light_image=pim, dark_image=pim, size=pim.size)
            self.lbl_preview.configure(image=cim, text="")
            self.lbl_preview._image = cim
            if hasattr(self, '_scopes_window') and self._scopes_window and self._scopes_window.winfo_exists():
                self._scopes_window.update_scope(frame)
        except Exception as exc:
            log.debug("_on_playback_frame render error: %s", exc)

    def _on_playback_end(self):
        """Called by FrameQueue when the clip reaches its end."""
        if getattr(self, "is_looping", False) and self.master_proxy_clip:
            self.scrub_to(0.0)
            self.is_playing = True
            self.btn_play.configure(text="⏸ Pause")
            if self._frame_queue is not None:
                self._frame_queue.start_playback(
                    self.master_proxy_clip,
                    self.fps,
                    0.0,
                    self,
                    self._on_playback_frame,
                    self._on_playback_end,
                )
            else:
                self._play_loop()
            return
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")
        if hasattr(self, "vu_meter") and self.vu_meter:
            self.vu_meter.reset()
        self.scrub_to(0.0)

    def _play_loop(self):
        """Fallback play loop used when FrameQueue is not available."""
        if not self.is_playing or not self.master_proxy_clip:
            return
        nxt = self.playhead_t + 1.0 / self.fps
        if nxt >= self.master_proxy_clip.duration:
            if getattr(self, "is_looping", False):
                self.scrub_to(0.0)
                interval = max(1, int(1000.0 / self.fps))
                self._play_after_id = self.after(interval, self._play_loop)
                return
            self.is_playing = False
            self.btn_play.configure(text="▶ Play")
            if hasattr(self, "vu_meter") and self.vu_meter:
                self.vu_meter.reset()
            self.scrub_to(0.0)
            return
        self.scrub_to(nxt)
        interval = max(1, int(1000.0 / self.fps))
        self._play_after_id = self.after(interval, self._play_loop)

    # ------------------------------------------------------------------
    # Mask preview / touchup
    # ------------------------------------------------------------------
    def preview_mask(self):
        cid = self.timeline.active_clip_id
        c   = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c or 'V' not in c['track']:
            return
        cap     = cv2.VideoCapture(c['path'])
        t_local = self.playhead_t - c['start'] + c['trim_in']
        cap.set(cv2.CAP_PROP_POS_FRAMES,
                int(t_local * (cap.get(cv2.CAP_PROP_FPS) or 30)))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        from engine import BGREngine
        frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = BGREngine(c.get('custom_masks', {})).get(frgb, int(t_local * 30))
        ov   = cv2.addWeighted(frgb, 0.7,
                                cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.3, 0)
        pim  = Image.fromarray(ov)
        pw   = self.preview_frame.winfo_width() - 10
        ph   = self.preview_frame.winfo_height() - 68
        pim.thumbnail((pw, ph), Image.Resampling.LANCZOS)
        cim = ctk.CTkImage(light_image=pim, dark_image=pim, size=pim.size)
        self.lbl_preview.configure(image=cim, text="")
        self.lbl_preview._image = cim

    def open_touchup(self):
        cid = self.timeline.active_clip_id
        c   = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c or 'V' not in c['track']:
            return
        cap     = cv2.VideoCapture(c['path'])
        t_local = self.playhead_t - c['start'] + c['trim_in']
        idx     = int(t_local * (cap.get(cv2.CAP_PROP_FPS) or 30))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = c.get('custom_masks', {}).get(idx)
        if mask is None:
            from engine import BGREngine
            mask = BGREngine(None).get(frgb, 0)
        MaskPainter(self, idx, frgb, mask, c)

    # ------------------------------------------------------------------
    # Project save / load / new
    # ------------------------------------------------------------------
    def new_project(self):
        if messagebox.askyesno("New Project", "Discard current project and start fresh?"):
            self.project_clips.clear()
            self.media_bin.clear()
            self.selected_media_index = None
            self._refresh_media_bin_ui()
            self._on_media_select()
            self.timeline.active_clip_id = None
            self.timeline.selected_clip_ids = set()
            if self.cmd_stack is not None:
                self.cmd_stack.clear()
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()
            self._current_project_path = None
            self.title("Video Studio")

    def save_project_as(self):
        path = filedialog.asksaveasfilename(
            title="Save Project As",
            defaultextension=".nleproj",
            filetypes=[("NLE Project", "*.nleproj"), ("All Files", "*.*")]
        )
        if path:
            self._save_to_path(path)

    def save_project(self):
        if not self._current_project_path:
            self.save_project_as()
        else:
            self._save_to_path(self._current_project_path)

    def _save_to_path(self, path):
        if not _HAS_PROJECT_IO:
            messagebox.showerror("Save Failed", "project_io module not found.")
            return
        try:
            save_project(path, self.project_clips, self.proj_w, self.proj_h, self.fps)
            self._current_project_path = path
            add_recent_project(config_mgr, path)
            self.title(f"Video Studio — {os.path.basename(path)}")
            if _HAS_AUTOSAVE:
                clear_autosave()
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))

    def open_project(self, path=None):
        if path is None:
            path = filedialog.askopenfilename(
                title="Open Project",
                filetypes=[("NLE Project", "*.nleproj"), ("All Files", "*.*")]
            )
        if not path:
            return
        if not _HAS_PROJECT_IO:
            messagebox.showerror("Open Failed", "project_io module not found.")
            return
        try:
            clips, w, h, fps = load_project(path)
            self.project_clips = clips
            self.proj_w, self.proj_h, self.fps = w, h, fps
            self.media_bin = list({c['path'] for c in clips if c.get('path') != '__title__'})
            self.selected_media_index = 0 if self.media_bin else None
            self._refresh_media_bin_ui()
            self._on_media_select()
            self.timeline.active_clip_id = None
            self.timeline.selected_clip_ids = set()
            if self.cmd_stack is not None:
                self.cmd_stack.clear()
            self._current_project_path = path
            add_recent_project(config_mgr, path)
            self.lbl_seq.configure(text=f"{w}×{h}  {fps}fps")
            self.title(f"Video Studio — {os.path.basename(path)}")
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()
        except Exception as e:
            messagebox.showerror("Open Failed", str(e))

    # ------------------------------------------------------------------
    # Studio Sequence Interchange (EDL & FCP7 XML)
    # ------------------------------------------------------------------
    def export_sequence(self):
        """Export project timeline as standard CMX 3600 EDL or Apple FCP7 XML."""
        if not self.project_clips:
            messagebox.showwarning("Export Sequence", "No clips on the timeline to export.", parent=self)
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Sequence Interchange",
            filetypes=[
                ("Final Cut Pro 7 XML (*.xml)", "*.xml"),
                ("CMX 3600 EDL (*.edl)", "*.edl"),
            ],
            defaultextension=".xml"
        )
        if not path:
            return

        try:
            import interchange
            if path.lower().endswith(".edl"):
                interchange.export_edl(self.project_clips, path, fps=self.fps)
            else:
                interchange.export_fcpxml(self.project_clips, path, proj_w=self.proj_w, proj_h=self.proj_h, fps=self.fps)
            messagebox.showinfo("Export Sequence", f"Sequence exported successfully:\n{path}", parent=self)
        except Exception as e:
            log.exception("Export sequence failed")
            messagebox.showerror("Export Sequence Failed", str(e), parent=self)

    def import_sequence(self):
        """Import CMX 3600 EDL or Apple FCP7 XML into the timeline."""
        path = filedialog.askopenfilename(
            parent=self,
            title="Import Sequence Interchange",
            filetypes=[
                ("Supported Sequence Files (*.xml, *.edl)", "*.xml;*.edl"),
                ("Final Cut Pro 7 XML (*.xml)", "*.xml"),
                ("CMX 3600 EDL (*.edl)", "*.edl"),
                ("All Files (*.*)", "*.*"),
            ]
        )
        if not path:
            return

        try:
            import interchange
            if path.lower().endswith(".edl"):
                clips = interchange.import_edl(path, fps=self.fps)
            else:
                clips = interchange.import_fcpxml(path, fps=self.fps)

            if not clips:
                messagebox.showwarning("Import Sequence", "No clips found in the imported sequence.", parent=self)
                return

            self.project_clips.extend(clips)
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()
            messagebox.showinfo("Import Sequence", f"Imported {len(clips)} clip(s) from:\n{path}", parent=self)
        except Exception as e:
            log.exception("Import sequence failed")
            messagebox.showerror("Import Sequence Failed", str(e), parent=self)


    # ------------------------------------------------------------------
    # Export queue
    # ------------------------------------------------------------------
    def open_export_queue(self):
        if not _HAS_EXPORT_QUEUE:
            messagebox.showinfo("Export Queue", "export_queue module not found.")
            return
        if not self._export_queue_window or not self._export_queue_window.winfo_exists():
            self._export_queue_window = ExportQueueWindow(self)
        else:
            self._export_queue_window.lift()

    # ------------------------------------------------------------------
    # AI menu actions
    # ------------------------------------------------------------------
    def _ai_subtitle(self):
        cid = self.timeline.active_clip_id
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            messagebox.showinfo("Auto-Subtitle", "Select a clip on the timeline first.")
            return
        if _HAS_SMART:
            SubtitleDialog(self, c)
        else:
            messagebox.showinfo("Auto-Subtitle", "smart_features module not found.")

    def _ai_scene_detect(self):
        cid = self.timeline.active_clip_id
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            messagebox.showinfo("Scene Detect", "Select a clip on the timeline first.")
            return
        if _HAS_SMART:
            SceneDetectDialog(self, c)
        else:
            messagebox.showinfo("Scene Detect", "smart_features module not found.")

    def _ai_smart_reframe(self):
        cid = self.timeline.active_clip_id
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            messagebox.showinfo("Smart Reframe", "Select a clip on the timeline first.")
            return
        if _HAS_SMART:
            SmartReframeDialog(self, c)
        else:
            messagebox.showinfo("Smart Reframe", "smart_features module not found.")

    def _ai_beat_markers(self):
        cid = self.timeline.active_clip_id
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            messagebox.showinfo("Beat Markers", "Select a clip on the timeline first.")
            return
        if not _HAS_AI:
            messagebox.showinfo("Beat Markers", "ai_features module not found.")
            return
        try:
            detector = BeatDetector(c['path'])
            self._beat_markers = detector.detect()
            if hasattr(self.timeline, 'beat_markers'):
                self.timeline.beat_markers = list(self._beat_markers)
            self.timeline.draw()
            self.lbl_status.configure(text=f"Beat markers: {len(self._beat_markers)} found.")
        except Exception as e:
            messagebox.showerror("Beat Detection Failed", str(e))

    def _ai_noise_reduce(self):
        cid = self.timeline.active_clip_id
        c = next((x for x in self.project_clips if x['id'] == cid), None)
        if not c:
            messagebox.showinfo("Noise Reduction", "Select a clip on the timeline first.")
            return
        if not _HAS_AI:
            messagebox.showinfo("Noise Reduction", "ai_features module not found.")
            return
        try:
            reducer = NoiseReducer(c['path'])
            out_path = reducer.process()
            c['path'] = out_path
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.lbl_status.configure(text=f"Noise reduction applied: {os.path.basename(out_path)}")
        except Exception as e:
            messagebox.showerror("Noise Reduction Failed", str(e))

    # ------------------------------------------------------------------
    # Titles menu actions
    # ------------------------------------------------------------------
    def _new_title_clip(self):
        if _HAS_TITLE:
            TitleDialog(self)
        else:
            messagebox.showinfo("Title Clip", "title_generator module not found.")

    # ------------------------------------------------------------------
    # Render / export
    # ------------------------------------------------------------------
    def render_project(self):
        if not self.project_clips:
            messagebox.showinfo("Render", "No clips on the timeline.")
            return
        fmt = self.exp_fmt.get()
        out = filedialog.asksaveasfilename(
            title="Save Render As",
            defaultextension=f".{fmt}",
            initialfile=f"sequence.{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}"), ("All Files", "*.*")]
        )
        if not out:
            return
        pw, ph = map(int, self.exp_res.get().split('x'))
        t_vols = getattr(self.timeline, 'track_volumes', None) if hasattr(self, 'timeline') else None
        t_pans = getattr(self.timeline, 'track_pans', None) if hasattr(self, 'timeline') else None
        t_spds = getattr(self.timeline, 'track_speeds', None) if hasattr(self, 'timeline') else None
        t_effs = getattr(self, 'track_audio_effects', None)
        bitrate_kbps = getattr(self, 'var_exp_bitrate', None)
        bitrate_str = f"{int(bitrate_kbps.get())}k" if bitrate_kbps else None
        self._start_task(
            RenderTask(
                self.project_clips, out, pw, ph,
                int(float(self.exp_fps.get())),
                self.exp_codec.get(), bitrate_str,
                self.on_success, self.on_error,
                lambda p: self.after(0, lambda: self.prog_bar.set(p / 100)),
                track_volumes=t_vols,
                track_pans=t_pans,
                track_audio_effects=t_effs,
                track_speeds=t_spds,
            ),
            "Rendering…"
        )

    def open_audio_rack(self, target_track=None, target_clip=None):
        """Open the Studio Audio FX Rack & VST3 Chain dialog."""
        try:
            from audio_rack_ui import AudioRackDialog
            AudioRackDialog(self, target_track=target_track, target_clip=target_clip)
        except Exception as e:
            log.exception("Failed to open Audio Rack")
            messagebox.showerror("Audio Rack Error", str(e), parent=self)

    def export_clip(self, is_gif):
        c = next((x for x in self.project_clips
                  if x['id'] == self.timeline.active_clip_id), None)
        if not c:
            return
        if is_gif:
            out = filedialog.asksaveasfilename(
                defaultextension=".gif", initialfile="clip.gif",
                filetypes=[("GIF", "*.gif")])
        else:
            out = filedialog.askdirectory(title="Choose PNG output folder")
        if not out:
            return
        self._start_task(
            SingleClipExportTask(c, out, is_gif,
                                  self.on_success, self.on_error,
                                  lambda p: self.after(0, lambda: self.prog_bar.set(p / 100))),
            "Exporting clip…"
        )

    def _start_task(self, task, status_text):
        self.is_playing = False
        self.btn_play.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.lbl_status.configure(text=status_text)
        self.prog_bar.set(0)
        self.current_task = task
        self.current_task.start()

    def cancel_task(self):
        if self.current_task and self.current_task.is_alive():
            self.lbl_status.configure(text="Cancelling…")
            self.current_task.cancel()

    def on_success(self):
        def _cb():
            self.prog_bar.set(1)
            self.lbl_status.configure(text="Done.")
            self.btn_play.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            messagebox.showinfo("Success", "Export complete!")
        self.after(0, _cb)
        self.current_task = None

    def on_error(self, err):
        def _cb():
            self.prog_bar.set(0)
            self.lbl_status.configure(text="Failed.")
            self.btn_play.configure(state="normal")
            self.btn_cancel.configure(state="disabled")
            if "cancel" not in err.lower():
                messagebox.showerror("Render Error", err)
        self.after(0, _cb)
        self.current_task = None


    # ------------------------------------------------------------------
    # Preferences dialog
    # ------------------------------------------------------------------
    def _open_preferences(self):
        if _HAS_PREFS:
            PreferencesDialog(self)
        else:
            messagebox.showinfo("Preferences", "preferences module not found.")

    # ------------------------------------------------------------------
    # Add timeline marker at current playhead position
    # ------------------------------------------------------------------
    def _add_marker_at_playhead(self):
        t = self.playhead_t
        if hasattr(self.timeline, 'add_marker'):
            self.timeline.add_marker(t, label='', color='#f39c12')
            self.lbl_status.configure(text=f"Marker added at {t:.3f}s")
        else:
            messagebox.showinfo("Markers", "Timeline does not support markers.")

    # ------------------------------------------------------------------
    # AI colour match: match active clip colours to the next clip
    # ------------------------------------------------------------------
    def _ai_color_match(self):
        if not _HAS_COLOR_MATCH:
            messagebox.showinfo("Colour Match", "match_clip_colors not available.")
            return
        cid = self.timeline.active_clip_id
        if not cid:
            messagebox.showinfo("Colour Match", "Select a source clip on the timeline first.")
            return
        src_clip = next((x for x in self.project_clips if x['id'] == cid), None)
        if src_clip is None:
            return
        src_track = src_clip.get('track', 'V1')
        clips_on_track = [c for c in self.project_clips if c.get('track') == src_track]
        clips_on_track.sort(key=lambda c: c.get('start', 0))
        idx = next((i for i, c in enumerate(clips_on_track) if c['id'] == cid), None)
        if idx is None or idx + 1 >= len(clips_on_track):
            messagebox.showinfo("Colour Match", "No next clip on the same track to match to.")
            return
        src = clips_on_track[idx]
        ref = clips_on_track[idx + 1]
        try:
            # Extract a representative frame from each clip dict
            def _get_mid_frame(clip_dict):
                import cv2 as _cv2
                cap = _cv2.VideoCapture(clip_dict['path'])
                mid_t = clip_dict['trim_in'] + (clip_dict['trim_out'] - clip_dict['trim_in']) * 0.5
                cap.set(_cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError(f"Cannot read frame from {clip_dict['path']}")
                return _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)

            src_frame = _get_mid_frame(src)
            ref_frame = _get_mid_frame(ref)
            matched   = match_clip_colors(src_frame, ref_frame)
            # Store the matched frame as a colour-grade override on the source clip
            src['_color_match_frame'] = matched
            self.lbl_status.configure(text="Colour match applied — re-render to see effect.")
        except Exception as e:
            messagebox.showerror("Colour Match Failed", str(e))

    # ------------------------------------------------------------------
    # Refresh timeline after autosave recovery load
    # ------------------------------------------------------------------
    def _refresh_timeline(self):
        self.timeline.active_clip_id = None
        self.timeline.selected_clip_ids = set()
        if self.cmd_stack is not None:
            self.cmd_stack.clear()
        self.rebuild_proxy_cache()
        self.timeline.draw()
        self.update_inspector()

    # ------------------------------------------------------------------
    # Drag-and-drop handlers
    # ------------------------------------------------------------------
    def _parse_dnd_paths(self, data: str) -> list:
        """
        Parse a tkinterdnd2 drop event data string into a list of file paths.
        Handles both space-separated plain paths and brace-quoted paths
        (e.g. ``{path with spaces} /plain/path``).
        """
        paths = []
        data = data.strip()
        while data:
            if data.startswith('{'):
                end = data.find('}')
                if end == -1:
                    break
                paths.append(data[1:end])
                data = data[end + 1:].strip()
            else:
                parts = data.split(None, 1)
                paths.append(parts[0])
                data = parts[1].strip() if len(parts) > 1 else ''
        return paths

    def _on_dnd_drop_media(self, event):
        """Files dropped onto the media bin."""
        _VIDEO_EXT  = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.flv'}
        _AUDIO_EXT  = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'}
        _IMAGE_EXT  = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'}
        _VALID_EXT  = _VIDEO_EXT | _AUDIO_EXT | _IMAGE_EXT
        added = 0
        for p in self._parse_dnd_paths(event.data):
            if not os.path.isfile(p):
                continue
            if os.path.splitext(p)[1].lower() not in _VALID_EXT:
                continue
            if p not in self.media_bin:
                self.media_bin.append(p)
                added += 1
        if added:
            self.selected_media_index = len(self.media_bin) - 1
            self._refresh_media_bin_ui()
            self._on_media_select()
            self.lbl_status.configure(text=f"Imported {added} file(s) via drag-and-drop.")

    def _on_dnd_drop_timeline(self, event):
        """
        Files dropped directly onto the timeline canvas.
        Each file is added to the media bin (if not already present) and
        immediately inserted onto the track nearest to the drop position.
        """
        _VIDEO_EXT = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v', '.flv'}
        _AUDIO_EXT = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'}
        _IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.tif'}

        # Determine which track row was dropped onto
        drop_y = event.y_root - self.timeline.canvas.winfo_rooty()
        track_idx = max(0, min(
            len(self.timeline.tracks) - 1,
            int(drop_y // self.timeline.track_h)
        ))
        trk_key = self.timeline.track_keys[track_idx]

        # Determine drop time from x position
        drop_x = event.x_root - self.timeline.canvas.winfo_rootx()
        drop_t = self.timeline._x_to_time(drop_x)

        added = 0
        for p in self._parse_dnd_paths(event.data):
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            is_video = ext in _VIDEO_EXT
            is_audio = ext in _AUDIO_EXT
            is_image = ext in _IMAGE_EXT
            if not is_video and not is_audio and not is_image:
                continue

            # Coerce track: audio files cannot go on video/image tracks; images should go on image/video tracks
            effective_trk = trk_key
            if is_audio and ('V' in trk_key or 'I' in trk_key or 'S' in trk_key or 'T' in trk_key):
                effective_trk = 'A1'
            elif (is_video or is_image) and 'A' in trk_key:
                effective_trk = 'I1' if (is_image and any(k.startswith('I') for k in self.timeline.track_keys)) else 'V1'

            # Add to media bin if new
            if p not in self.media_bin:
                self.media_bin.append(p)
                self.selected_media_index = len(self.media_bin) - 1
                self._refresh_media_bin_ui()

            try:
                if is_image:
                    dur = 5.0
                elif is_video:
                    clip_obj = VideoFileClip(p)
                    dur = clip_obj.duration
                    clip_obj.close()
                else:
                    clip_obj = AudioFileClip(p)
                    dur = clip_obj.duration
                    clip_obj.close()
            except Exception as exc:
                log.warning("DnD drop failed for '%s': %s", p, exc)
                continue

            new_clip = {
                'id':           str(uuid.uuid4()),
                'path':         p,
                'track':        effective_trk,
                'start':        max(0.0, drop_t),
                'trim_in':      0.0,
                'trim_out':     dur,
                'scale':        1.0,
                'x':            0,
                'y':            0,
                'volume':       1.0,
                'speed':        1.0,
                'mute_audio':   False,
                'rm_bg':        False,
                'custom_masks': {},
                'aspect':       'Original',
                'rot':          '0°',
                'wm_text':      '',
                'wm_pos':       'Bottom Right',
                'transition_type': 'None',
                'transition_dur':  0.5,
                'notes':        '',
            }

            if _HAS_COMMANDS and self.cmd_stack is not None:
                self.cmd_stack.push(AddClipCommand(self, new_clip))
            else:
                self.project_clips.append(new_clip)
                self.rebuild_proxy_cache()
                self.timeline.draw()

            # Offset next drop so overlapping files don't stack on same frame
            drop_t += dur
            added += 1

        if added:
            self.lbl_status.configure(
                text=f"Dropped {added} file(s) onto timeline track {trk_key}.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if platform.system() == 'Windows':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GashWare.VideoStudio.NLE.1.0")
        except Exception:
            pass
    toggle_console(False)
    app = VideoStudioApp()
    try:
        app.mainloop()
    finally:
        for c in app.open_proxy_clips:
            try:
                c.close()
            except Exception:
                pass
