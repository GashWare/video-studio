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
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Sequence Settings")
        self.geometry("340x220")
        self.resizable(False, False)
        self.grab_set()

        def row(lbl, widget_cls, **kw):
            f = ctk.CTkFrame(self, fg_color="transparent")
            f.pack(fill="x", padx=20, pady=4)
            ctk.CTkLabel(f, text=lbl, width=90, anchor="w").pack(side="left")
            w = widget_cls(f, **kw)
            w.pack(side="right", fill="x", expand=True)
            return w

        self.var_w   = ctk.StringVar(value=str(app.proj_w))
        self.var_h   = ctk.StringVar(value=str(app.proj_h))
        self.var_fps = ctk.StringVar(value=str(app.fps))

        self.ent_w   = row("Width px:", ctk.CTkEntry, textvariable=self.var_w)
        self.ent_h   = row("Height px:", ctk.CTkEntry, textvariable=self.var_h)
        self.cmb_fps = row("FPS:", ctk.CTkComboBox,
                           variable=self.var_fps, values=["23.976", "24", "25", "29.97", "30", "50", "59.94", "60"])

        ctk.CTkButton(self, text="Apply", command=self._apply).pack(pady=16, padx=20, fill="x")

    def _apply(self):
        try:
            w, h = int(self.var_w.get()), int(self.var_h.get())
            fps  = float(self.var_fps.get())
            self.app.proj_w, self.app.proj_h = w, h
            self.app.fps = fps
            self.app.lbl_seq.configure(text=f"{w}×{h}  {fps}fps")
            self.destroy()
        except ValueError:
            messagebox.showerror("Invalid", "Enter valid numbers.", parent=self)


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


class VideoStudioApp(_AppBase):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode(config_mgr.get("theme") or "Dark")
        ctk.set_default_color_theme(config_mgr.get("accent_color") or "blue")
        self.title("PyDev Ultimate NLE")

        geom = config_mgr.get("geometry")
        self.geometry(geom if geom else "1500x940")
        self.minsize(1200, 750)

        # Project state
        self.project_clips     = []
        self.media_bin         = []          # list of file paths
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
        pf = ctk.CTkFrame(self)
        pf.grid(row=0, column=0, padx=(10, 5), pady=(10, 5), sticky="nsew")
        pf.grid_rowconfigure(0, weight=1)
        pf.grid_columnconfigure(0, weight=1)
        self.preview_frame = pf

        self.lbl_preview = ctk.CTkLabel(pf, text="No Sequence", fg_color="black",
                                         font=("Segoe UI", 13), text_color="#555555")
        self.lbl_preview.grid(row=0, column=0, padx=4, pady=(4, 0), sticky="nsew")

        # Scrub slider
        self.scrub_var = tk.DoubleVar(value=0.0)
        self.scrub_slider = ctk.CTkSlider(pf, from_=0, to=1, variable=self.scrub_var,
                                           command=self._on_scrub_slider,
                                           height=14, button_color="gray50")
        self.scrub_slider.grid(row=1, column=0, padx=6, pady=(2, 0), sticky="ew")

        # In/Out marker labels
        self.lbl_inout = ctk.CTkLabel(pf, text="In: —   Out: —",
                                       text_color="#666666", font=("Segoe UI", 8))
        self.lbl_inout.grid(row=2, column=0, padx=6, sticky="w")

        # Transport controls
        ctrl = ctk.CTkFrame(pf, fg_color="transparent")
        ctrl.grid(row=3, column=0, pady=(3, 6))

        self.btn_rw  = ctk.CTkButton(ctrl, text="⏮", width=36, fg_color="transparent",
                                      hover_color="#333333", command=self._kb_goto_start)
        self.btn_rw.pack(side="left", padx=2)
        _Tooltip(self.btn_rw, "Go to Start (Home)")

        self.btn_step_b = ctk.CTkButton(ctrl, text="◀|", width=36, fg_color="transparent",
                                         hover_color="#333333", command=self._step_back)
        self.btn_step_b.pack(side="left", padx=2)
        _Tooltip(self.btn_step_b, "Step Back 1 Frame (←)")

        self.btn_play = ctk.CTkButton(ctrl, text="▶ Play", width=110,
                                       fg_color="teal", hover_color="#006666",
                                       command=self.toggle_play)
        self.btn_play.pack(side="left", padx=6)
        _Tooltip(self.btn_play, "Play / Pause (Space)")

        self.btn_step_f = ctk.CTkButton(ctrl, text="|▶", width=36, fg_color="transparent",
                                         hover_color="#333333", command=self._step_forward)
        self.btn_step_f.pack(side="left", padx=2)
        _Tooltip(self.btn_step_f, "Step Forward 1 Frame (→)")

        self.btn_ff  = ctk.CTkButton(ctrl, text="⏭", width=36, fg_color="transparent",
                                      hover_color="#333333", command=self._kb_goto_end)
        self.btn_ff.pack(side="left", padx=2)
        _Tooltip(self.btn_ff, "Go to End (End)")

        self.lbl_time = ctk.CTkLabel(ctrl, text="00:00:00.00", width=110,
                                      font=("Consolas", 12))
        self.lbl_time.pack(side="left", padx=10)

        # Sequence info label
        self.lbl_seq = ctk.CTkLabel(ctrl,
                                     text=f"{self.proj_w}×{self.proj_h}  {self.fps}fps",
                                     text_color="#888888", font=("Segoe UI", 9))
        self.lbl_seq.pack(side="left", padx=6)
        self.lbl_seq.bind("<Button-1>", lambda e: self._open_seq_settings())
        _Tooltip(self.lbl_seq, "Click to change sequence settings")

    def _build_panels(self):
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, padx=(5, 10), pady=(10, 5), sticky="nsew")
        self.tab_media = self.tabs.add("Media")
        self.tab_edit  = self.tabs.add("Inspector")
        self.tab_export= self.tabs.add("Export")
        self._build_media_panel()
        self._build_inspector_panel()
        self._build_export_panel()

    def _build_media_panel(self):
        btn_row = ctk.CTkFrame(self.tab_media, fg_color="transparent")
        btn_row.pack(fill="x", padx=6, pady=(8, 4))
        ctk.CTkButton(btn_row, text="➕ Import", command=self.import_media,
                       width=90).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="✕ Remove", command=self._remove_media_item,
                       fg_color="#555555", hover_color="#444444", width=90).pack(side="right", padx=2)

        self.media_listbox = tk.Listbox(
            self.tab_media, bg="#1e1e1e", fg="#dddddd",
            selectbackground="#2d6fbf", activestyle="none",
            font=("Segoe UI", 9), relief="flat", bd=0,
            highlightthickness=1, highlightcolor="#333333",
            highlightbackground="#333333"
        )
        self.media_listbox.pack(fill="both", expand=True, padx=6, pady=2)
        self.media_listbox.bind("<Double-1>",   self._media_double_click)
        self.media_listbox.bind("<Return>",      lambda e: self.insert_to_timeline())

        # Drag-and-drop files onto the media bin
        if _HAS_DND:
            self.media_listbox.drop_target_register(DND_FILES)
            self.media_listbox.dnd_bind('<<Drop>>', self._on_dnd_drop_media)

        ins_row = ctk.CTkFrame(self.tab_media, fg_color="transparent")
        ins_row.pack(fill="x", padx=6, pady=(4, 8))
        self.ins_trk_var = ctk.StringVar(value="V1")
        self.ins_trk_combo = ctk.CTkComboBox(ins_row, variable=self.ins_trk_var,
                                             values=["V2", "V1", "A1", "A2"], width=70)
        self.ins_trk_combo.pack(side="left", padx=4)
        ctk.CTkButton(ins_row, text="⬇ Insert to Timeline",
                       command=self.insert_to_timeline).pack(side="right", padx=2)

    def refresh_track_selectors(self):
        """Update track combo boxes to reflect current timeline tracks."""
        if hasattr(self, 'ins_trk_combo') and hasattr(self, 'timeline'):
            self.ins_trk_combo.configure(values=list(self.timeline.track_keys))
            if self.ins_trk_var.get() not in self.timeline.track_keys and self.timeline.track_keys:
                self.ins_trk_var.set(self.timeline.track_keys[0])

    def _build_inspector_panel(self):
        self.insp_lbl = ctk.CTkLabel(self.tab_edit, text="Select a clip on the timeline.",
                                      text_color="gray", font=("Segoe UI", 10))
        self.insp_lbl.pack(pady=30)

        self.insp_c = ctk.CTkScrollableFrame(self.tab_edit, fg_color="transparent")

        def mk_row(lbl, widget_cls, **kwargs):
            f = ctk.CTkFrame(self.insp_c, fg_color="transparent")
            f.pack(fill="x", pady=2)
            if lbl:
                ctk.CTkLabel(f, text=lbl, width=68, anchor="w",
                              font=("Segoe UI", 9)).pack(side="left")
            w = widget_cls(f, **kwargs)
            w.pack(side="right", fill="x", expand=True)
            return w

        def section(txt):
            ctk.CTkLabel(self.insp_c, text=txt,
                          font=("Segoe UI", 10, "bold"),
                          text_color="#aaaaaa").pack(anchor="w", pady=(10, 2))
            ctk.CTkFrame(self.insp_c, height=1, fg_color="#333333").pack(fill="x", pady=(0, 4))

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
        ctk.CTkEntry(tr, textvariable=self.var_tin, width=58).pack(side="left", padx=2)
        ctk.CTkButton(tr, text="[  In", width=46, height=28,
                       command=lambda: self.set_trim('in')).pack(side="left", padx=1)
        ctk.CTkButton(tr, text="Out  ]", width=46, height=28,
                       command=lambda: self.set_trim('out')).pack(side="right", padx=1)
        ctk.CTkEntry(tr, textvariable=self.var_tout, width=58).pack(side="right", padx=2)

        # ── Transition ──
        section("Transition (into next clip)")
        mk_row("Type:", ctk.CTkComboBox, variable=self.var_transition,
               values=TRANSITION_TYPES)
        mk_row("Duration s:", ctk.CTkEntry, textvariable=self.var_trans_dur)

        # ── Transform ──
        section("Transform")
        mk_row("Aspect:", ctk.CTkComboBox, variable=self.var_aspect,
               values=["Original", "16:9", "9:16", "1:1", "4:3"])
        mk_row("Rotate:", ctk.CTkComboBox, variable=self.var_rot,
               values=["0°", "90°", "180°", "270°"])

        sc_row = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        sc_row.pack(fill="x", pady=2)
        ctk.CTkLabel(sc_row, text="Scale:", width=68, anchor="w",
                      font=("Segoe UI", 9)).pack(side="left")
        self.lbl_sc = ctk.CTkLabel(sc_row, text="1.0×", width=38,
                                    font=("Consolas", 9))
        self.lbl_sc.pack(side="right")
        ctk.CTkSlider(sc_row, variable=self.var_scale, from_=0.1, to=3.0,
                       command=lambda v: self.lbl_sc.configure(text=f"{v:.2f}×")
                       ).pack(side="right", fill="x", expand=True, padx=4)

        pos_row = mk_row("Pos X / Y:", ctk.CTkEntry, textvariable=self.var_x, width=58)
        ctk.CTkEntry(pos_row.master, textvariable=self.var_y, width=58).pack(side="right", padx=2)

        # ── Audio ──
        section("Audio")
        vol_row = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        vol_row.pack(fill="x", pady=2)
        ctk.CTkLabel(vol_row, text="Volume:", width=68, anchor="w",
                      font=("Segoe UI", 9)).pack(side="left")
        self.lbl_vol = ctk.CTkLabel(vol_row, text="100%", width=38,
                                     font=("Consolas", 9))
        self.lbl_vol.pack(side="right")
        ctk.CTkSlider(vol_row, variable=self.var_vol, from_=0.0, to=2.0,
                       command=lambda v: self.lbl_vol.configure(text=f"{int(v*100)}%")
                       ).pack(side="right", fill="x", expand=True, padx=4)
        mk_row("", ctk.CTkCheckBox, text="Mute", variable=self.var_mute)

        # ── Effects ──
        section("Effects")
        sp_row = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        sp_row.pack(fill="x", pady=2)
        ctk.CTkLabel(sp_row, text="Speed:", width=68, anchor="w",
                      font=("Segoe UI", 9)).pack(side="left")
        self.lbl_sp = ctk.CTkLabel(sp_row, text="1.00×", width=38,
                                    font=("Consolas", 9))
        self.lbl_sp.pack(side="right")
        ctk.CTkSlider(sp_row, variable=self.var_speed, from_=0.25, to=4.0,
                       command=lambda v: self.lbl_sp.configure(text=f"{v:.2f}×")
                       ).pack(side="right", fill="x", expand=True, padx=4)

        mk_row("Watermark:", ctk.CTkEntry, textvariable=self.var_wm)
        mk_row("WM Pos:", ctk.CTkComboBox, variable=self.var_wmpos,
               values=["Bottom Right", "Bottom Left", "Top Right", "Top Left", "Center"])

        # ── AI / Background ──
        section("AI / Background")
        mk_row("", ctk.CTkCheckBox, text="Remove Background (AI)", variable=self.var_rmbg)
        ai_f = ctk.CTkFrame(self.insp_c, fg_color="transparent")
        ai_f.pack(fill="x", pady=2)
        ctk.CTkButton(ai_f, text="👁 Preview Mask", width=110, height=28,
                       command=self.preview_mask).pack(side="left", padx=2)
        ctk.CTkButton(ai_f, text="🖌 Touchup", width=90, height=28,
                       command=self.open_touchup).pack(side="right", padx=2)

        # ── Colour Grading ──
        section("Colour Grading")
        if _HAS_COLOR_GRADING:
            self.color_grading_panel = ColorGradingPanel(self.insp_c)
            self.color_grading_panel.pack(fill="x", pady=2)
        else:
            self.color_grading_panel = None
            ctk.CTkLabel(self.insp_c, text="(color_grading module not found)",
                          text_color="#555555", font=("Segoe UI", 8)).pack(anchor="w")

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
                          text_color="#555555", font=("Segoe UI", 8)).pack(anchor="w")

        ctk.CTkButton(self.insp_c, text="✔ Apply Changes", height=36,
                       command=self.apply_inspector_changes).pack(pady=12, fill="x")

        # Keyframes button
        ctk.CTkButton(self.insp_c, text="⏱ Edit Keyframes", height=28,
                       fg_color="#333333",
                       command=self._open_keyframe_editor).pack(pady=4, fill="x")

        # ── Clip Tools ──
        section("Clip Tools")
        ctk.CTkButton(self.insp_c, text="Export Transparent GIF", fg_color="teal",
                       hover_color="#006666", height=30,
                       command=lambda: self.export_clip(True)).pack(pady=2, fill="x")
        ctk.CTkButton(self.insp_c, text="Extract PNG Frames", fg_color="teal",
                       hover_color="#006666", height=30,
                       command=lambda: self.export_clip(False)).pack(pady=2, fill="x")
        ctk.CTkButton(self.insp_c, text="🗑 Delete Clip", height=30,
                       fg_color="#6b1111", hover_color="#4a0a0a",
                       command=self._kb_delete).pack(pady=(10, 4), fill="x")

        # ── Notes ──
        section("Notes")
        ctk.CTkEntry(self.insp_c, textvariable=self.var_notes,
                     placeholder_text="Clip notes…").pack(fill="x", pady=2)

    def _build_export_panel(self):
        def row(lbl, widget_cls, **kw):
            f = ctk.CTkFrame(self.tab_export, fg_color="transparent")
            f.pack(fill="x", pady=5, padx=8)
            ctk.CTkLabel(f, text=lbl, width=72, anchor="w").pack(side="left")
            w = widget_cls(f, **kw)
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

        ctk.CTkButton(self.tab_export, text="⬛  Render Project", height=46,
                       fg_color="#1a6e1a", hover_color="#145414",
                       font=("Segoe UI", 12, "bold"),
                       command=self.render_project).pack(pady=20, padx=8, fill="x")

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
        sb = ctk.CTkFrame(self, height=32, corner_radius=0)
        sb.grid(row=2, column=0, columnspan=2, padx=0, pady=0, sticky="ew")
        sb.grid_columnconfigure(1, weight=1)

        self.lbl_status = ctk.CTkLabel(sb, text="Ready.", anchor="w",
                                        font=("Segoe UI", 9))
        self.lbl_status.grid(row=0, column=1, padx=12, sticky="w")

        self.prog_bar = ctk.CTkProgressBar(sb, mode="determinate", width=240, height=8)
        self.prog_bar.grid(row=0, column=2, padx=8, pady=8)
        self.prog_bar.set(0)

        self.btn_cancel = ctk.CTkButton(sb, text="✕ Cancel", width=80, height=24,
                                         fg_color="#7a2020", hover_color="#5c1515",
                                         font=("Segoe UI", 9),
                                         command=self.cancel_task, state="disabled")
        self.btn_cancel.grid(row=0, column=3, padx=(0, 8))

        ctk.CTkLabel(sb, text="Space=Play  I/O=Trim  Del=Delete  S=Split  D=Dup  Z=Undo  Ctrl+Scroll=Zoom",
                      text_color="#555555", font=("Segoe UI", 8)
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
        for p in paths:
            if p not in self.media_bin:
                self.media_bin.append(p)
                self.media_listbox.insert("end", os.path.basename(p))
        log.info("Imported %d file(s)", len(paths))

    def _remove_media_item(self):
        sel = self.media_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.media_bin.pop(idx)
        self.media_listbox.delete(idx)

    def _media_double_click(self, event):
        self.insert_to_timeline()

    def insert_to_timeline(self):
        sel = self.media_listbox.curselection()
        if not sel:
            return
        path = self.media_bin[sel[0]]
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
            return
        try:
            tin  = float(self.var_tin.get())
            tout = float(self.var_tout.get())
            if tout <= tin:
                messagebox.showerror("Invalid Trim", "Out point must be after In point.")
                return

            # Capture old props for undo
            old_props = copy.deepcopy(c)

            c['trim_in']  = tin
            c['trim_out'] = tout
            c['scale']    = self.var_scale.get()
            c['speed']    = self.var_speed.get()
            c['volume']   = self.var_vol.get()
            c['mute_audio'] = self.var_mute.get()
            x = self.var_x.get().strip()
            y = self.var_y.get().strip()
            c['x'] = int(x) if x.lstrip('-').isdigit() else x
            c['y'] = int(y) if y.lstrip('-').isdigit() else y
            c['aspect']   = self.var_aspect.get()
            c['rot']      = self.var_rot.get()
            c['wm_text']  = self.var_wm.get()
            c['wm_pos']   = self.var_wmpos.get()
            c['rm_bg']    = self.var_rmbg.get()
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
        except ValueError as e:
            messagebox.showerror("Invalid Value", str(e))

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
            )
        else:
            # Synchronous fallback (blocks UI — kept for compatibility)
            for c in self.open_proxy_clips:
                try: c.close()
                except Exception: pass
            self.open_proxy_clips = []
            max_dur = max(c['start'] + (c['trim_out'] - c['trim_in'])
                          for c in self.project_clips)
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
                    self.open_proxy_clips.append(clip)
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
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")
        self.scrub_to(0.0)

    def _play_loop(self):
        """Fallback play loop used when FrameQueue is not available."""
        if not self.is_playing or not self.master_proxy_clip:
            return
        nxt = self.playhead_t + 1.0 / self.fps
        if nxt >= self.master_proxy_clip.duration:
            self.is_playing = False
            self.btn_play.configure(text="▶ Play")
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
            self.media_listbox.delete(0, 'end')
            self.timeline.active_clip_id = None
            self.timeline.selected_clip_ids = set()
            if self.cmd_stack is not None:
                self.cmd_stack.clear()
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.update_inspector()
            self._current_project_path = None
            self.title("PyDev Ultimate NLE")

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
            self.lbl_status.configure(text=f"Saved: {os.path.basename(path)}")
            self.title(f"PyDev Ultimate NLE — {os.path.basename(path)}")
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
            self.media_listbox.delete(0, 'end')
            for p in self.media_bin:
                self.media_listbox.insert('end', os.path.basename(p))
            self.timeline.active_clip_id = None
            self.timeline.selected_clip_ids = set()
            if self.cmd_stack is not None:
                self.cmd_stack.clear()
            self._current_project_path = path
            add_recent_project(config_mgr, path)
            self.lbl_seq.configure(text=f"{w}×{h}  {fps}fps")
            self.title(f"PyDev Ultimate NLE — {os.path.basename(path)}")
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
        t_effs = getattr(self, 'track_audio_effects', None)
        self._start_task(
            RenderTask(
                self.project_clips, out, pw, ph,
                int(float(self.exp_fps.get())),
                self.exp_codec.get(), None,
                self.on_success, self.on_error,
                lambda p: self.after(0, lambda: self.prog_bar.set(p / 100)),
                track_volumes=t_vols,
                track_pans=t_pans,
                track_audio_effects=t_effs,
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
        """Files dropped onto the media bin listbox."""
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
                self.media_listbox.insert('end', os.path.basename(p))
                added += 1
        if added:
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
                self.media_listbox.insert('end', os.path.basename(p))

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

            # Offset next drop so overlapping files don't stack on same frame
            drop_t += dur
            added += 1

        if added:
            self.lbl_status.configure(
                text=f"Dropped {added} file(s) onto timeline track {trk_key}.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
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
