# video_studio.py

# STREAMING_CHUNK:Initializing Bootloader and Imports
import sys
import subprocess
import os
import importlib.util
import gc
import uuid

# --- BOOTLOADER ---
def _bootloader():
    """Checks for and installs required dependencies before launching."""
    reqs = {
        'customtkinter': 'customtkinter', 
        'moviepy': 'moviepy', 
        'cv2': 'opencv-python', 
        'PIL': 'Pillow', 
        'proglog': 'proglog',
        'rembg': 'rembg',
        'imageio': 'imageio',
        'numpy': 'numpy'
    }
    missing = [pip_name for mod_name, pip_name in reqs.items() if importlib.util.find_spec(mod_name) is None]
    if missing:
        print(f"Bootloader: Installing {', '.join(missing)}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
            os.execv(sys.executable, ['python'] + sys.argv)
        except Exception as e:
            print(f"Bootloader Failed: {e}"); sys.exit(1)

_bootloader()

# --- LIBRARIES ---
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import json
import logging
import threading
import ctypes
import cv2
import numpy as np
from PIL import Image, ImageTk
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, CompositeAudioClip, ColorClip  # requires moviepy<2
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx
from proglog import ProgressBarLogger

# STREAMING_CHUNK:Configuring State Management and Logging
CONFIG_FILE = "config.json"

class ConfigManager:
    def __init__(self):
        self.defaults = {"geometry": None, "theme": "Dark", "accent_color": "blue", "log_level": "INFO"}
        self.config = self.load_config()
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f: return {**self.defaults, **json.load(f)}
            except Exception: return self.defaults.copy()
        return self.defaults.copy()
    def save_config(self, key, value):
        self.config[key] = value
        try:
            with open(CONFIG_FILE, 'w') as f: json.dump(self.config, f, indent=4)
        except Exception as e: logging.error(f"Config save failed: {e}")
    def get(self, key): return self.config.get(key)

config_mgr = ConfigManager()

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config_mgr.get("log_level").upper(), logging.INFO))
    if logger.hasHandlers(): logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in [logging.FileHandler("app.log"), logging.StreamHandler()]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)

setup_logging()

def toggle_console(show: bool):
    if os.name == 'nt':
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd: ctypes.windll.user32.ShowWindow(hwnd, 5 if show else 0)
        except Exception as e: pass

toggle_console(True)

def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass

# STREAMING_CHUNK:AI Tools & MoviePy Helpers
class BGREngine:
    def __init__(self):
        from rembg import remove
        self.remove = remove
        self.p_frame = self.p_mask = None
    def get(self, frame):
        if self.p_frame is not None:
            mse = (cv2.absdiff(frame, self.p_frame)**2).mean()
            if mse < 5.0: return self.p_mask
        mask = self.remove(frame)[:, :, 3]
        self.p_frame, self.p_mask = frame, mask
        return mask

class CustomMpyLogger(ProgressBarLogger):
    def __init__(self, progress_callback, cancel_flag):
        super().__init__()
        self.ui_progress_callback = progress_callback
        self.cancel_flag = cancel_flag
    def bars_callback(self, bar, attr, value, old_value=None):
        if self.cancel_flag.is_set(): raise InterruptedError("Cancelled by user.")
        total = self.bars[bar].get('total', 0)
        if total > 0: self.ui_progress_callback((value / total) * 100)

class VideoTask(threading.Thread):
    def __init__(self, callback, error_callback, progress_callback=None):
        super().__init__(daemon=True)
        self.callback, self.error_callback, self.progress_callback = callback, error_callback, progress_callback
        self._cancel_flag = threading.Event()
    def cancel(self): self._cancel_flag.set()

# STREAMING_CHUNK:Multi-Track Rendering Engine
class RenderTask(VideoTask):
    """Handles the final high-resolution render of the multi-track composition."""
    def __init__(self, project_clips, out_path, proj_w, proj_h, fps, codec, bitrate, cb, err_cb, prog_cb):
        super().__init__(cb, err_cb, prog_cb)
        self.project_clips = project_clips
        self.out_path, self.proj_w, self.proj_h = out_path, proj_w, proj_h
        self.fps, self.codec, self.bitrate = fps, codec, bitrate

    def run(self):
        open_clips = []
        try:
            logging.info(f"Starting Multi-Track Render: {self.out_path}")
            
            # Layer Order: Base (Color), V1 (Main), V2 (Overlay)
            v_layers = []
            a_layers = []
            
            # Base Canvas
            max_dur = max([c['start'] + (c['trim_out'] - c['trim_in']) for c in self.project_clips] + [1.0])
            bg_clip = ColorClip(size=(self.proj_w, self.proj_h), color=(0,0,0), duration=max_dur)
            v_layers.append(bg_clip)
            
            # Sort clips: V1 first (bottom layer), then V2 (top layer)
            vid_clips = [c for c in self.project_clips if c['track'] in ['V1', 'V2']]
            vid_clips.sort(key=lambda x: 0 if x['track'] == 'V1' else 1)
            
            for c_data in vid_clips:
                if self._cancel_flag.is_set(): raise InterruptedError()
                
                c = VideoFileClip(c_data['path']).subclip(c_data['trim_in'], c_data['trim_out'])
                open_clips.append(c)
                
                # Apply Audio Mute/Volume if part of video clip
                if c_data.get('mute_audio', False) and c.audio: c = c.without_audio()
                elif c.audio and c_data.get('volume', 1.0) != 1.0: c = c.fx(afx.volumex, c_data['volume'])
                
                # Transformations
                if c_data.get('scale', 1.0) != 1.0: c = c.resize(c_data['scale'])
                if c_data.get('x', 'center') != 'center' or c_data.get('y', 'center') != 'center':
                    c = c.set_pos((c_data.get('x', 'center'), c_data.get('y', 'center')))
                else: c = c.set_pos("center")
                
                # AI Background Removal
                if c_data.get('rm_bg', False):
                    engine = BGREngine()
                    def apply_bg_removal(gf, t):
                        frame = gf(t)
                        mask = engine.get(frame)
                        return cv2.bitwise_and(frame, frame, mask=mask)
                    
                    def mask_func(gf, t):
                        frame = gf(t)
                        return engine.get(frame) / 255.0

                    c = c.fl(apply_bg_removal)
                    from moviepy.video.VideoClip import VideoClip
                    m_clip = VideoClip(lambda t: mask_func(c.get_frame, t), duration=c.duration, ismask=True)
                    c = c.set_mask(m_clip)
                
                c = c.set_start(c_data['start'])
                v_layers.append(c)
                
            # Compile Audio Tracks (A1, A2)
            aud_clips = [c for c in self.project_clips if c['track'] in ['A1', 'A2']]
            for c_data in aud_clips:
                if self._cancel_flag.is_set(): raise InterruptedError()
                c = AudioFileClip(c_data['path']).subclip(c_data['trim_in'], c_data['trim_out'])
                open_clips.append(c)
                if c_data.get('volume', 1.0) != 1.0: c = c.fx(afx.volumex, c_data['volume'])
                c = c.set_start(c_data['start'])
                a_layers.append(c)
                
            # Build Master Composition
            final_vid = CompositeVideoClip(v_layers, size=(self.proj_w, self.proj_h)).set_duration(max_dur)
            
            # Combine extracted video audio with dedicated audio tracks
            all_audio = a_layers
            for v in v_layers:
                if getattr(v, 'audio', None) is not None: all_audio.append(v.audio.set_start(v.start))
            if all_audio:
                final_aud = CompositeAudioClip(all_audio).set_duration(max_dur)
                final_vid = final_vid.set_audio(final_aud)

            if self._cancel_flag.is_set(): raise InterruptedError()

            logger = CustomMpyLogger(self.progress_callback, self._cancel_flag)
            opt_threads = max(2, (os.cpu_count() or 4) - 2)
            
            write_kwargs = {
                "logger": logger, "codec": self.codec,
                "audio_codec": 'aac' if self.codec != "libvpx-vp9" else 'libvorbis',
                "threads": opt_threads, "preset": "fast", "fps": self.fps
            }
            if self.bitrate: write_kwargs["bitrate"] = self.bitrate

            final_vid.write_videofile(self.out_path, **write_kwargs)
            
            if not self._cancel_flag.is_set(): self.callback()

        except InterruptedError: logging.info("Render cancelled.")
        except Exception as e:
            logging.error(f"RenderTask failed: {e}")
            if not self._cancel_flag.is_set(): self.error_callback(str(e))
        finally:
            for c in open_clips: c.close()
            free_memory()

# STREAMING_CHUNK:Multi-Track UI Canvas
class MultiTrackTimeline(tk.Canvas):
    """NLE Interactive Timeline for placing and splicing clips."""
    def __init__(self, master, app, **kwargs):
        super().__init__(master, bg="#1a1a1a", highlightthickness=0, cursor="arrow", **kwargs)
        self.app = app
        self.tracks = ['V2 (Overlay)', 'V1 (Main)', 'A1 (Audio)', 'A2 (Audio)']
        self.track_keys = ['V2', 'V1', 'A1', 'A2']
        self.track_h = 50
        self.px_per_sec = 20
        self.playhead_time = 0.0
        
        self.active_clip_id = None
        self.drag_data = {"id": None, "start_x": 0, "orig_start": 0}
        
        self.bind("<Configure>", self.draw)
        self.bind("<Button-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)

    def set_playhead(self, t):
        self.playhead_time = max(0, t)
        self.draw()

    def draw(self, event=None):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        
        # 1. Draw Track Backgrounds and Headers
        for i, trk_name in enumerate(self.tracks):
            y = i * self.track_h
            # Track bg
            self.create_rectangle(0, y, w, y + self.track_h, fill="#252525" if i%2==0 else "#2a2a2a", outline="")
            # Header separator
            self.create_line(100, y, 100, y + self.track_h, fill="#555555")
            # Track Name
            self.create_text(10, y + 25, text=trk_name, anchor="w", fill="white", font=("Arial", 9, "bold"))
            
        # 2. Draw Clips
        for clip in self.app.project_clips:
            trk_idx = self.track_keys.index(clip['track'])
            duration = clip['trim_out'] - clip['trim_in']
            x1 = 100 + clip['start'] * self.px_per_sec
            x2 = x1 + duration * self.px_per_sec
            y1 = trk_idx * self.track_h + 5
            y2 = y1 + self.track_h - 10
            
            is_sel = (clip['id'] == self.active_clip_id)
            color = "#3a7ebf" if 'V' in clip['track'] else "#27ae60"
            out_color = "white" if is_sel else ""
            
            # Clip Block
            self.create_rectangle(x1, y1, x2, y2, fill=color, outline=out_color, width=2 if is_sel else 0)
            
            # Clip Label
            name = os.path.basename(clip['path'])
            self.create_text(x1 + 5, y1 + 15, text=name[:15]+"..." if len(name)>15 else name, anchor="w", fill="white", font=("Arial", 8))

        # 3. Draw Timeline Ruler Header
        header_y = len(self.tracks) * self.track_h
        self.create_rectangle(0, header_y, w, h, fill="#111111", outline="")
        for s in range(0, int(w / self.px_per_sec) + 10, 5):
            x = 100 + s * self.px_per_sec
            self.create_line(x, header_y, x, header_y + 10, fill="gray")
            self.create_text(x, header_y + 18, text=f"{s}s", fill="gray", font=("Arial", 8))

        # 4. Draw Playhead
        px = 100 + self.playhead_time * self.px_per_sec
        self.create_line(px, 0, px, h, fill="red", width=2)
        self.create_polygon(px-6, header_y, px+6, header_y, px, header_y+10, fill="red")

    def on_press(self, event):
        w = self.winfo_width()
        # Check if clicked on Playhead ruler
        header_y = len(self.tracks) * self.track_h
        if event.y > header_y or event.x < 100:
            t = max(0, (event.x - 100) / self.px_per_sec)
            self.app.scrub_to(t)
            self.active_clip_id = None
            self.app.update_inspector()
            self.draw()
            return
            
        # Check if clicked a clip
        clicked_id = None
        for clip in self.app.project_clips:
            trk_idx = self.track_keys.index(clip['track'])
            dur = clip['trim_out'] - clip['trim_in']
            x1 = 100 + clip['start'] * self.px_per_sec
            x2 = x1 + dur * self.px_per_sec
            y1 = trk_idx * self.track_h + 5
            y2 = y1 + self.track_h - 10
            
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                clicked_id = clip['id']
                self.drag_data["orig_start"] = clip['start']
                break
                
        self.active_clip_id = clicked_id
        self.drag_data["id"] = clicked_id
        self.drag_data["start_x"] = event.x
        
        self.app.update_inspector()
        self.draw()

    def on_drag(self, event):
        if not self.drag_data["id"]:
            # Scrub playhead if dragging on empty space/ruler
            t = max(0, (event.x - 100) / self.px_per_sec)
            self.app.scrub_to(t)
            return
            
        # Move Clip
        dx = event.x - self.drag_data["start_x"]
        dt = dx / self.px_per_sec
        
        for clip in self.app.project_clips:
            if clip['id'] == self.drag_data["id"]:
                clip['start'] = max(0.0, self.drag_data["orig_start"] + dt)
                break
                
        self.app.rebuild_proxy_cache() # Triggers visual update
        self.draw()

# STREAMING_CHUNK:Building the Master Application (NLE UI)
class VideoStudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode(config_mgr.get("theme"))
        ctk.set_default_color_theme(config_mgr.get("accent_color"))
        
        self.title("Video Studio")
        self.geometry("1400x900")
        self.minsize(1200, 800)

        # Set Window and Taskbar Icon
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _icon_ico = os.path.join(_script_dir, "assets", "icon.ico")
        _icon_png = os.path.join(_script_dir, "assets", "icon.png")
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
        
        # Project Data
        self.project_clips = [] # List of clip dictionaries
        self.media_bin = [] # List of loaded file paths
        
        self.master_proxy_clip = None # The low-res compiled moviepy clip for fast live preview
        self.open_proxy_clips = [] # Keep track to close them
        
        self.proj_w, self.proj_h = 1920, 1080 # Default HD Project
        self.proxy_w, self.proxy_h = 640, 360 # Fast live-render resolution
        self.fps = 30
        
        self.is_playing = False
        self.playhead_t = 0.0
        self.current_task = None
        
        self.setup_ui()
        self.update_inspector() # Initialize empty inspector

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=2) # Preview Pane
        self.grid_columnconfigure(1, weight=1) # Inspector Pane
        self.grid_rowconfigure(0, weight=3) # Top section
        self.grid_rowconfigure(1, weight=2) # Timeline section

        # --- TOP LEFT: PREVIEW VIEWER ---
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=0, padx=(10,5), pady=(10,5), sticky="nsew")
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        
        self.lbl_preview = ctk.CTkLabel(self.preview_frame, text="NLE Preview Screen", fg_color="black")
        self.lbl_preview.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Playback Controls
        ctrl_frame = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        ctrl_frame.grid(row=1, column=0, pady=5)
        
        self.btn_play = ctk.CTkButton(ctrl_frame, text="▶ Play Sequence", width=120, command=self.toggle_play, fg_color="teal")
        self.btn_play.pack(side="left", padx=10)
        
        self.lbl_time = ctk.CTkLabel(ctrl_frame, text="00:00.00", width=80)
        self.lbl_time.pack(side="left")

        # --- TOP RIGHT: TABBED WORKSPACE ---
        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=0, column=1, padx=(5,10), pady=(10,5), sticky="nsew")
        
        self.tab_media = self.tabs.add("Media Bin")
        self.tab_edit = self.tabs.add("Inspector")
        self.tab_export = self.tabs.add("Export")
        
        self.setup_media_tab()
        self.setup_inspector_tab()
        self.setup_export_tab()

        # --- BOTTOM: TIMELINE ---
        self.timeline_frame = ctk.CTkFrame(self)
        self.timeline_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        self.timeline_frame.grid_rowconfigure(1, weight=1)
        self.timeline_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(self.timeline_frame, text="Multi-Track Timeline", font=("Inter", 14, "bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(5,0))
        
        self.timeline = MultiTrackTimeline(self.timeline_frame, self)
        self.timeline.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Status Footer
        self.status_frame = ctk.CTkFrame(self, height=30)
        self.status_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=(0,10), sticky="ew")
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        self.lbl_status = ctk.CTkLabel(self.status_frame, text="Project Ready.")
        self.lbl_status.grid(row=0, column=1, padx=10, sticky="w")
        self.prog_bar = ctk.CTkProgressBar(self.status_frame, mode="determinate", width=300)
        self.prog_bar.grid(row=0, column=2, padx=10, pady=5)
        self.prog_bar.set(0)

    # STREAMING_CHUNK:Tab Constructions
    def setup_media_tab(self):
        btn_add = ctk.CTkButton(self.tab_media, text="➕ Import File", command=self.import_media)
        btn_add.pack(fill="x", padx=10, pady=10)
        
        self.media_listbox = tk.Listbox(self.tab_media, bg="#2b2b2b", fg="white", selectbackground="#1f538d", height=8)
        self.media_listbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Track selection dropdown for insertion
        ins_frame = ctk.CTkFrame(self.tab_media, fg_color="transparent")
        ins_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(ins_frame, text="Target:").pack(side="left")
        self.ins_trk_var = ctk.StringVar(value="V1")
        ctk.CTkComboBox(ins_frame, variable=self.ins_trk_var, values=["V2", "V1", "A1", "A2"], width=80).pack(side="left", padx=5)
        
        ctk.CTkButton(ins_frame, text="⬇ Insert to Timeline", command=self.insert_to_timeline).pack(side="right")

    def setup_inspector_tab(self):
        # We define the UI variables here, and update them when a clip is clicked
        self.insp_lbl = ctk.CTkLabel(self.tab_edit, text="Select a clip on the timeline.", font=("Inter", 12, "italic"), text_color="gray")
        self.insp_lbl.pack(pady=20)
        
        self.insp_container = ctk.CTkScrollableFrame(self.tab_edit, fg_color="transparent")
        # Pack this later when a clip is selected
        
        def mk_row(lbl):
            f = ctk.CTkFrame(self.insp_container, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=lbl, width=60, anchor="w").pack(side="left")
            return f

        self.var_trim_in = ctk.StringVar()
        self.var_trim_out = ctk.StringVar()
        r1 = mk_row("Trim In/Out:")
        ctk.CTkEntry(r1, textvariable=self.var_trim_in, width=60).pack(side="left", padx=2)
        ctk.CTkEntry(r1, textvariable=self.var_trim_out, width=60).pack(side="left", padx=2)
        
        self.var_scale = ctk.DoubleVar()
        r2 = mk_row("Scale:")
        self.lbl_scale = ctk.CTkLabel(r2, text="1.0x", width=40)
        self.lbl_scale.pack(side="right")
        ctk.CTkSlider(r2, variable=self.var_scale, from_=0.1, to=3.0, command=lambda v: self.lbl_scale.configure(text=f"{v:.1f}x")).pack(side="left", fill="x", expand=True)
        
        self.var_x = ctk.StringVar()
        self.var_y = ctk.StringVar()
        r3 = mk_row("Pos X/Y:")
        ctk.CTkEntry(r3, textvariable=self.var_x, width=60, placeholder_text="center/px").pack(side="left", padx=2)
        ctk.CTkEntry(r3, textvariable=self.var_y, width=60, placeholder_text="center/px").pack(side="left", padx=2)
        
        self.var_vol = ctk.DoubleVar()
        r4 = mk_row("Volume:")
        self.lbl_vol = ctk.CTkLabel(r4, text="1.0x", width=40)
        self.lbl_vol.pack(side="right")
        ctk.CTkSlider(r4, variable=self.var_vol, from_=0.0, to=3.0, command=lambda v: self.lbl_vol.configure(text=f"{v:.1f}x")).pack(side="left", fill="x", expand=True)

        self.var_rmbg = ctk.BooleanVar()
        ctk.CTkCheckBox(self.insp_container, text="AI Background Removal (V1/V2 only)", variable=self.var_rmbg).pack(pady=10, anchor="w")
        
        ctk.CTkButton(self.insp_container, text="Apply Changes", command=self.apply_inspector_changes).pack(pady=15, fill="x")
        ctk.CTkButton(self.insp_container, text="🗑 Delete Clip", fg_color="#8B0000", hover_color="#5C0000", command=self.delete_active_clip).pack(pady=5, fill="x")

    def setup_export_tab(self):
        def mk_row(lbl, widget_cls, **kwargs): 
            f = ctk.CTkFrame(self.tab_export, fg_color="transparent")
            f.pack(fill="x", pady=5)
            ctk.CTkLabel(f, text=lbl, width=60, anchor="w").pack(side="left")
            w = widget_cls(f, **kwargs)
            w.pack(side="right", fill="x", expand=True)
            return w

        self.exp_fmt = ctk.StringVar(value="mp4")
        mk_row("Format:", ctk.CTkComboBox, variable=self.exp_fmt, values=["mp4", "avi", "mkv", "webm"])
        
        self.exp_codec = ctk.StringVar(value="libx264")
        mk_row("Codec:", ctk.CTkComboBox, variable=self.exp_codec, values=["libx264", "libvpx-vp9", "h264_nvenc"])
        
        self.exp_fps = ctk.StringVar(value="30")
        mk_row("FPS:", ctk.CTkComboBox, variable=self.exp_fps, values=["24", "30", "60"])
        
        self.exp_res = ctk.StringVar(value="1920x1080")
        mk_row("Res:", ctk.CTkComboBox, variable=self.exp_res, values=["1920x1080", "1280x720", "854x480"])
        
        ctk.CTkButton(self.tab_export, text="Render Project", height=50, command=self.render_project).pack(pady=30, fill="x")

    # STREAMING_CHUNK:Application Logic
    def import_media(self):
        paths = filedialog.askopenfilenames(filetypes=[("Media Files", "*.mp4 *.avi *.mkv *.mov *.mp3 *.wav")])
        for p in paths:
            if p not in self.media_bin:
                self.media_bin.append(p)
                self.media_listbox.insert("end", os.path.basename(p))

    def insert_to_timeline(self):
        sel = self.media_listbox.curselection()
        if not sel: return
        path = self.media_bin[sel[0]]
        trk = self.ins_trk_var.get()
        
        try:
            # Determine duration
            if 'V' in trk: c = VideoFileClip(path)
            else: c = AudioFileClip(path)
            dur = c.duration
            c.close()
            
            # Find free space at end of track
            trk_clips = [c for c in self.project_clips if c['track'] == trk]
            start_t = max([c['start'] + (c['trim_out'] - c['trim_in']) for c in trk_clips]) if trk_clips else 0.0
            
            clip_data = {
                'id': str(uuid.uuid4()), 'path': path, 'track': trk,
                'start': start_t, 'trim_in': 0.0, 'trim_out': dur,
                'scale': 1.0, 'x': 'center', 'y': 'center', 'volume': 1.0, 'rm_bg': False
            }
            self.project_clips.append(clip_data)
            
            self.rebuild_proxy_cache()
            self.timeline.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse media:\n{e}")

    def update_inspector(self):
        clip_id = self.timeline.active_clip_id
        if not clip_id:
            self.insp_container.pack_forget()
            self.insp_lbl.pack(pady=20)
            return
            
        clip = next((c for c in self.project_clips if c['id'] == clip_id), None)
        if not clip: return
        
        self.insp_lbl.pack_forget()
        self.insp_container.pack(fill="both", expand=True)
        
        self.var_trim_in.set(str(round(clip['trim_in'], 2)))
        self.var_trim_out.set(str(round(clip['trim_out'], 2)))
        self.var_scale.set(clip.get('scale', 1.0))
        self.lbl_scale.configure(text=f"{clip.get('scale', 1.0):.1f}x")
        self.var_x.set(str(clip.get('x', 'center')))
        self.var_y.set(str(clip.get('y', 'center')))
        self.var_vol.set(clip.get('volume', 1.0))
        self.lbl_vol.configure(text=f"{clip.get('volume', 1.0):.1f}x")
        self.var_rmbg.set(clip.get('rm_bg', False))

    def show_toast(self, message: str, success: bool = True, duration_ms: int = 2500):
        """Display a slick, modern non-blocking HUD toast notification in the UI."""
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
        except Exception:
            pass

    def apply_inspector_changes(self):
        clip_id = self.timeline.active_clip_id
        clip = next((c for c in self.project_clips if c['id'] == clip_id), None)
        if not clip:
            self.show_toast("Changes failed: No clip selected.", success=False)
            messagebox.showwarning("No Clip Selected", "Please select a clip on the timeline first.")
            return
        
        try:
            tin_str = self.var_trim_in.get().strip()
            tout_str = self.var_trim_out.get().strip()
            tin = float(tin_str) if tin_str else 0.0
            tout = float(tout_str) if tout_str else float(clip.get('trim_out', 0.0))
            if tout <= tin:
                self.show_toast("Changes failed: Out point must be after In point.", success=False)
                messagebox.showerror("Invalid Trim", "Out point must be after In point.")
                return

            clip['trim_in'] = tin
            clip['trim_out'] = tout
            clip['scale'] = float(self.var_scale.get())
            
            x_val = self.var_x.get().strip()
            y_val = self.var_y.get().strip()
            clip['x'] = int(x_val) if (x_val.startswith('-') and x_val[1:].isdigit()) or x_val.isdigit() else x_val
            clip['y'] = int(y_val) if (y_val.startswith('-') and y_val[1:].isdigit()) or y_val.isdigit() else y_val
            
            clip['volume'] = float(self.var_vol.get())
            clip['rm_bg'] = bool(self.var_rmbg.get())
            
            self.rebuild_proxy_cache()
            self.timeline.draw()
            self.show_toast("Changes applied successfully!", success=True)
            if hasattr(self, 'lbl_status'):
                self.lbl_status.configure(text="Changes applied to clip.")
        except ValueError as e:
            self.show_toast(f"Changes failed to apply: {e}", success=False)
            messagebox.showerror("Error", f"Invalid number format: {e}")
        except Exception as e:
            self.show_toast(f"Changes failed to apply: {e}", success=False)
            messagebox.showerror("Error", f"Failed to apply changes: {e}")

    def delete_active_clip(self):
        clip_id = self.timeline.active_clip_id
        self.project_clips = [c for c in self.project_clips if c['id'] != clip_id]
        self.timeline.active_clip_id = None
        self.update_inspector()
        self.rebuild_proxy_cache()
        self.timeline.draw()

    # STREAMING_CHUNK:Live Proxy Engine
    def rebuild_proxy_cache(self):
        """Builds a low-res composite moviepy clip for real-time live preview scrubbing."""
        if self.is_playing: self.toggle_play()
        
        for c in self.open_proxy_clips: c.close()
        self.open_proxy_clips = []
        
        if not self.project_clips:
            self.master_proxy_clip = None
            self.lbl_preview.configure(image="", text="Empty Sequence")
            return
            
        max_dur = max([c['start'] + (c['trim_out'] - c['trim_in']) for c in self.project_clips])
        
        # Base proxy layer
        v_layers = [ColorClip(size=(self.proxy_w, self.proxy_h), color=(0,0,0), duration=max_dur)]
        
        vid_clips = [c for c in self.project_clips if c['track'] in ['V1', 'V2']]
        vid_clips.sort(key=lambda x: 0 if x['track'] == 'V1' else 1) # V1 bottom, V2 top
        
        for c_data in vid_clips:
            try:
                # Load, trim, scale down massively for fast UI preview
                c = VideoFileClip(c_data['path']).subclip(c_data['trim_in'], c_data['trim_out']).resize(height=self.proxy_h)
                self.open_proxy_clips.append(c)
                
                # Apply user scaling on top of proxy scale
                if c_data.get('scale', 1.0) != 1.0: c = c.resize(c_data['scale'])
                
                if c_data.get('x', 'center') != 'center' or c_data.get('y', 'center') != 'center':
                    # Rough proxy position approximation
                    px_x = c_data.get('x') if isinstance(c_data.get('x'), str) else int(c_data.get('x') * (self.proxy_w/self.proj_w))
                    px_y = c_data.get('y') if isinstance(c_data.get('y'), str) else int(c_data.get('y') * (self.proxy_h/self.proj_h))
                    c = c.set_pos((px_x, px_y))
                else: c = c.set_pos("center")
                
                c = c.set_start(c_data['start'])
                v_layers.append(c)
            except Exception as e:
                logging.error(f"Failed to load proxy clip: {e}")
                
        self.master_proxy_clip = CompositeVideoClip(v_layers, size=(self.proxy_w, self.proxy_h)).set_duration(max_dur)
        self.scrub_to(self.playhead_t) # Refresh frame

    def scrub_to(self, t):
        self.playhead_t = t
        self.timeline.set_playhead(t)
        
        m, s = divmod(t, 60); h, m = divmod(m, 60)
        self.lbl_time.configure(text=f"{int(h):02d}:{int(m):02d}:{s:05.2f}")
        
        if not self.master_proxy_clip: return
        
        try:
            # Moviepy expects t to be within duration
            t_safe = min(t, self.master_proxy_clip.duration - 0.01)
            frame = self.master_proxy_clip.get_frame(t_safe)
            
            pil_im = Image.fromarray(frame)
            # Scale up proxy image to fit UI label
            w, h_ui = self.preview_frame.winfo_width() - 10, self.preview_frame.winfo_height() - 50
            if w > 10 and h_ui > 10:
                pil_im.thumbnail((w, h_ui), Image.Resampling.BILINEAR)
                
            ctk_img = ctk.CTkImage(light_image=pil_im, dark_image=pil_im, size=pil_im.size)
            self.lbl_preview.configure(image=ctk_img, text="")
            self.lbl_preview._image = ctk_img # GC prevent
        except Exception as e:
            logging.error(f"Preview render error: {e}")

    def toggle_play(self):
        if not self.master_proxy_clip: return
        self.is_playing = not self.is_playing
        self.btn_play.configure(text="⏸ Pause" if self.is_playing else "▶ Play Sequence")
        if self.is_playing: self.play_loop()

    def play_loop(self):
        if not self.is_playing or not self.master_proxy_clip: return
        
        next_t = self.playhead_t + (1.0 / self.fps)
        if next_t > self.master_proxy_clip.duration:
            self.is_playing = False
            self.btn_play.configure(text="▶ Play Sequence")
            return
            
        self.scrub_to(next_t)
        # Note: Audio is NOT played live during Tkinter loop to prevent thread locking. Render to hear mix.
        self.after(int(1000 / self.fps), self.play_loop)

    # STREAMING_CHUNK:Final Export Routing
    def render_project(self):
        if not self.project_clips: return messagebox.showwarning("Warning", "Timeline is empty.")
        
        fmt = self.exp_fmt.get()
        out_path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", initialfile=f"sequence.{fmt}", filetypes=[(f"{fmt.upper()} File", f"*.{fmt}")])
        if not out_path: return

        # Parse Resolution
        res_str = self.exp_res.get()
        pw, ph = map(int, res_str.split('x'))
        fps = int(self.exp_fps.get())
        
        # Toggle UI
        self.btn_play.configure(state="disabled")
        self.is_playing = False
        self.btn_cancel.configure(state="normal")
        self.prog_bar.set(0)
        self.lbl_status.configure(text="Rendering Multi-Track Sequence...")

        self.current_task = RenderTask(
            self.project_clips, out_path, pw, ph, fps, self.exp_codec.get(), None,
            self.on_success, self.on_error, lambda p: self.after(0, lambda: self.prog_bar.set(p/100))
        )
        self.current_task.start()

    def cancel_task(self):
        if self.current_task and self.current_task.is_alive():
            self.lbl_status.configure(text="Cancelling Render...")
            self.current_task.cancel()

    def on_success(self):
        self.after(0, lambda: [self.prog_bar.set(1), self.lbl_status.configure(text="Render Complete."), 
                               self.btn_cancel.configure(state="disabled"), self.btn_play.configure(state="normal"), 
                               messagebox.showinfo("Success", "Render Complete!")])
        self.current_task = None

    def on_error(self, err):
        def _err():
            self.prog_bar.set(0); self.lbl_status.configure(text="Render Failed.")
            self.btn_cancel.configure(state="disabled"); self.btn_play.configure(state="normal")
            if "cancel" not in err.lower(): messagebox.showerror("Error", err)
        self.after(0, _err)
        self.current_task = None

if __name__ == "__main__":
    app = VideoStudioApp()
    try: app.mainloop()
    finally:
        for c in app.open_proxy_clips: c.close()