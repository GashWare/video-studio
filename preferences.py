# preferences.py
"""
Application Preferences dialog.
Opens from View → Preferences or Ctrl+,
Saves all settings to config_mgr on Apply.
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from config import config_mgr


_SHORTCUTS = [
    ("Space",   "Play / Pause"),
    ("I",       "Trim In"),
    ("O",       "Trim Out"),
    ("S",       "Split clip at playhead"),
    ("D",       "Duplicate clip"),
    ("Delete",  "Delete selected clip"),
    ("J",       "Shuttle Reverse"),
    ("K",       "Stop Shuttle"),
    ("L",       "Shuttle Forward"),
    ("=",       "Zoom In timeline"),
    ("-",       "Zoom Out timeline"),
    ("M",       "Add Marker"),
    ("Ctrl+Z",  "Undo"),
    ("Ctrl+Y",  "Redo"),
    ("Ctrl+S",  "Save Project"),
    ("Ctrl+E",  "Export"),
    ("Ctrl+,",  "Open Preferences"),
]


class PreferencesDialog(ctk.CTkToplevel):
    """
    Multi-tab preferences window with 4 tabs:
    - Appearance: theme (Dark/Light/System), accent colour
    - Performance: proxy resolution (360p/480p/720p), thread count (1-16),
                   GPU toggle (bool), autosave interval (10-300s)
    - Paths: default export folder, proxy cache folder, font directory
    - Shortcuts: read-only list of current keyboard shortcuts

    All settings read from config_mgr on open, written back on Apply.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Preferences")
        self.geometry("500x420")
        self.resizable(False, False)
        self.grab_set()

        self._build_ui()
        self._load_values()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.tabview = ctk.CTkTabview(self, width=480, height=360)
        self.tabview.pack(padx=10, pady=(10, 0), fill="both", expand=True)

        for name in ("Appearance", "Performance", "Paths", "Shortcuts"):
            self.tabview.add(name)

        self._build_appearance(self.tabview.tab("Appearance"))
        self._build_performance(self.tabview.tab("Performance"))
        self._build_paths(self.tabview.tab("Paths"))
        self._build_shortcuts(self.tabview.tab("Shortcuts"))

        # Bottom button row
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=(6, 10))
        ctk.CTkButton(btn_frame, text="Cancel", width=100, command=self.destroy).pack(side="right", padx=(6, 0))
        ctk.CTkButton(btn_frame, text="Apply",  width=100, command=self._apply).pack(side="right")

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row(parent, label_text, row):
        """Return a (label, right_frame) pair placed on *row* of a grid parent."""
        ctk.CTkLabel(parent, text=label_text, anchor="w").grid(
            row=row, column=0, padx=(12, 6), pady=8, sticky="w"
        )

    @staticmethod
    def _browse(var, title, mode="dir"):
        if mode == "dir":
            path = filedialog.askdirectory(title=title)
        else:
            path = filedialog.askopenfilename(title=title)
        if path:
            var.set(path)

    # ── Appearance tab ───────────────────────────────────────────────────────

    def _build_appearance(self, tab):
        tab.columnconfigure(1, weight=1)

        self._theme_var = ctk.StringVar()
        self._accent_var = ctk.StringVar()

        self._row(tab, "Theme:", 0)
        ctk.CTkOptionMenu(tab, variable=self._theme_var,
                          values=["Dark", "Light", "System"]).grid(
            row=0, column=1, padx=(0, 12), pady=8, sticky="ew")

        self._row(tab, "Accent Colour:", 1)
        ctk.CTkOptionMenu(tab, variable=self._accent_var,
                          values=["blue", "green", "dark-blue"]).grid(
            row=1, column=1, padx=(0, 12), pady=8, sticky="ew")

    # ── Performance tab ──────────────────────────────────────────────────────

    def _build_performance(self, tab):
        tab.columnconfigure(1, weight=1)

        self._proxy_var   = ctk.StringVar()
        self._threads_var = ctk.IntVar()
        self._gpu_var     = ctk.BooleanVar()
        self._autosave_var = ctk.IntVar()

        self._row(tab, "Proxy Resolution:", 0)
        ctk.CTkOptionMenu(tab, variable=self._proxy_var,
                          values=["360", "480", "720"]).grid(
            row=0, column=1, padx=(0, 12), pady=8, sticky="ew")

        self._row(tab, "Render Threads (1-16):", 1)
        ctk.CTkSlider(tab, variable=self._threads_var, from_=1, to=16,
                      number_of_steps=15).grid(
            row=1, column=1, padx=(0, 12), pady=8, sticky="ew")

        self._row(tab, "Use GPU:", 2)
        ctk.CTkSwitch(tab, text="", variable=self._gpu_var).grid(
            row=2, column=1, padx=(0, 12), pady=8, sticky="w")

        self._row(tab, "Autosave Interval (s):", 3)
        ctk.CTkSlider(tab, variable=self._autosave_var, from_=10, to=300,
                      number_of_steps=29).grid(
            row=3, column=1, padx=(0, 12), pady=8, sticky="ew")

    # ── Paths tab ────────────────────────────────────────────────────────────

    def _build_paths(self, tab):
        tab.columnconfigure(1, weight=1)

        self._export_var = ctk.StringVar()
        self._proxy_dir_var = ctk.StringVar()
        self._font_var = ctk.StringVar()

        path_rows = [
            ("Export Folder:",      self._export_var,    "Select Export Folder"),
            ("Proxy Cache Folder:", self._proxy_dir_var, "Select Proxy Cache Folder"),
            ("Font Directory:",     self._font_var,      "Select Font Directory"),
        ]

        for i, (label, var, title) in enumerate(path_rows):
            self._row(tab, label, i)
            frame = ctk.CTkFrame(tab, fg_color="transparent")
            frame.grid(row=i, column=1, padx=(0, 12), pady=8, sticky="ew")
            frame.columnconfigure(0, weight=1)
            ctk.CTkEntry(frame, textvariable=var).grid(row=0, column=0, sticky="ew", padx=(0, 4))
            ctk.CTkButton(frame, text="Browse…", width=70,
                          command=lambda v=var, t=title: self._browse(v, t)).grid(row=0, column=1)

    # ── Shortcuts tab ────────────────────────────────────────────────────────

    def _build_shortcuts(self, tab):
        frame = ctk.CTkScrollableFrame(tab)
        frame.pack(fill="both", expand=True, padx=6, pady=6)
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Shortcut", font=ctk.CTkFont(weight="bold"),
                     anchor="w").grid(row=0, column=0, padx=(4, 12), pady=2, sticky="w")
        ctk.CTkLabel(frame, text="Action", font=ctk.CTkFont(weight="bold"),
                     anchor="w").grid(row=0, column=1, padx=4, pady=2, sticky="w")

        for i, (key, action) in enumerate(_SHORTCUTS, start=1):
            ctk.CTkLabel(frame, text=key, anchor="w").grid(
                row=i, column=0, padx=(4, 12), pady=1, sticky="w")
            ctk.CTkLabel(frame, text=action, anchor="w").grid(
                row=i, column=1, padx=4, pady=1, sticky="w")

    # ── Load / Apply ─────────────────────────────────────────────────────────

    def _load_values(self):
        self._theme_var.set(config_mgr.get("theme") or "Dark")
        self._accent_var.set(config_mgr.get("accent_color") or "blue")
        self._proxy_var.set(str(config_mgr.get("proxy_height") or 480))
        self._threads_var.set(int(config_mgr.get("render_threads") or 4))
        self._gpu_var.set(bool(config_mgr.get("use_gpu") or False))
        self._autosave_var.set(int(config_mgr.get("autosave_interval") or 60))
        self._export_var.set(config_mgr.get("export_folder") or "")
        self._proxy_dir_var.set(config_mgr.get("proxy_folder") or "")
        self._font_var.set(config_mgr.get("font_folder") or "")

    def _apply(self):
        config_mgr.save_config("theme",             self._theme_var.get())
        config_mgr.save_config("accent_color",      self._accent_var.get())
        config_mgr.save_config("proxy_height",      int(self._proxy_var.get()))
        config_mgr.save_config("render_threads",    int(self._threads_var.get()))
        config_mgr.save_config("use_gpu",           bool(self._gpu_var.get()))
        config_mgr.save_config("autosave_interval", int(self._autosave_var.get()))
        config_mgr.save_config("export_folder",     self._export_var.get())
        config_mgr.save_config("proxy_folder",      self._proxy_dir_var.get())
        config_mgr.save_config("font_folder",       self._font_var.get())
        self.destroy()
