# audio_rack_ui.py
"""
DAW-grade Studio Audio FX Rack & VST3 Chain Dialog.
Allows adding, configuring, reordering, and previewing audio effects
on individual clips or full timeline tracks.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from audio_effects import (
    AudioEffect, AudioEffectChain,
    ParametricEQEffect, CompressorEffect, ReverbEffect,
    StereoDelayEffect, LimiterEffect, HighPassFilterEffect,
    LowPassFilterEffect, VST3PluginEffect
)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "Dialogue Clarity (Podcast)": [
        HighPassFilterEffect(cutoff_freq=80.0),
        ParametricEQEffect(low_shelf_freq=120.0, low_shelf_gain=-3.0,
                           mid1_freq=400.0, mid1_gain=-2.5, mid1_q=1.2,
                           mid2_freq=3500.0, mid2_gain=3.5, mid2_q=1.0,
                           high_shelf_freq=10000.0, high_shelf_gain=2.0),
        CompressorEffect(threshold_db=-20.0, ratio=3.5, attack_ms=10.0, release_ms=80.0, makeup_gain_db=2.5),
        LimiterEffect(ceiling_db=-0.5),
    ],
    "Vocal Warmth & Leveler": [
        HighPassFilterEffect(cutoff_freq=60.0),
        ParametricEQEffect(low_shelf_freq=150.0, low_shelf_gain=2.0,
                           mid1_freq=800.0, mid1_gain=1.0, mid1_q=0.8,
                           mid2_freq=4500.0, mid2_gain=2.0, mid2_q=1.0,
                           high_shelf_freq=12000.0, high_shelf_gain=1.5),
        CompressorEffect(threshold_db=-16.0, ratio=2.5, attack_ms=25.0, release_ms=150.0, makeup_gain_db=1.5),
    ],
    "Broadcast Loudness Maximizer": [
        CompressorEffect(threshold_db=-14.0, ratio=4.0, attack_ms=8.0, release_ms=60.0, makeup_gain_db=3.0),
        LimiterEffect(ceiling_db=-0.1, release_ms=30.0),
    ],
    "Studio Ambience Reverb": [
        ReverbEffect(room_size=0.5, damping=0.4, wet_level=0.25, dry_level=0.9),
    ],
    "Stereo Slapback Delay": [
        StereoDelayEffect(delay_ms=120.0, feedback=0.25, mix=0.3),
    ],
    "Radio / Telephone Voice": [
        HighPassFilterEffect(cutoff_freq=400.0),
        LowPassFilterEffect(cutoff_freq=3500.0),
        ParametricEQEffect(mid1_freq=1200.0, mid1_gain=6.0, mid1_q=2.0),
        CompressorEffect(threshold_db=-24.0, ratio=8.0, attack_ms=2.0, release_ms=40.0, makeup_gain_db=5.0),
    ],
}


# ---------------------------------------------------------------------------
# AudioRackDialog
# ---------------------------------------------------------------------------

class AudioRackDialog(ctk.CTkToplevel):
    """
    Studio Audio Effects Rack window.
    """

    def __init__(self, app, target_track: str = None, target_clip: dict = None):
        super().__init__(app)
        self.app = app
        self.target_track = target_track
        self.target_clip = target_clip
        self.selected_effect_idx = 0

        self.title("Studio Audio FX Rack & VST3 Chain")
        self.geometry("780x540")
        self.minsize(700, 480)

        self._init_data()
        self._build_ui()

    def _init_data(self):
        """Load or initialize effect chain for target clip/track."""
        if not hasattr(self.app, "track_audio_effects"):
            self.app.track_audio_effects = {}

        if self.target_clip is not None:
            if "audio_effects" not in self.target_clip:
                self.target_clip["audio_effects"] = []
            self.chain = AudioEffectChain.from_list(self.target_clip["audio_effects"])
        elif self.target_track is not None:
            if self.target_track not in self.app.track_audio_effects:
                self.app.track_audio_effects[self.target_track] = []
            self.chain = AudioEffectChain.from_list(self.app.track_audio_effects[self.target_track])
        else:
            # Default to first audio track or active clip
            cid = getattr(self.app.timeline, "active_clip_id", None) if hasattr(self.app, "timeline") else None
            c = next((x for x in getattr(self.app, "project_clips", []) if x["id"] == cid), None)
            if c is not None:
                self.target_clip = c
                if "audio_effects" not in self.target_clip:
                    self.target_clip["audio_effects"] = []
                self.chain = AudioEffectChain.from_list(self.target_clip["audio_effects"])
            else:
                self.target_track = "Master"
                if "Master" not in self.app.track_audio_effects:
                    self.app.track_audio_effects["Master"] = []
                self.chain = AudioEffectChain.from_list(self.app.track_audio_effects["Master"])

    def _sync_back(self):
        """Save chain back into target clip or track dictionary."""
        serialized = self.chain.to_list()
        if self.target_clip is not None:
            self.target_clip["audio_effects"] = serialized
        elif self.target_track is not None:
            self.app.track_audio_effects[self.target_track] = serialized

        try:
            self.app.rebuild_proxy_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Top Bar ──────────────────────────────────────────────────
        top_bar = ctk.CTkFrame(self, fg_color="#1f232b", height=52, corner_radius=0)
        top_bar.pack(fill="x", side="top", padx=0, pady=0)

        # Target Selector
        ctk.CTkLabel(top_bar, text="Target:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(12, 6), pady=8)

        track_opts = ["Master"]
        if hasattr(self.app, "timeline") and hasattr(self.app.timeline, "track_keys"):
            for tk_name in self.app.timeline.track_keys:
                track_opts.append(f"Track {tk_name}")
        track_opts.append("Active Clip")

        current_val = "Active Clip" if self.target_clip is not None else (f"Track {self.target_track}" if self.target_track != "Master" else "Master")
        self._target_var = ctk.StringVar(value=current_val)
        self._target_cmb = ctk.CTkComboBox(
            top_bar, variable=self._target_var, values=track_opts,
            command=self._on_target_change, width=170, height=32, font=("Segoe UI", 11, "bold")
        )
        self._target_cmb.pack(side="left", padx=4, pady=8)

        # Preset Selector
        ctk.CTkLabel(top_bar, text="Preset:", font=("Segoe UI", 12, "bold")).pack(side="left", padx=(16, 6), pady=8)
        self._preset_var = ctk.StringVar(value="-- Select Preset --")
        self._preset_cmb = ctk.CTkComboBox(
            top_bar, variable=self._preset_var,
            values=["-- Select Preset --"] + list(PRESETS.keys()),
            command=self._apply_preset, width=230, height=32, font=("Segoe UI", 11)
        )
        self._preset_cmb.pack(side="left", padx=4, pady=8)

        # Bypass All Checkbox
        self._bypass_all_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            top_bar, text="Bypass All", variable=self._bypass_all_var,
            command=self._toggle_bypass_all, width=100, font=("Segoe UI", 11, "bold")
        ).pack(side="right", padx=12, pady=8)

        # ── Main Content Split ───────────────────────────────────────
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=8, pady=8)

        # Left Column: Effects Chain List
        left_pane = ctk.CTkFrame(content, width=290, fg_color="#242831", corner_radius=6)
        left_pane.pack(side="left", fill="both", padx=(0, 6), pady=0)
        left_pane.pack_propagate(False)

        chain_header = ctk.CTkFrame(left_pane, fg_color="transparent")
        chain_header.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(chain_header, text="INSERT EFFECTS", font=("Segoe UI", 12, "bold"), text_color="#00d2be").pack(side="left")

        # Effect Actions Toolbar
        add_bar = ctk.CTkFrame(left_pane, fg_color="transparent")
        add_bar.pack(fill="x", padx=8, pady=4)

        self._add_var = ctk.StringVar(value="+ Add Effect…")
        add_opts = [
            "+ Add Effect…",
            "4-Band Parametric EQ",
            "Dynamic Compressor",
            "Studio Reverb",
            "Stereo Delay",
            "Peak Limiter",
            "High Pass Filter",
            "Low Pass Filter",
            "VST3 Plugin…",
        ]
        ctk.CTkComboBox(
            add_bar, variable=self._add_var, values=add_opts,
            command=self._on_add_effect_selected, width=170, height=32, font=("Segoe UI", 10, "bold")
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(add_bar, text="▲", width=32, height=32, font=("Segoe UI", 11, "bold"), command=lambda: self._move_effect(-1)).pack(side="left", padx=2)
        ctk.CTkButton(add_bar, text="▼", width=32, height=32, font=("Segoe UI", 11, "bold"), command=lambda: self._move_effect(1)).pack(side="left", padx=2)
        ctk.CTkButton(add_bar, text="✕", width=32, height=32, font=("Segoe UI", 11, "bold"), fg_color="#7a2020", command=self._delete_selected_effect).pack(side="left", padx=2)

        # Scrollable Chain List
        self._chain_scroll = ctk.CTkScrollableFrame(left_pane, label_text="")
        self._chain_scroll.pack(fill="both", expand=True, padx=6, pady=6)

        # Right Column: Parameter Editor Panel
        self._param_pane = ctk.CTkFrame(content, fg_color="#242831", corner_radius=6)
        self._param_pane.pack(side="right", fill="both", expand=True, padx=(6, 0), pady=0)

        self._refresh_chain_list()
        self._refresh_params()

    # ------------------------------------------------------------------
    # Target / Preset handling
    # ------------------------------------------------------------------

    def _on_target_change(self, val: str):
        if val == "Active Clip":
            cid = getattr(self.app.timeline, "active_clip_id", None) if hasattr(self.app, "timeline") else None
            c = next((x for x in getattr(self.app, "project_clips", []) if x["id"] == cid), None)
            if not c:
                messagebox.showinfo("Audio Rack", "No active clip selected on the timeline.", parent=self)
                return
            self.target_clip = c
            self.target_track = None
        elif val.startswith("Track "):
            self.target_track = val.replace("Track ", "")
            self.target_clip = None
        else:
            self.target_track = "Master"
            self.target_clip = None

        self._init_data()
        self.selected_effect_idx = 0
        self._refresh_chain_list()
        self._refresh_params()

    def _apply_preset(self, preset_name: str):
        if preset_name not in PRESETS:
            return
        preset_effects = PRESETS[preset_name]
        self.chain.effects = [copy.deepcopy(e) for e in preset_effects]
        self.selected_effect_idx = 0
        self._sync_back()
        self._refresh_chain_list()
        self._refresh_params()

    def _toggle_bypass_all(self):
        bypassed = self._bypass_all_var.get()
        for e in self.chain.effects:
            e.enabled = not bypassed
        self._sync_back()
        self._refresh_chain_list()

    # ------------------------------------------------------------------
    # Chain Manipulation
    # ------------------------------------------------------------------

    def _on_add_effect_selected(self, choice: str):
        self._add_var.set("+ Add Effect…")
        if choice == "4-Band Parametric EQ":
            eff = ParametricEQEffect()
        elif choice == "Dynamic Compressor":
            eff = CompressorEffect()
        elif choice == "Studio Reverb":
            eff = ReverbEffect()
        elif choice == "Stereo Delay":
            eff = StereoDelayEffect()
        elif choice == "Peak Limiter":
            eff = LimiterEffect()
        elif choice == "High Pass Filter":
            eff = HighPassFilterEffect()
        elif choice == "Low Pass Filter":
            eff = LowPassFilterEffect()
        elif choice == "VST3 Plugin…":
            path = filedialog.askopenfilename(
                title="Select VST3 Audio Plugin",
                filetypes=[("VST3 Plugins", "*.vst3"), ("All Files", "*.*")],
                parent=self
            )
            if not path:
                return
            eff = VST3PluginEffect(name=os.path.basename(path).replace(".vst3", ""), plugin_path=path)
        else:
            return

        self.chain.add_effect(eff)
        self.selected_effect_idx = len(self.chain.effects) - 1
        self._sync_back()
        self._refresh_chain_list()
        self._refresh_params()

    def _move_effect(self, direction: int):
        if not self.chain.effects:
            return
        self.chain.move_effect(self.selected_effect_idx, direction)
        self.selected_effect_idx = max(0, min(len(self.chain.effects) - 1, self.selected_effect_idx + direction))
        self._sync_back()
        self._refresh_chain_list()
        self._refresh_params()

    def _delete_selected_effect(self):
        if not self.chain.effects or self.selected_effect_idx >= len(self.chain.effects):
            return
        del self.chain.effects[self.selected_effect_idx]
        self.selected_effect_idx = max(0, min(len(self.chain.effects) - 1, self.selected_effect_idx))
        self._sync_back()
        self._refresh_chain_list()
        self._refresh_params()

    # ------------------------------------------------------------------
    # Chain & Params Rendering
    # ------------------------------------------------------------------

    def _refresh_chain_list(self):
        for w in self._chain_scroll.winfo_children():
            w.destroy()

        if not self.chain.effects:
            ctk.CTkLabel(self._chain_scroll, text="No insert effects.\nUse '+ Add Effect' above.",
                         text_color="gray").pack(pady=30)
            return

        for idx, eff in enumerate(self.chain.effects):
            is_sel = (idx == self.selected_effect_idx)
            bg_col = "#1f538d" if is_sel else "#2e3440"

            row = ctk.CTkFrame(self._chain_scroll, fg_color=bg_col, corner_radius=4)
            row.pack(fill="x", pady=2, padx=2)

            chk_var = ctk.BooleanVar(value=eff.enabled)
            def _on_toggle(e=eff, v=chk_var):
                e.enabled = v.get()
                self._sync_back()

            ctk.CTkCheckBox(row, text="", variable=chk_var, width=20, height=20,
                            command=_on_toggle).pack(side="left", padx=(6, 2), pady=6)

            lbl = ctk.CTkLabel(row, text=eff.name, font=("Segoe UI", 11, "bold" if is_sel else "normal"),
                               anchor="w")
            lbl.pack(side="left", fill="x", expand=True, padx=4, pady=6)

            # Click row to select
            def _select(i=idx):
                self.selected_effect_idx = i
                self._refresh_chain_list()
                self._refresh_params()

            row.bind("<Button-1>", lambda e, s=_select: s())
            lbl.bind("<Button-1>", lambda e, s=_select: s())

    def _refresh_params(self):
        for w in self._param_pane.winfo_children():
            w.destroy()

        if not self.chain.effects or self.selected_effect_idx >= len(self.chain.effects):
            ctk.CTkLabel(self._param_pane, text="Select an effect from the insert rack to edit its parameters.",
                         text_color="gray").pack(expand=True)
            return

        eff = self.chain.effects[self.selected_effect_idx]

        header = ctk.CTkFrame(self._param_pane, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(12, 6))
        ctk.CTkLabel(header, text=eff.name.upper(), font=("Segoe UI", 14, "bold"), text_color="#ff9900").pack(side="left")

        # Dynamic parameter controls
        scroll = ctk.CTkScrollableFrame(self._param_pane, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=12, pady=6)

        if isinstance(eff, ParametricEQEffect):
            self._build_eq_params(scroll, eff)
        elif isinstance(eff, CompressorEffect):
            self._build_compressor_params(scroll, eff)
        elif isinstance(eff, ReverbEffect):
            self._build_reverb_params(scroll, eff)
        elif isinstance(eff, StereoDelayEffect):
            self._build_delay_params(scroll, eff)
        elif isinstance(eff, LimiterEffect):
            self._build_limiter_params(scroll, eff)
        elif isinstance(eff, HighPassFilterEffect) or isinstance(eff, LowPassFilterEffect):
            self._build_filter_params(scroll, eff)
        elif isinstance(eff, VST3PluginEffect):
            self._build_vst3_params(scroll, eff)

    # ------------------------------------------------------------------
    # Parameter Sliders Builders
    # ------------------------------------------------------------------

    def _add_slider(self, parent, label: str, val: float, min_v: float, max_v: float, unit: str, on_change_fn):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=4)
        ctk.CTkLabel(row, text=label, width=130, anchor="w", font=("Segoe UI", 11, "bold")).pack(side="left")

        val_lbl = ctk.CTkLabel(row, text=f"{val:.1f} {unit}", width=65, anchor="e", font=("Consolas", 11, "bold"))
        val_lbl.pack(side="right", padx=(4, 0))

        def _on_move(v):
            v_float = float(v)
            val_lbl.configure(text=f"{v_float:.1f} {unit}")
            on_change_fn(v_float)
            self._sync_back()

        slider = ctk.CTkSlider(row, from_=min_v, to=max_v, height=18, command=_on_move)
        slider.set(val)
        slider.pack(side="left", fill="x", expand=True, padx=6)

    def _build_eq_params(self, parent, eq: ParametricEQEffect):
        ctk.CTkLabel(parent, text="Low Shelf Band", font=("Segoe UI", 11, "bold"), text_color="#00d2be").pack(anchor="w", pady=(4, 2))
        self._add_slider(parent, "Frequency", eq.low_shelf_freq, 20.0, 500.0, "Hz", lambda v: setattr(eq, "low_shelf_freq", v))
        self._add_slider(parent, "Gain", eq.low_shelf_gain, -18.0, 18.0, "dB", lambda v: setattr(eq, "low_shelf_gain", v))

        ctk.CTkLabel(parent, text="Mid Peaking 1", font=("Segoe UI", 11, "bold"), text_color="#00d2be").pack(anchor="w", pady=(8, 2))
        self._add_slider(parent, "Frequency", eq.mid1_freq, 100.0, 5000.0, "Hz", lambda v: setattr(eq, "mid1_freq", v))
        self._add_slider(parent, "Gain", eq.mid1_gain, -18.0, 18.0, "dB", lambda v: setattr(eq, "mid1_gain", v))
        self._add_slider(parent, "Q Factor", eq.mid1_q, 0.1, 8.0, "", lambda v: setattr(eq, "mid1_q", v))

        ctk.CTkLabel(parent, text="Mid Peaking 2", font=("Segoe UI", 11, "bold"), text_color="#00d2be").pack(anchor="w", pady=(8, 2))
        self._add_slider(parent, "Frequency", eq.mid2_freq, 500.0, 12000.0, "Hz", lambda v: setattr(eq, "mid2_freq", v))
        self._add_slider(parent, "Gain", eq.mid2_gain, -18.0, 18.0, "dB", lambda v: setattr(eq, "mid2_gain", v))
        self._add_slider(parent, "Q Factor", eq.mid2_q, 0.1, 8.0, "", lambda v: setattr(eq, "mid2_q", v))

        ctk.CTkLabel(parent, text="High Shelf Band", font=("Segoe UI", 11, "bold"), text_color="#00d2be").pack(anchor="w", pady=(8, 2))
        self._add_slider(parent, "Frequency", eq.high_shelf_freq, 2000.0, 20000.0, "Hz", lambda v: setattr(eq, "high_shelf_freq", v))
        self._add_slider(parent, "Gain", eq.high_shelf_gain, -18.0, 18.0, "dB", lambda v: setattr(eq, "high_shelf_gain", v))

    def _build_compressor_params(self, parent, comp: CompressorEffect):
        self._add_slider(parent, "Threshold", comp.threshold_db, -50.0, 0.0, "dB", lambda v: setattr(comp, "threshold_db", v))
        self._add_slider(parent, "Ratio", comp.ratio, 1.0, 20.0, ":1", lambda v: setattr(comp, "ratio", v))
        self._add_slider(parent, "Attack", comp.attack_ms, 1.0, 150.0, "ms", lambda v: setattr(comp, "attack_ms", v))
        self._add_slider(parent, "Release", comp.release_ms, 10.0, 800.0, "ms", lambda v: setattr(comp, "release_ms", v))
        self._add_slider(parent, "Makeup Gain", comp.makeup_gain_db, 0.0, 24.0, "dB", lambda v: setattr(comp, "makeup_gain_db", v))

    def _build_reverb_params(self, parent, rev: ReverbEffect):
        self._add_slider(parent, "Room Size", rev.room_size, 0.0, 1.0, "", lambda v: setattr(rev, "room_size", v))
        self._add_slider(parent, "Damping", rev.damping, 0.0, 1.0, "", lambda v: setattr(rev, "damping", v))
        self._add_slider(parent, "Wet Level", rev.wet_level, 0.0, 1.0, "", lambda v: setattr(rev, "wet_level", v))
        self._add_slider(parent, "Dry Level", rev.dry_level, 0.0, 1.0, "", lambda v: setattr(rev, "dry_level", v))

    def _build_delay_params(self, parent, dly: StereoDelayEffect):
        self._add_slider(parent, "Delay Time", dly.delay_ms, 10.0, 1000.0, "ms", lambda v: setattr(dly, "delay_ms", v))
        self._add_slider(parent, "Feedback", dly.feedback, 0.0, 0.95, "", lambda v: setattr(dly, "feedback", v))
        self._add_slider(parent, "Wet Mix", dly.mix, 0.0, 1.0, "", lambda v: setattr(dly, "mix", v))

    def _build_limiter_params(self, parent, lim: LimiterEffect):
        self._add_slider(parent, "Ceiling", lim.ceiling_db, -12.0, 0.0, "dBFS", lambda v: setattr(lim, "ceiling_db", v))
        self._add_slider(parent, "Release", lim.release_ms, 5.0, 200.0, "ms", lambda v: setattr(lim, "release_ms", v))

    def _build_filter_params(self, parent, flt):
        self._add_slider(parent, "Cutoff Frequency", flt.cutoff_freq, 20.0, 20000.0, "Hz", lambda v: setattr(flt, "cutoff_freq", v))

    def _build_vst3_params(self, parent, vst: VST3PluginEffect):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=6)
        ctk.CTkLabel(row, text="Plugin Path:", width=90, anchor="w").pack(side="left")
        p_lbl = ctk.CTkLabel(row, text=vst.plugin_path or "(No plugin loaded)", text_color="gray", anchor="w")
        p_lbl.pack(side="left", fill="x", expand=True, padx=6)

        def _browse():
            path = filedialog.askopenfilename(
                title="Select VST3 Audio Plugin",
                filetypes=[("VST3 Plugins", "*.vst3"), ("All Files", "*.*")],
                parent=self
            )
            if path:
                vst.plugin_path = path
                vst.name = os.path.basename(path).replace(".vst3", "")
                vst._load_plugin()
                self._sync_back()
                self._refresh_chain_list()
                self._refresh_params()

        ctk.CTkButton(parent, text="📂 Load / Change VST3 Plugin…", command=_browse).pack(pady=10)
        status_txt = "✔ Plugin Loaded & Ready" if vst._plugin_instance is not None else "⚠ Plugin not active (Install 'pedalboard' for VST3 hosting)"
        status_col = "#2ecc71" if vst._plugin_instance is not None else "orange"
        ctk.CTkLabel(parent, text=status_txt, text_color=status_col).pack(pady=4)
