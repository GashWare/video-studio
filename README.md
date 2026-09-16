# 🎬 Video Studio — Enterprise Non-Linear Video Editor & Audio DAW

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Tests: Passing](https://img.shields.io/badge/Tests-118%20Passing-success.svg)](tests/)

**Video Studio** is a lightweight, high-performance, enterprise-grade Non-Linear Video Editor (NLE) and Digital Audio Workstation (DAW) built with Python, CustomTkinter, NumPy, SciPy, OpenCV, and MoviePy.

Designed for professional creators, editors, and audio engineers, Video Studio combines real-time video scoping, 3D LUT color grading, precision timeline trimming, DAW-style audio mixing with VST3 hosting, and modern AI automation.

---

## 🌟 Key Features

### 🎞️ 1. Multi-Track Timeline & Precision Trimming
- **Multi-Track Canvas**: Flexible video (`V#`), audio (`A#`), title (`T#`), subtitle (`S#`), and image (`I#`) tracks.
- **NLE Precision Trimming**:
  - **Ripple Delete** (`Shift+Del`): Closes timeline gaps automatically while maintaining A/V sync.
  - **Slip Tool**: Adjusts media in/out points while preserving timeline position and sequence duration.
  - **Slide Tool**: Shifts clips along the timeline while dynamically trimming adjacent neighbor clips.
  - **Razor Split** (`S` or `Ctrl+K`) and **Duplicate** (`D` or `Ctrl+D`).
- **Audio-Video Separation & Linking**: Extract audio from video to dedicated tracks, lock sync, and perform synchronized cuts.

### 🎨 2. Studio Colorist Suite & Real-Time Scopes
- **Studio Video Scopes**:
  - **Luma Waveform Monitor**: Real-time ITU-R BT.709 luma distribution with phosphor green glow ($0\%\text{--}100\%$ IRE).
  - **RGB Parade**: Synchronized Red, Green, and Blue intensity profiles side-by-side.
  - **Vectorscope**: Polar color-wheel chrominance plot with standardized I/Q skin-tone reference axis ($103^\circ$).
  - **Histogram**: Real-time luminance histogram with 256 bins and IRE overlays.
- **3D `.cube` LUT Loader**: Full parser and trilinear 3D interpolation engine for camera log normalization (Sony S-Log3, Canon C-Log, Panasonic V-Log, ARRI LogC) and creative look grading.
- **Color Grading**: Brightness, contrast, saturation, hue, and 3-Way Lift / Gamma / Gain balance controls.

### 🎛️ 3. Studio Audio FX Rack & VST3 Plugin Chains
- **FL Studio-Style Mixer Controls**:
  - Per-track **Volume rotary knobs** ($0\%\text{--}200\%$).
  - Per-track **Bipolar Stereo Panning knobs** ($100\%\text{ L} \longleftrightarrow 100\%\text{ R}$ with center detent).
- **Studio DSP Audio Effects Rack**:
  - **4-Band Parametric Equalizer**: Biquad IIR filters with Low Shelf, High Shelf, and 2x Peaking Bell filters.
  - **Studio Dynamic Compressor**: Configurable threshold, ratio ($1:1\text{--}20:1$), attack ($1\text{--}150\text{ ms}$), release ($10\text{--}800\text{ ms}$), and makeup gain.
  - **Stereo Reverb & Delay**: Algorithmic room reverberator and feedback delay.
  - **Brickwall Peak Limiter & Maximizer**: True peak ceiling protection with soft clipping.
  - **High-Pass & Low-Pass Filters**: 2nd-order Butterworth cut filters for rumble and hiss removal.
- **VST3 Plugin Hosting**: Load standard external `.vst3` plugin bundles (FabFilter, iZotope, Waves, Valhalla, Vital, etc.).
- **Studio Presets**: Dialogue Clarity (Podcast), Vocal Warmth, Broadcast Maximizer, Studio Reverb, Slapback Delay, and Radio Voice.

### ✨ 4. Motion Graphics & 14 Blend Modes
- **14 NLE / Photoshop Blend Modes**: Vectorized float32 implementations for `Normal`, `Multiply`, `Screen`, `Overlay`, `Soft Light`, `Hard Light`, `Color Dodge`, `Color Burn`, `Darken`, `Lighten`, `Difference`, `Exclusion`, `Luminosity`, `Add`, and `Subtract`.
- **Keyframe Velocity Curves**: `Linear`, `Ease In`, `Ease Out`, `Ease In/Out`, `Exponential`, and 5th-order `Bezier (S-Curve)` smootherstep curve models for transform and opacity parameters.

### 🤖 5. AI Automation Suite
- **OpenAI Whisper Auto-Subtitles**: Transcribes speech directly to subtitle tracks (`S1`, `T1`).
- **AI Scene Cut Detection**: Automated frame-accurate scene change detection and clip splitting via PySceneDetect.
- **AI Smart Reframe (9:16)**: MediaPipe face tracking dynamically centers subjects for vertical formats (Shorts/Reels/TikTok).
- **AI Audio Auto-Ducking**: Sidechain-style dialogue detection that smoothly ducks background music ($-12\text{ dB}$) during speech.
- **AI Spectral Noise Reduction**: Eliminates background hum and hiss using spectral gating.

### ⚡ 6. Hardware Acceleration & Studio Interchange
- **GPU Hardware Encoding**: Direct presets for **NVIDIA NVENC** (`h264_nvenc`, `hevc_nvenc`), **Intel QuickSync** (`h264_qsv`), **AMD AMF** (`h264_amf`), and software encoders (`libx264`, `libx265`, `libvpx-vp9`).
- **Studio Sequence Interchange**:
  - **CMX 3600 EDL**: Standard Edit Decision List export and import.
  - **Final Cut Pro 7 XML** (`<xmeml version="4">`): Full multi-track sequence interchange compatible with **DaVinci Resolve**, **Adobe Premiere Pro**, and **Final Cut Pro**.

---

## ⌨️ Universal Keyboard Shortcuts

| Key | Action | Description |
|---|---|---|
| <kbd>Space</kbd> | **Play / Pause** | Toggle video & audio timeline playback |
| <kbd>J</kbd> | **Shuttle Reverse** | Nudge playhead backwards by 5 frames |
| <kbd>K</kbd> | **Pause** | Instant playback pause |
| <kbd>L</kbd> | **Shuttle Forward** | Nudge playhead forward by 5 frames |
| <kbd>←</kbd> / <kbd>→</kbd> | **Frame Step** | Step playhead backward/forward exactly 1 video frame |
| <kbd>Shift</kbd>+<kbd>←</kbd> / <kbd>→</kbd> | **1-Second Jump** | Jump playhead backward/forward by 1.0 second |
| <kbd>Home</kbd> / <kbd>End</kbd> | **Start / End** | Jump playhead to start ($0.0\text{s}$) or sequence end |
| <kbd>I</kbd> / <kbd>O</kbd> | **Mark In / Mark Out** | Set source/clip In or Out trim points |
| <kbd>S</kbd> / <kbd>Ctrl</kbd>+<kbd>K</kbd> | **Split Clip** | Razor split active clip at playhead |
| <kbd>D</kbd> / <kbd>Ctrl</kbd>+<kbd>D</kbd> | **Duplicate Clip** | Duplicate active clip immediately adjacent on timeline |
| <kbd>Del</kbd> / <kbd>Backspace</kbd> | **Delete Clip** | Remove active clip from timeline (Undo supported) |
| <kbd>Shift</kbd>+<kbd>Del</kbd> | **Ripple Delete** | Remove clip and shift downstream clips left to close the gap |
| <kbd>M</kbd> | **Add Marker** | Drop a visual timeline marker at playhead |
| <kbd>F11</kbd> / <kbd>`</kbd> | **Full-Screen Preview**| Toggle borderless fullscreen video playback |
| <kbd>=</kbd> / <kbd>-</kbd> | **Zoom Timeline** | Zoom in / out timeline scale |
| <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd>| **Undo / Redo** | Multi-level CommandStack undo/redo |
| <kbd>Ctrl</kbd>+<kbd>,</kbd> | **Preferences** | Open Studio Preferences dialog |

---

## 📦 Installation & Setup

### Prerequisites
- **Python 3.10+** (64-bit recommended)
- **FFmpeg** installed and accessible on your system `PATH`

### 1. Clone the Repository
```bash
git clone https://github.com/GashWare/video-studio.git
cd video-studio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional AI & DSP Dependencies
To enable Whisper subtitles, VST3 hosting, and computer vision features:
```bash
# AI Subtitles & Face Tracking
pip install openai-whisper mediapipe scenedetect[opencv]

# VST3 Plugin Hosting & Audio Tools
pip install pedalboard noisereduce librosa
```

### 4. Launch Video Studio
```bash
python main.py
```

---

## 🧪 Testing & Verification

Run the full pytest suite:
```bash
pytest tests/test_core.py -v
```
All 118 unit and integration tests verify:
- Timeline commands, multi-level undo/redo, slip/slide/ripple trimming
- Real-time video scopes (Waveform, RGB Parade, Vectorscope, Histogram)
- 3D LUT `.cube` parsing and trilinear interpolation
- 14 blend modes and keyframe velocity curves
- Parametric EQ, Compressor, Reverb, Delay, and Limiter DSP processors
- CMX 3600 EDL and FCP7 XML sequence interchange round-tripping

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
