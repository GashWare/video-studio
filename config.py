import os
import json
import logging
import ctypes
import gc

CONFIG_FILE = "config.json"

class ConfigManager:
    """Handles loading and immediate saving of application state."""
    def __init__(self):
        self.defaults = {
            "geometry": None, 
            "theme": "Dark", 
            "accent_color": "blue", 
            "log_level": "INFO", 
            "export_quality": 85
        }
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

    # Alias so project_io.py can call either save_config() or set()
    def set(self, key, value):
        self.save_config(key, value)

    def get(self, key, default=None):
        return self.config.get(key, default)

config_mgr = ConfigManager()

def setup_logging():
    logger = logging.getLogger()
    log_lvl = config_mgr.get("log_level") or "INFO"
    logger.setLevel(getattr(logging, log_lvl.upper(), logging.INFO))
    if logger.hasHandlers():
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler with explicit UTF-8
    file_handler = logging.FileHandler("app.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Safe stream handler with UTF-8 / replace
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

def toggle_console(show: bool):
    if os.name == 'nt':
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd: ctypes.windll.user32.ShowWindow(hwnd, 5 if show else 0)
        except Exception: pass

def free_memory():
    """Aggressively reclaims system RAM and GPU VRAM."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass