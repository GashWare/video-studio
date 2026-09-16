# interchange.py
"""
Studio Sequence Interchange for Video Studio.
Supports:
- CMX 3600 Edit Decision List (.edl) Import & Export
- Final Cut Pro 7 XML (.xml) Sequence Interchange (DaVinci Resolve / Premiere Pro / FCP compatible)
"""

import os
import re
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, unquote, quote
from clip_data import make_clip


# ---------------------------------------------------------------------------
# Timecode Helpers
# ---------------------------------------------------------------------------

def seconds_to_timecode(seconds: float, fps: float = 30.0) -> str:
    """Convert floating seconds to standard non-drop frame timecode HH:MM:SS:FF."""
    if seconds < 0:
        seconds = 0.0
    total_frames = int(round(seconds * fps))
    frames = total_frames % int(round(fps))
    total_sec = total_frames // int(round(fps))
    secs = total_sec % 60
    mins = (total_sec // 60) % 60
    hours = total_sec // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def timecode_to_seconds(tc: str, fps: float = 30.0) -> float:
    """Convert standard timecode HH:MM:SS:FF (or HH:MM:SS;FF) to floating seconds."""
    if not tc or not isinstance(tc, str):
        return 0.0
    parts = re.split(r"[:;]", tc.strip())
    if len(parts) == 4:
        try:
            h, m, s, f = [int(p) for p in parts]
            return h * 3600.0 + m * 60.0 + s + (f / fps)
        except ValueError:
            return 0.0
    elif len(parts) == 3:
        try:
            h, m, s = [float(p) for p in parts]
            return h * 3600.0 + m * 60.0 + s
        except ValueError:
            return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# CMX 3600 EDL Export & Import
# ---------------------------------------------------------------------------

def export_edl(project_clips: list, out_path: str, fps: float = 30.0, title: str = "VIDEO_STUDIO_SEQUENCE") -> str:
    """
    Export project clips to standard CMX 3600 EDL format.
    Returns the output path.
    """
    lines = [
        f"TITLE: {title.upper().replace(' ', '_')}",
        "FCM: NON-DROP FRAME",
        ""
    ]

    # Filter video/visual clips sorted by timeline start
    v_clips = sorted(
        [c for c in project_clips if not str(c.get("track", "")).startswith("A")],
        key=lambda x: x.get("start", 0.0)
    )

    event_num = 1
    for c in v_clips:
        src_in = c.get("trim_in", 0.0)
        src_out = c.get("trim_out", src_in + 1.0)
        rec_in = c.get("start", 0.0)
        rec_out = rec_in + (src_out - src_in)

        src_in_tc = seconds_to_timecode(src_in, fps)
        src_out_tc = seconds_to_timecode(src_out, fps)
        rec_in_tc = seconds_to_timecode(rec_in, fps)
        rec_out_tc = seconds_to_timecode(rec_out, fps)

        reel = f"AX{event_num:02d}"[:8].ljust(8)
        line = f"{event_num:03d}  {reel} V     C        {src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}"
        lines.append(line)

        clip_name = os.path.basename(c.get("path", "CLIP"))
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        lines.append("")
        event_num += 1

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def import_edl(edl_path: str, fps: float = 30.0) -> list:
    """
    Import a CMX 3600 EDL file into a list of clip dictionaries.
    """
    if not os.path.isfile(edl_path):
        return []

    clips = []
    with open(edl_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_clip_name = None
    event_pattern = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2})\s+(\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2})\s+(\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2})\s+(\d{2}[:;]\d{2}[:;]\d{2}[:;]\d{2})")

    pending_event = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = event_pattern.match(line)
        if match:
            if pending_event is not None:
                clips.append(pending_event)
                pending_event = None

            _ev, _reel, trk_type, _trans, src_in_tc, src_out_tc, rec_in_tc, rec_out_tc = match.groups()
            src_in = timecode_to_seconds(src_in_tc, fps)
            src_out = timecode_to_seconds(src_out_tc, fps)
            rec_in = timecode_to_seconds(rec_in_tc, fps)
            rec_out = timecode_to_seconds(rec_out_tc, fps)
            track = "A1" if "A" in trk_type.upper() else "V1"
            t_out = src_out if src_out > src_in else src_in + (rec_out - rec_in)
            clip_dict = make_clip(
                path="imported_clip",
                track=track,
                start=rec_in,
                duration=max(0.1, t_out - src_in),
                trim_in=src_in,
                trim_out=t_out,
            )
            pending_event = clip_dict

        elif line.startswith("* FROM CLIP NAME:"):
            name = line.replace("* FROM CLIP NAME:", "").strip()
            if pending_event is not None:
                pending_event["path"] = name

    if pending_event is not None:
        clips.append(pending_event)

    return clips


# ---------------------------------------------------------------------------
# Final Cut Pro 7 XML (xmeml v4) Export & Import
# ---------------------------------------------------------------------------

def export_fcpxml(project_clips: list, out_path: str, proj_w: int = 1920, proj_h: int = 1080, fps: float = 30.0, title: str = "Video Studio Sequence") -> str:
    """
    Export project clips to Final Cut Pro 7 XML format (<xmeml version="4">).
    Supported by DaVinci Resolve, Adobe Premiere Pro, Final Cut Pro.
    """
    root = ET.Element("xmeml", version="4")
    sequence = ET.SubElement(root, "sequence")
    ET.SubElement(sequence, "name").text = title

    fps_int = int(round(fps))
    total_frames = int(max([c.get("start", 0) + (c.get("trim_out", 0) - c.get("trim_in", 0)) for c in project_clips] or [10.0]) * fps_int)

    ET.SubElement(sequence, "duration").text = str(total_frames)

    # Rate
    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(fps_int)
    ET.SubElement(rate, "ntsc").text = "FALSE"

    # Timecode
    tc_elem = ET.SubElement(sequence, "timecode")
    tc_rate = ET.SubElement(tc_elem, "rate")
    ET.SubElement(tc_rate, "timebase").text = str(fps_int)
    ET.SubElement(tc_elem, "string").text = "00:00:00:00"
    ET.SubElement(tc_elem, "frame").text = "0"

    # Media
    media = ET.SubElement(sequence, "media")

    # Video track container
    video = ET.SubElement(media, "video")
    format_elem = ET.SubElement(video, "format")
    sample_char = ET.SubElement(format_elem, "samplecharacteristics")
    ET.SubElement(sample_char, "width").text = str(proj_w)
    ET.SubElement(sample_char, "height").text = str(proj_h)

    # Audio track container
    audio = ET.SubElement(media, "audio")

    # Group clips by track
    tracks = {}
    for c in project_clips:
        trk = c.get("track", "V1")
        tracks.setdefault(trk, []).append(c)

    for trk_name, clip_list in sorted(tracks.items()):
        is_audio = trk_name.startswith("A")
        parent_media = audio if is_audio else video
        track_elem = ET.SubElement(parent_media, "track")

        for c in clip_list:
            clipitem = ET.SubElement(track_elem, "clipitem", id=f"clipitem-{c.get('id', uuid.uuid4())}")
            c_name = os.path.basename(c.get("path", "clip"))
            ET.SubElement(clipitem, "name").text = c_name

            dur_sec = max(0.01, c.get("trim_out", 1.0) - c.get("trim_in", 0.0))
            dur_frames = int(round(dur_sec * fps_int))
            start_frame = int(round(c.get("start", 0.0) * fps_int))
            in_frame = int(round(c.get("trim_in", 0.0) * fps_int))
            out_frame = in_frame + dur_frames

            ET.SubElement(clipitem, "duration").text = str(dur_frames)
            ET.SubElement(clipitem, "start").text = str(start_frame)
            ET.SubElement(clipitem, "end").text = str(start_frame + dur_frames)
            ET.SubElement(clipitem, "in").text = str(in_frame)
            ET.SubElement(clipitem, "out").text = str(out_frame)

            # File ref
            file_elem = ET.SubElement(clipitem, "file", id=f"file-{uuid.uuid4()}")
            ET.SubElement(file_elem, "name").text = c_name
            file_path = c.get("path", "")
            if file_path and not file_path.startswith("__"):
                path_url = "file://localhost/" + quote(os.path.abspath(file_path).replace("\\", "/"))
                ET.SubElement(file_elem, "pathurl").text = path_url

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def import_fcpxml(xml_path: str, fps: float = 30.0) -> list:
    """
    Import Final Cut Pro 7 XML sequence into a list of clip dictionaries.
    """
    if not os.path.isfile(xml_path):
        return []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    fps_val = fps
    rate_elem = root.find(".//sequence/rate/timebase")
    if rate_elem is not None and rate_elem.text:
        try:
            fps_val = float(rate_elem.text)
        except ValueError:
            pass

    clips = []

    # Parse video tracks
    v_tracks = root.findall(".//media/video/track")
    for trk_idx, v_trk in enumerate(v_tracks, start=1):
        trk_name = f"V{trk_idx}"
        for item in v_trk.findall("clipitem"):
            start_f = int(item.findtext("start", "0"))
            in_f = int(item.findtext("in", "0"))
            out_f = int(item.findtext("out", str(in_f + int(item.findtext("duration", "30")))))

            pathurl = item.findtext(".//file/pathurl", "")
            if pathurl:
                parsed = urlparse(pathurl)
                path = unquote(parsed.path)
                if path.startswith("/") and len(path) > 2 and path[2] == ":":
                    path = path[1:]  # Windows drive letter /C:/ -> C:/
            else:
                path = item.findtext("name", "clip")

            t_in = in_f / fps_val
            t_out = out_f / fps_val
            c = make_clip(
                path=path,
                track=trk_name,
                start=start_f / fps_val,
                duration=max(0.1, t_out - t_in),
                trim_in=t_in,
                trim_out=t_out,
            )
            clips.append(c)

    # Parse audio tracks
    a_tracks = root.findall(".//media/audio/track")
    for trk_idx, a_trk in enumerate(a_tracks, start=1):
        trk_name = f"A{trk_idx}"
        for item in a_trk.findall("clipitem"):
            start_f = int(item.findtext("start", "0"))
            in_f = int(item.findtext("in", "0"))
            out_f = int(item.findtext("out", str(in_f + int(item.findtext("duration", "30")))))

            pathurl = item.findtext(".//file/pathurl", "")
            if pathurl:
                parsed = urlparse(pathurl)
                path = unquote(parsed.path)
                if path.startswith("/") and len(path) > 2 and path[2] == ":":
                    path = path[1:]
            else:
                path = item.findtext("name", "audio_clip")

            t_in = in_f / fps_val
            t_out = out_f / fps_val
            c = make_clip(
                path=path,
                track=trk_name,
                start=start_f / fps_val,
                duration=max(0.1, t_out - t_in),
                trim_in=t_in,
                trim_out=t_out,
            )
            clips.append(c)

    return clips
