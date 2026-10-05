"""Use independently inspected presentation timestamps as the source-service oracle."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout

with tempfile.TemporaryDirectory(prefix="ninfer-video-source-") as directory:
    root = Path(directory)
    manifest = []
    for name, filters, audio, interlace, vfr in [
        ("cfr", "setpts=PTS+100/TB", True, "progressive", False),
        ("vfr", "setpts=if(lt(N\\,4)\\,N\\,4+2*(N-4))/(4*TB)", False, "progressive", True),
        ("interlaced", "setfield=tff", False, "interlaced", False),
    ]:
        path = root / (name + ".mp4")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
               "testsrc2=size=96x64:rate=4:duration=2"]
        if audio:
            cmd += ["-f", "lavfi", "-i", "sine=frequency=400:duration=2"]
        cmd += ["-vf", filters, "-c:v", "libx264", "-bf", "3", "-g", "4", "-fps_mode", "vfr"]
        if interlace == "interlaced":
            cmd += ["-flags", "+ilme+ildct", "-x264-params", "tff=1"]
        if audio:
            cmd += ["-c:a", "aac"]
        run(*cmd, str(path))
        probe = json.loads(run("ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
                               "-of", "json", str(path)))
        stamps = [float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]]
        manifest.append({"path": str(path), "times": [value-stamps[0] for value in stamps],
                         "audio": int(audio), "interlace": interlace, "vfr": vfr, "tie": name == "cfr"})
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    subprocess.run([sys.argv[1], str(manifest_path)], check=True)
