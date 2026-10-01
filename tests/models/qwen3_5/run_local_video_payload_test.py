"""Generate a deterministic H.264/MP4 fixture and run the local-video payload regression."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_local_video_payload_test.py TEST_EXECUTABLE")
    if shutil.which("ffmpeg") is None:
        print("SKIP: ffmpeg executable is unavailable")
        raise SystemExit(77)
    with tempfile.TemporaryDirectory(prefix="ninfer-local-video-") as directory:
        video = Path(directory) / "seven-frames.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=96x64:rate=7",
                "-frames:v",
                "7",
                "-c:v",
                "libx264",
                "-g",
                "7",
                "-bf",
                "3",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
        )
        subprocess.run([sys.argv[1], str(video)], check=True)


if __name__ == "__main__":
    main()
