"""
Download a YouTube football clip for testing.

Usage:
    uv run python scripts/download_clip.py <URL> [--name sample] [--seconds 60]

Examples:
    uv run python scripts/download_clip.py "https://www.youtube.com/watch?v=VIDEO_ID"
    uv run python scripts/download_clip.py "URL" --name arsenal_vs_chelsea --seconds 90
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "test_clips"

# Common Windows ffmpeg install locations (winget, choco, manual)
_FFMPEG_CANDIDATES = [
    r"C:\Users\maith\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
]


def _find_ffmpeg() -> str | None:
    """Return path to ffmpeg binary, or None if not found."""
    # Try PATH first
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Try known Windows install locations
    for candidate in _FFMPEG_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def download(url: str, name: str = "sample", seconds: int = 60, start: int = 0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{name}.mp4"

    ffmpeg_path = _find_ffmpeg()
    if not ffmpeg_path:
        print("ERROR: ffmpeg not found. Install with: winget install Gyan.FFmpeg")
        print("       Then open a new terminal.")
        sys.exit(1)

    ffmpeg_dir = str(Path(ffmpeg_path).parent)
    end = start + seconds

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg_dir,
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "-o", str(output_path),
        "--no-playlist",
        "--no-warnings",
        url,
    ]

    label = f"{start}s-{end}s" if start > 0 else f"first {seconds}s"
    print(f"Downloading {label} -> {output_path.name}")
    print(f"  ffmpeg: {ffmpeg_path}")
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0 and output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nSaved: {output_path.name} ({size_mb:.1f} MB)")
        return output_path
    else:
        print("\nDownload failed. Check the URL and try again.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a YouTube football clip")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--name", default="sample", help="Output filename (without .mp4)")
    parser.add_argument("--seconds", type=int, default=60, help="Clip duration to download")
    parser.add_argument("--start", type=int, default=0, help="Start offset in seconds")
    args = parser.parse_args()
    download(args.url, args.name, args.seconds, args.start)
