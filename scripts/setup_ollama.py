"""
Check Ollama is running and pull qwen2.5:3b if needed.
Called by: make setup

Usage:
    uv run python scripts/setup_ollama.py
"""

import subprocess
import sys
import json
import urllib.request
import urllib.error

TARGET_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434"


def check_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def list_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def main():
    print("Checking Ollama...")

    if not check_running():
        print("ERROR: Ollama is not running.")
        print()
        print("Start it with one of these methods:")
        print("  Windows: Open Ollama from Start Menu, or run 'ollama serve' in a terminal")
        print("  Then re-run this script.")
        sys.exit(1)

    print("✓ Ollama is running")

    models = list_models()
    print(f"  Available models: {models if models else '(none)'}")

    if any(TARGET_MODEL in m for m in models):
        print(f"✓ {TARGET_MODEL} already available")
        return

    print(f"Pulling {TARGET_MODEL} (1.9 GB — this takes a few minutes)...")
    result = subprocess.run(["ollama", "pull", TARGET_MODEL])

    if result.returncode == 0:
        print(f"✓ {TARGET_MODEL} downloaded successfully")
    else:
        print(f"ERROR: Failed to pull {TARGET_MODEL}")
        print("Try manually: ollama pull qwen2.5:3b")
        sys.exit(1)


if __name__ == "__main__":
    main()
