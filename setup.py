#!/usr/bin/env python3
"""
rewise setup wizard.
Run once: python setup.py
"""
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


# ── Helpers ──────────────────────────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def hr():
    print("─" * 44)

def step(n, total, title):
    print()
    hr()
    print(f"  Step {n} of {total}  —  {title}")
    hr()
    print()

def ask(prompt, default=None):
    try:
        val = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else (default or "")

def run_cmd(cmd, cwd=None):
    try:
        subprocess.run(cmd, shell=True, cwd=str(cwd or ROOT))
    except KeyboardInterrupt:
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    clear()
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║          rewise  —  Setup            ║")
    print("  ╚══════════════════════════════════════╝")
    print()
    print("  This will get you running in 3 steps.")
    print()

    # ── Detect OS ────────────────────────────────────────────────────────────
    system = platform.system()
    os_names = {"Windows": "Windows", "Darwin": "Mac", "Linux": "Linux"}
    detected = os_names.get(system, system)
    is_windows = system == "Windows"

    print(f"  Detected OS: {detected}")
    change = ask("  Is that right? (y/n, default y): ", default="y")
    if change.lower() == "n":
        print()
        print("  Pick your OS:")
        print("    1. Windows")
        print("    2. Mac")
        print("    3. Linux")
        print()
        choice = ask("  Enter 1, 2, or 3: ", default="3")
        is_windows = choice == "1"

    # ── Step 1: Install dependencies ─────────────────────────────────────────
    step(1, 3, "Install dependencies")

    pip = f'"{sys.executable}" -m pip'
    print(f"  Command:  {pip} install -r requirements.txt")
    print()
    go = ask("  Press Enter to run it, or type 's' to skip: ", default="")
    if go.lower() != "s":
        print()
        run_cmd(f"{pip} install -r requirements.txt")
        print()
        print("  ✓ Done.")

    # ── Step 2: API keys ──────────────────────────────────────────────────────
    step(2, 3, "API keys")

    print("  You need two free API keys.")
    print()
    print("  OpenRouter  (LLM + embeddings)")
    print("  → https://openrouter.ai/keys")
    print()
    print("  Cohere  (reranker)")
    print("  → https://dashboard.cohere.com/api-keys")
    print()

    env_path = ROOT / ".env"
    existing = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    def get_key(name, env_var, placeholder):
        current = existing.get(env_var, "")
        if current:
            preview = current[:10] + "…" if len(current) > 10 else current
            print(f"  {name} key already saved ({preview})")
            change = ask("  Change it? (y/n, default n): ", default="n")
            if change.lower() != "y":
                return current
        val = ask(f"  Paste your {name} key: ", default="")
        while not val:
            print("  (Key cannot be empty.)")
            val = ask(f"  Paste your {name} key: ", default="")
        return val

    or_key = get_key("OpenRouter", "OPENROUTER_API_KEY", "sk-or-v1-…")
    print()
    co_key = get_key("Cohere", "COHERE_API_KEY", "…")

    env_path.write_text(
        f"OPENROUTER_API_KEY={or_key}\nCOHERE_API_KEY={co_key}\n",
        encoding="utf-8",
    )
    print()
    print("  ✓ Keys saved to .env")

    # ── Step 3: Start the server ──────────────────────────────────────────────
    step(3, 3, "Start the server")

    if is_windows:
        print("  Run these two commands in your terminal:")
        print()
        print("    cd src")
        print("    uvicorn app:app --reload")
    else:
        print("  Run this in your terminal:")
        print()
        print("    cd src && uvicorn app:app --reload")

    print()
    go = ask("  Start it now in this window? (y/n, default y): ", default="y")
    print()

    if go.lower() != "n":
        print("  Starting… open  http://localhost:8000  in your browser.")
        print("  Press Ctrl+C to stop the server.")
        print()
        run_cmd("uvicorn app:app --reload", cwd=ROOT / "src")
    else:
        print("  Whenever you're ready, run:")
        if is_windows:
            print("    cd src")
            print("    uvicorn app:app --reload")
        else:
            print("    cd src && uvicorn app:app --reload")
        print()
        print("  Then open:  http://localhost:8000")

    print()
    hr()
    print("  All done. Enjoy rewise!")
    hr()
    print()


if __name__ == "__main__":
    main()
