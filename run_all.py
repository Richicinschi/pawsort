"""Start the three PawSort servers. Ctrl+C stops all of them."""
import subprocess
import sys
import time

COMMANDS = [
    ("orchestrator", [sys.executable, "-m", "uvicorn", "orchestrator.app:build_app", "--factory", "--port", "8001", "--no-access-log"]),
    ("interface", [sys.executable, "-m", "uvicorn", "interface.app:build_app", "--factory", "--port", "8000"]),
    ("ai server", [sys.executable, "-m", "ai_server.worker"]),
]


def main() -> None:
    procs = []
    try:
        for name, cmd in COMMANDS:
            procs.append(subprocess.Popen(cmd))
            print(f"started {name}")
            time.sleep(1.5)
        print("PawSort is running: open http://127.0.0.1:8000  (Ctrl+C to stop)")
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
