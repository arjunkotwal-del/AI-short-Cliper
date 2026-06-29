import threading
import subprocess

cancel_event = threading.Event()
active_processes = set()
active_processes_lock = threading.Lock()

# Save original Popen
_original_Popen = subprocess.Popen

class WatchedPopen(_original_Popen):
    def __init__(self, *args, **kwargs):
        if cancel_event.is_set():
            raise RuntimeError("Pipeline execution cancelled.")
        super().__init__(*args, **kwargs)
        with active_processes_lock:
            # Prune finished processes to avoid memory leaks
            for p in list(active_processes):
                if p.poll() is not None:
                    active_processes.discard(p)
            active_processes.add(self)

# Monkey patch subprocess Popen to automatically register all spawned commands
subprocess.Popen = WatchedPopen

def check_cancelled():
    if cancel_event.is_set():
        raise RuntimeError("Pipeline execution cancelled.")

def cancel_pipeline():
    cancel_event.set()
    with active_processes_lock:
        print(f"[Cancellation] Terminating {len(active_processes)} active subprocesses...", flush=True)
        for proc in list(active_processes):
            try:
                proc.terminate()
            except Exception as e:
                print(f"[Cancellation] Error terminating process: {e}", flush=True)
        active_processes.clear()

def reset_cancellation():
    cancel_event.clear()
