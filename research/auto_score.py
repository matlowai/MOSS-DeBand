#!/usr/bin/env python3
"""Every few minutes, score drift arms whose render count changed and rebuild compare_solve.html. Stops when the queue is empty.

  comfyui_venv/bin/python workflows/moss/auto_score.py arm [arm ...]
"""
import glob, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "output", "moss", "drift")
PY = sys.executable
arms, seen = sys.argv[1:], {}


def queue_empty():
    try:
        q = json.load(urllib.request.urlopen("http://127.0.0.1:8188/queue"))
    except OSError:  # server restarting
        return False
    return not q["queue_running"] and not q["queue_pending"]


while True:
    last = queue_empty()
    changed = [a for a in arms if len(glob.glob(os.path.join(D, a, "*.png"))) != seen.get(a, 0)]
    for a in changed:
        seen[a] = len(glob.glob(os.path.join(D, a, "*.png")))
    if changed:
        out = subprocess.run([PY, os.path.join(HERE, "drift_score.py"), "sft", *changed], capture_output=True, text=True).stdout
        print(time.strftime("%H:%M"), "\n".join(l for l in out.splitlines() if l.startswith(("===", "  solid", "  people"))), flush=True)
        subprocess.run([PY, os.path.join(HERE, "build_compare.py")])
    if last:
        break
    time.sleep(600)
