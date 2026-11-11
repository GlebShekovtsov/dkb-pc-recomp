"""Benchmark dkb.exe the way a weak PC would run it, and print one comparable table.

Runs the game for a fixed time with TOTK_TRACE=perf and the draw profile on,
optionally restricted to the first N logical processors, samples its memory
every 5 s, then parses the game log. Every optimisation gets a before/after
table from this. Windows first (affinity and memory sampling use ctypes);
standard library only. Parsing is in parse_log(), which tests/lowend_bench_test.py
feeds with a synthetic log.

Frame figures are per 60-frame window (one perf line), not per frame: the
"1% low" is the 99th percentile of the per-window max interval.
"""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

RE_BOOT = re.compile(r"boot: frames (\d+) presented \(.*?validation errors (\d+), shader failures (\d+), draws skipped (\d+)\)")
RE_FRAME = re.compile(r"perf: frame (\d+): interval avg ([\d.]+) ms max ([\d.]+) ms \(([\d.]+) fps\); render thread blocked: "
                      r"pacing (\d+)% fence (\d+)% drain (\d+)% pipeline build (\d+)% shader xlate (\d+)% texture (\d+)%")
RE_GPU = re.compile(r"perf: frame \d+: gpu avg ([\d.]+) ms")
RE_REPLAY = re.compile(r"replay avg ([\d.]+) ms")
RE_PROFILE = re.compile(r"vk_exec: draw profile over \d+ draws .*?ns per draw: (.*)$")
BLOCK_NAMES = ("pacing", "fence", "drain", "pipeline", "xlate", "texture")


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def parse_log(text):
    """Return the figures found in a game log; missing ones are None (or 0 for counts)."""
    r = dict(frames=None, errors=None, shader_failures=None, draws_skipped=None,
             windows=0, steady_fps=None, worst_ms=None, low1_ms=None, blocked=None,
             gpu_ms=None, replay_ms=None, profile=None, svc_blocked=0, romfs_slow=0)
    avgs, maxs, gpus, replays = [], [], [], []
    blocked = [[] for _ in BLOCK_NAMES]
    profile = None
    for line in text.splitlines():
        m = RE_BOOT.search(line)
        if m:
            r["frames"], r["errors"], r["shader_failures"], r["draws_skipped"] = map(int, m.groups())
        m = RE_FRAME.search(line)
        if m:
            avgs.append(float(m.group(2)))
            maxs.append(float(m.group(3)))
            for lst, v in zip(blocked, m.groups()[4:]):
                lst.append(int(v))
            m2 = RE_REPLAY.search(line)
            if m2:
                replays.append(float(m2.group(1)))
        m = RE_GPU.search(line)
        if m:
            gpus.append(float(m.group(1)))
        m = RE_PROFILE.search(line)
        if m:
            profile = m.group(1).split("; total")[0]
        if "svc: " in line and " blocked for " in line:
            r["svc_blocked"] += 1
        if "romfs: slow read" in line:
            r["romfs_slow"] += 1
    r["windows"] = len(avgs)
    if avgs:
        r["steady_fps"] = 1000 / statistics.median(avgs)
        r["worst_ms"] = max(maxs)
        r["low1_ms"] = pct(maxs, 99)
        r["blocked"] = [statistics.median(v) for v in blocked]
    if gpus:
        r["gpu_ms"] = statistics.mean(gpus)
    if replays:
        r["replay_ms"] = statistics.mean(replays)
    r["profile"] = profile
    return r


def fmt(v, spec="%.1f"):
    return "-" if v is None else spec % v


if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as w

    class PMC(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi")
    k32.OpenProcess.restype = w.HANDLE
    k32.SetProcessAffinityMask.argtypes = [w.HANDLE, ctypes.c_size_t]
    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(PMC), w.DWORD]

    def proc_handle(pid):
        return k32.OpenProcess(0x0400 | 0x0200 | 0x1000, False, pid)   # query, set information, limited query

    def set_affinity(h, cpus):
        k32.SetProcessAffinityMask(h, (1 << cpus) - 1)

    def mem_sample(h):
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return None
        return pmc.PrivateUsage, pmc.WorkingSetSize, pmc.PeakPagefileUsage
else:
    def proc_handle(pid):
        return None

    def set_affinity(h, cpus):
        print("lowend_bench: --cpus needs Windows, ignored", file=sys.stderr)

    def mem_sample(h):
        return None


def gpu_mem_mb():
    # ponytail: system-wide nvidia-smi reading (not per process), '-' on other vendors
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return float(out[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.label or "run")
    log_path, csv_path = out / (stem + ".log"), out / (stem + "-mem.csv")
    env = dict(os.environ, TOTK_TRACE="perf", DKB_DRAW_PROFILE="1", TOTK_DRAW_PROFILE="1")
    if args.script:
        env["TOTK_INPUT_SCRIPT"] = args.script
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    exe = Path(args.exe).resolve()
    samples = []
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen([str(exe), "--seconds", str(args.seconds)], cwd=exe.parent, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
        h = proc_handle(p.pid)
        if args.cpus:
            set_affinity(h, args.cpus)
        while p.poll() is None:
            m = mem_sample(h) if h else None
            if m:
                samples.append((round(time.time() - t0, 1),) + m + (gpu_mem_mb(),))
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
        code = p.returncode
    with open(csv_path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["t_s", "commit_bytes", "workingset_bytes", "peak_pagefile_bytes", "gpu_mem_mb"])
        cw.writerows(samples)
    res = parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
    mb = lambda b: b / 2**20
    at10 = [s[1] for s in samples if s[0] >= 10]
    gpu = [s[4] for s in samples if s[4] is not None]
    res.update(label=args.label or "-", cpus=args.cpus, exit=code, seconds=args.seconds,
               mean_fps=res["frames"] / args.seconds if res["frames"] is not None else None,
               commit_10s_mb=mb(at10[0]) if at10 else None,
               commit_max_mb=mb(max(s[1] for s in samples)) if samples else None,
               ws_max_mb=mb(max(s[2] for s in samples)) if samples else None,
               peak_pagefile_mb=mb(max(s[3] for s in samples)) if samples else None,
               gpu_max_mb=max(gpu) if gpu else None, log=str(log_path), mem_csv=str(csv_path))
    return res


def table(r):
    b = r["blocked"]
    rows = [
        ("label", r["label"]), ("cpus", r["cpus"] or "all"), ("exit code", r["exit"]),
        ("frames presented", fmt(r["frames"], "%d")),
        ("errors/shader fail/draws skipped", "%s/%s/%s" % (fmt(r["errors"], "%d"), fmt(r["shader_failures"], "%d"), fmt(r["draws_skipped"], "%d"))),
        ("mean fps (frames/seconds)", fmt(r["mean_fps"])), ("steady fps (median window)", fmt(r["steady_fps"])),
        ("worst window max frame ms", fmt(r["worst_ms"])), ("1% low ms (p99 of window max)", fmt(r["low1_ms"])),
        ("render blocked % (median)", "-" if b is None else " ".join("%s %d" % (n, v) for n, v in zip(BLOCK_NAMES, b))),
        ("gpu avg ms", fmt(r["gpu_ms"], "%.2f")), ("replay avg ms", fmt(r["replay_ms"], "%.2f")),
        ("draw profile ns/draw", r["profile"] or "-"),
        ("svc blocked / romfs slow reads", "%d / %d" % (r["svc_blocked"], r["romfs_slow"])),
        ("commit MB at 10 s / max", "%s / %s" % (fmt(r["commit_10s_mb"], "%.0f"), fmt(r["commit_max_mb"], "%.0f"))),
        ("working set max MB", fmt(r["ws_max_mb"], "%.0f")), ("peak pagefile MB", fmt(r["peak_pagefile_mb"], "%.0f")),
        ("gpu mem max MB", fmt(r["gpu_max_mb"], "%.0f")),
    ]
    print("lowend bench (frame figures are per 60-frame window, not per frame)")
    for k, v in rows:
        print("%-34s %s" % (k, v))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--script")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--cpus", type=int)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="build/lowend-bench")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = run(args)
    table(res)
    if args.json:
        print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
