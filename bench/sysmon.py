#!/usr/bin/env python3
"""Jetson system sampler (run as root inside the container; power rails need root).

  sysmon.py record <out.csv> [interval_s]   # sample until SIGINT/SIGTERM
  sysmon.py summarize <out.csv> [fps]       # print JSON summary (energy/frame if fps given)

Columns: t, vdd_in_mw, vdd_cpu_gpu_cv_mw, vdd_soc_mw, gpu_load_pct, ram_used_mb
"""
import glob
import json
import os
import signal
import sys
import time

GPU_LOAD = "/sys/devices/platform/17000000.ga10b/load"  # 0..1000


def rails():
    out = {}
    for lab in glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_label"):
        d, n = os.path.dirname(lab), os.path.basename(lab)[2:-6]
        try:
            name = open(lab).read().strip()
            mv = int(open(f"{d}/in{n}_input").read())
            ma = int(open(f"{d}/curr{n}_input").read())
            out[name] = (d, n)
        except (OSError, ValueError):
            pass
    return out


def read_power(r, name):
    if name not in r:
        return ""
    d, n = r[name]
    mv = int(open(f"{d}/in{n}_input").read())
    ma = int(open(f"{d}/curr{n}_input").read())
    return mv * ma / 1000.0  # mW


def ram_used_mb():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        m[k] = int(v.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) / 1024.0


def record(path, interval):
    r = rails()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    with open(path, "w") as f:
        f.write("t,vdd_in_mw,vdd_cpu_gpu_cv_mw,vdd_soc_mw,gpu_load_pct,ram_used_mb\n")
        while not stop:
            try:
                gpu = int(open(GPU_LOAD).read()) / 10.0
            except OSError:
                gpu = ""
            f.write(f"{time.time():.3f},{read_power(r, 'VDD_IN')},"
                    f"{read_power(r, 'VDD_CPU_GPU_CV')},{read_power(r, 'VDD_SOC')},"
                    f"{gpu},{ram_used_mb():.0f}\n")
            f.flush()
            time.sleep(interval)


def summarize(path, fps=None, t0=None, t1=None):
    import csv
    rows = list(csv.DictReader(open(path)))
    if t0 is not None:
        rows = [x for x in rows if t0 <= float(x["t"]) <= t1]

    def col(c):
        return [float(x[c]) for x in rows if x[c] != ""]

    def mean(v):
        return sum(v) / len(v) if v else None

    s = {
        "samples": len(rows),
        "power_w_mean": mean(col("vdd_in_mw")) and mean(col("vdd_in_mw")) / 1000,
        "power_w_peak": col("vdd_in_mw") and max(col("vdd_in_mw")) / 1000,
        "power_cpu_gpu_cv_w_mean": mean(col("vdd_cpu_gpu_cv_mw")) and mean(col("vdd_cpu_gpu_cv_mw")) / 1000,
        "gpu_load_pct_mean": mean(col("gpu_load_pct")),
        "ram_used_mb_peak": col("ram_used_mb") and max(col("ram_used_mb")),
    }
    if fps and s["power_w_mean"]:
        s["energy_j_per_frame"] = s["power_w_mean"] / fps
    return s


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.2)
    else:
        fps = float(sys.argv[3]) if len(sys.argv) > 3 else None
        print(json.dumps(summarize(sys.argv[2], fps), indent=2))
