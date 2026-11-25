#!/usr/bin/env python3
"""Offline nuScenes benchmark for CUDA-BEVFusion (libpybev): latency + mAP/NDS.

Replicates the mmdet3d BEVFusion test pipeline in plain numpy:
  - per-sample camera2lidar / intrinsics / lidar2image / img_aug_matrix
  - 1 keyframe + 9 sweeps, 5th channel = time lag (s), remove_close on sweeps
  - output boxes -> nuScenes global frame (same as output_to_nusc_box +
    lidar_nusc_box_to_global in bevfusion/mmdet3d/datasets/nuscenes_dataset.py)

Run inside the container after `source env.sh`:
  python3 bench/nusc_bench.py --model resnet50 --precision fp16 --split mini_val
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from pyquaternion import Quaternion

REPO_ROOT = os.environ.get(
    "CUDA_BEVFUSION_ROOT", "/workspace/Lidar_AI_Solution/CUDA-BEVFusion")
sys.path.insert(0, os.path.join(REPO_ROOT, "build"))

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer",
           "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
DEFAULT_ATTR = {
    "car": "vehicle.parked", "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked", "truck": "vehicle.parked",
    "bus": "vehicle.moving", "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider", "barrier": "", "traffic_cone": "",
}
SWEEPS = 9
# Test-time ImageAug3D: resize 0.48 (1600x900 -> 768x432), crop to 704x256.
IMG_AUG = np.array([[0.48, 0, 0, -32.0],
                    [0, 0.48, 0, -176.0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float32)


def tf(rotation, translation):
    m = np.eye(4)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = translation
    return m


def sensor2global(nusc, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return tf(pose["rotation"], pose["translation"]) @ tf(cs["rotation"], cs["translation"])


def build_inputs(nusc, sample):
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar2global = sensor2global(nusc, lidar_sd)
    global2lidar = np.linalg.inv(lidar2global)

    images, cam2lidar, intr, lidar2img = [], [], [], []
    for cam in CAMS:
        sd = nusc.get("sample_data", sample["data"][cam])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        c2l = global2lidar @ sensor2global(nusc, sd)
        k = np.eye(4)
        k[:3, :3] = np.array(cs["camera_intrinsic"])
        cam2lidar.append(c2l)
        intr.append(k)
        lidar2img.append(k @ np.linalg.inv(c2l))
        img = cv2.imread(os.path.join(nusc.dataroot, sd["filename"]))
        images.append(img[..., ::-1])  # BGR -> RGB, as tool/pybev.py does

    # Points: keyframe (time 0) + up to 9 previous sweeps, in keyframe lidar frame.
    def load_bin(sd):
        return np.fromfile(os.path.join(nusc.dataroot, sd["filename"]),
                           dtype=np.float32).reshape(-1, 5).copy()

    ts0 = lidar_sd["timestamp"] / 1e6
    pts = load_bin(lidar_sd)
    pts[:, 4] = 0.0
    clouds = [pts]
    sd = lidar_sd
    for _ in range(SWEEPS):
        if not sd["prev"]:
            break
        sd = nusc.get("sample_data", sd["prev"])
        sp = load_bin(sd)
        close = (np.abs(sp[:, 0]) < 1.0) & (np.abs(sp[:, 1]) < 1.0)
        sp = sp[~close]
        s2l = global2lidar @ sensor2global(nusc, sd)
        sp[:, :3] = sp[:, :3] @ s2l[:3, :3].T + s2l[:3, 3]
        sp[:, 4] = ts0 - sd["timestamp"] / 1e6
        clouds.append(sp)
    if len(clouds) == 1:  # pad_empty_sweeps
        clouds += [pts] * SWEEPS
    points = np.concatenate(clouds, 0)
    keep = ((np.abs(points[:, 0]) < 54) & (np.abs(points[:, 1]) < 54)
            & (points[:, 2] > -5) & (points[:, 2] < 3))
    points = np.ascontiguousarray(points[keep].astype(np.float16))

    f32 = lambda xs: np.ascontiguousarray(np.stack(xs)[None].astype(np.float32))
    calib = (f32(cam2lidar), f32(intr), f32(lidar2img),
             np.ascontiguousarray(np.tile(IMG_AUG, (1, 6, 1, 1))))
    imgs = np.ascontiguousarray(np.stack(images)[None])
    return imgs, points, calib, lidar_sd


def to_global(nusc, boxes, lidar_sd, sample_token, eval_cfg):
    """[x,y,z(bottom),dx,dy,dz,yaw,vx,vy,label,score] (lidar) -> nuScenes dicts."""
    from nuscenes.utils.data_classes import Box
    cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    out = []
    for b in boxes:
        label = int(b[9])
        if not 0 <= label < len(CLASSES) or not np.all(np.isfinite(b)):
            continue
        name = CLASSES[label]
        center = [b[0], b[1], b[2] + b[5] / 2]
        box = Box(center, [b[3], b[4], b[5]],
                  Quaternion(axis=[0, 0, 1], radians=-b[6] - np.pi / 2),
                  velocity=(b[7], b[8], 0.0))
        box.rotate(Quaternion(cs["rotation"]))
        box.translate(np.array(cs["translation"]))
        if np.linalg.norm(box.center[:2]) > eval_cfg.class_range[name]:
            continue
        box.rotate(Quaternion(pose["rotation"]))
        box.translate(np.array(pose["translation"]))
        if np.hypot(b[7], b[8]) > 0.2:
            if name in ("car", "construction_vehicle", "bus", "truck", "trailer"):
                attr = "vehicle.moving"
            elif name in ("bicycle", "motorcycle"):
                attr = "cycle.with_rider"
            else:
                attr = DEFAULT_ATTR[name]
        else:
            attr = {"pedestrian": "pedestrian.standing",
                    "bus": "vehicle.stopped"}.get(name, DEFAULT_ATTR[name])
        out.append({
            "sample_token": sample_token,
            "translation": box.center.tolist(),
            "size": box.wlh.tolist(),
            "rotation": box.orientation.elements.tolist(),
            "velocity": box.velocity[:2].tolist(),
            "detection_name": name,
            "detection_score": float(b[10]),
            "attribute_name": attr,
        })
    out.sort(key=lambda d: -d["detection_score"])
    return out[:500]


def pct(xs, q):
    return float(np.percentile(np.asarray(xs), q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50int8",
                    help="model/<name> dir in CUDA-BEVFusion")
    ap.add_argument("--precision", default="int8", choices=["fp16", "int8"],
                    help="sparse-conv lidar backbone precision")
    ap.add_argument("--dataroot", default="/workspace/data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--split", default="mini_val",
                    choices=["mini_val", "mini_train"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed passes over the split (accuracy from pass 1)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N samples")
    ap.add_argument("--out", default="/workspace/bench/results")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    import libpybev
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.utils.splits import create_splits_scenes

    tag = f"{args.model}_{args.precision}_{args.split}"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    scenes = set(create_splits_scenes()[args.split])
    samples = [s for s in nusc.sample
               if nusc.get("scene", s["scene_token"])["name"] in scenes]
    samples.sort(key=lambda s: (s["scene_token"], s["timestamp"]))
    if args.limit:
        samples = samples[:args.limit]
    eval_cfg = config_factory("detection_cvpr_2019")

    mdir = os.path.join(REPO_ROOT, "model", args.model)
    core = libpybev.load_bevfusion(
        os.path.join(mdir, "build/camera.backbone.plan"),
        os.path.join(mdir, "build/camera.vtransform.plan"),
        os.path.join(mdir, "lidar.backbone.xyz.onnx"),
        os.path.join(mdir, "build/fuser.plan"),
        os.path.join(mdir, "build/head.bbox.plan"),
        args.precision)
    if core is None:
        sys.exit("libpybev.load_bevfusion failed")

    # Inputs are built per frame (CPU, untimed) to keep RAM flat on 8 GB boards.
    print(f"[{tag}] {len(samples)} samples, warmup {args.warmup}", flush=True)
    imgs, pts, calib, _ = build_inputs(nusc, samples[0])
    core.update(*calib)
    for _ in range(args.warmup):
        core.forward(imgs, pts, with_normalization=True)

    lat, results, npts = [], {}, []
    for r in range(args.repeats):
        for s in samples:
            imgs, pts, calib, lidar_sd = build_inputs(nusc, s)
            npts.append(pts.shape[0])
            core.update(*calib)  # calibration upload, excluded from latency
            t = time.perf_counter()
            boxes = core.forward(imgs, pts, with_normalization=True)
            lat.append((time.perf_counter() - t) * 1000)
            if r == 0:
                results[s["token"]] = to_global(nusc, boxes, lidar_sd,
                                                s["token"], eval_cfg)
        print(f"[{tag}] pass {r + 1}/{args.repeats} done", flush=True)

    summary = {
        "model": args.model, "lidar_precision": args.precision,
        "split": args.split, "num_samples": len(samples),
        "repeats": args.repeats, "warmup": args.warmup,
        "latency_ms": {"mean": float(np.mean(lat)), "p50": pct(lat, 50),
                       "p90": pct(lat, 90), "p99": pct(lat, 99),
                       "min": float(np.min(lat))},
        "fps_p50": 1000.0 / pct(lat, 50),
        "avg_points": float(np.mean(npts)),
    }
    res_path = os.path.join(out_dir, "results_nusc.json")
    with open(res_path, "w") as f:
        json.dump({"meta": {"use_camera": True, "use_lidar": True,
                            "use_radar": False, "use_map": False,
                            "use_external": False},
                   "results": results}, f)

    if not args.no_eval and not args.limit:
        from nuscenes.eval.detection.evaluate import DetectionEval
        ev = DetectionEval(nusc, eval_cfg, res_path, eval_set=args.split,
                           output_dir=out_dir, verbose=False)
        m = ev.main(render_curves=False)
        summary.update({
            "mAP": m["mean_ap"], "NDS": m["nd_score"],
            "tp_errors": m["tp_errors"],
            "per_class_AP": m["mean_dist_aps"],
        })

    np.save(os.path.join(out_dir, "latency_ms.npy"), np.asarray(lat))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("per_class_AP", "tp_errors")}, indent=2))


if __name__ == "__main__":
    main()
