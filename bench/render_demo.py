#!/usr/bin/env python3
"""Render a demo video from a ROS run's detections.jsonl (logged by detection_logger_node).

Left: 6 cameras with projected 3D boxes. Right: BEV LiDAR with boxes.
Bottom: metrics measured in that same run (summary.json / system.json).

  python3 bench/render_demo.py <run_dir> --scene scene-0103 --out demo.mp4
"""
import argparse
import json
import os

import cv2
import numpy as np
from pyquaternion import Quaternion

CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]  # 2x3 mosaic order
COLORS = {  # BGR
    "car": (0, 165, 255), "truck": (0, 215, 255), "bus": (0, 255, 255),
    "trailer": (0, 128, 255), "construction_vehicle": (30, 105, 210),
    "pedestrian": (255, 0, 0), "motorcycle": (255, 0, 255), "bicycle": (180, 0, 255),
    "traffic_cone": (0, 0, 255), "barrier": (128, 128, 128),
}
TILE_W, TILE_H = 480, 270
BEV_PX, BEV_RANGE = 540, 50.0  # pixels, metres (half-width)


def ros_det_to_corners(det):
    """ROS Detection3D (as logged) -> 8 corners in lidar frame.

    bev_node publishes: position = raw (x, y, z_bottom), size = (det[4], det[3], det[5]),
    yaw = raw CUDA-BEVFusion yaw. Same convention as bench/nusc_bench.py:to_global.
    """
    x, y, z = det["position"]
    sx, sy, sz = det["size"]
    q = det["orientation"]
    yaw = 2 * np.arctan2(q["z"], q["w"])
    w, l, h = sy, sx, sz  # nuScenes Box(wlh=[det3, det4, det5])
    rot = Quaternion(axis=[0, 0, 1], radians=-yaw - np.pi / 2).rotation_matrix
    xs = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    ys = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    zs = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return (rot @ np.vstack([xs, ys, zs])).T + np.array([x, y, z + h / 2])


def tf(rotation, translation):
    m = np.eye(4)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = translation
    return m


def s2g(nusc, sd):
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return tf(pose["rotation"], pose["translation"]) @ tf(cs["rotation"], cs["translation"])


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_cam(nusc, sample, cam, l2g, boxes):
    sd = nusc.get("sample_data", sample["data"][cam])
    img = cv2.imread(os.path.join(nusc.dataroot, sd["filename"]))
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    lidar2cam = np.linalg.inv(s2g(nusc, sd)) @ l2g
    for corners, name, _ in boxes:
        pc = (lidar2cam[:3, :3] @ corners.T).T + lidar2cam[:3, 3]
        if np.any(pc[:, 2] < 0.5):
            continue
        uv = (K @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        if (uv[:, 0].max() < 0 or uv[:, 0].min() > img.shape[1]
                or uv[:, 1].max() < 0 or uv[:, 1].min() > img.shape[0]):
            continue
        for a, b in EDGES:
            cv2.line(img, tuple(map(int, uv[a])), tuple(map(int, uv[b])),
                     COLORS.get(name, (255, 255, 255)), 3, cv2.LINE_AA)
    img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    cv2.putText(img, cam.replace("CAM_", ""), (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def draw_bev(nusc, sample, boxes):
    bev = np.zeros((BEV_PX, BEV_PX, 3), np.uint8)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pts = np.fromfile(os.path.join(nusc.dataroot, sd["filename"]), np.float32).reshape(-1, 5)
    s = BEV_PX / (2 * BEV_RANGE)

    def to_px(x, y):  # nuScenes LIDAR_TOP: x right, y forward -> forward is up
        return (BEV_PX / 2 + x * s), (BEV_PX / 2 - y * s)

    u, v = to_px(pts[:, 0], pts[:, 1])
    m = (u >= 0) & (u < BEV_PX) & (v >= 0) & (v < BEV_PX)
    shade = np.clip(80 + pts[m, 2] * 40, 60, 255).astype(np.uint8)
    bev[v[m].astype(int), u[m].astype(int)] = np.stack([shade] * 3, 1)
    for r in (10, 20, 30, 40, 50):
        cv2.circle(bev, (BEV_PX // 2, BEV_PX // 2), int(r * s), (50, 50, 50), 1)
    for corners, name, _ in boxes:
        fp = np.array([to_px(c[0], c[1]) for c in corners[[0, 1, 5, 4]]], np.int32)
        cv2.polylines(bev, [fp], True, COLORS.get(name, (255, 255, 255)), 2, cv2.LINE_AA)
    cv2.drawMarker(bev, (BEV_PX // 2, BEV_PX // 2), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 12, 2)
    cv2.putText(bev, "LIDAR BEV (50 m)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return bev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--scene", default="scene-0103")
    ap.add_argument("--dataroot", default="/workspace/data/nuscenes")
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--fps", type=float, default=2.0, help="video playback rate")
    ap.add_argument("--title", default="CUDA-BEVFusion (TensorRT INT8) in ROS 2 | Jetson Orin Nano 8GB")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from nuscenes import NuScenes
    nusc = NuScenes("v1.0-mini", args.dataroot, verbose=False)
    scene = next(s for s in nusc.scene if s["name"] == args.scene)

    dets = {}
    for line in open(os.path.join(args.run_dir, "detections.jsonl")):
        r = json.loads(line)
        if r.get("sample_token"):
            dets[r["sample_token"]] = r
    summ = json.load(open(os.path.join(args.run_dir, "summary.json")))
    sysm = {}
    if os.path.exists(os.path.join(args.run_dir, "system.json")):
        sysm = json.load(open(os.path.join(args.run_dir, "system.json")))
    st = summ["stages"]
    bar_txt = (f"model {st['inference_ms']['p50']:.0f} ms | end-to-end {summ['e2e_ms']['p50']:.0f} ms p50"
               f" | input {summ['offered_fps']:.0f} Hz, drops {summ['drop_rate'] * 100:.0f}%")
    if sysm.get("power_w_mean"):
        bar_txt += f" | {sysm['power_w_mean']:.1f} W | RAM {sysm['ram_used_mb_peak'] / 1024:.1f} GB"

    out = args.out or os.path.join(args.run_dir, f"demo_{args.scene}.mp4")
    W, H = TILE_W * 3 + BEV_PX, max(TILE_H * 2, BEV_PX) + 70
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    token, n = scene["first_sample_token"], 0
    while token:
        sample = nusc.get("sample", token)
        token = sample["next"]
        rec = dets.get(sample["token"])
        if rec is None:
            continue
        boxes = []
        for d in rec["detections"]:
            c = d["classes"][0] if d["classes"] else {"class_id": "", "score": 0}
            if c["score"] >= args.score:
                boxes.append((ros_det_to_corners(d), c["class_id"], c["score"]))
        l2g = s2g(nusc, nusc.get("sample_data", sample["data"]["LIDAR_TOP"]))
        tiles = [draw_cam(nusc, sample, cam, l2g, boxes) for cam in CAMS]
        mosaic = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:mosaic.shape[0], :mosaic.shape[1]] = mosaic
        frame[:BEV_PX, mosaic.shape[1]:] = draw_bev(nusc, sample, boxes)
        y0 = H - 70
        cv2.putText(frame, args.title, (10, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{bar_txt} | frame e2e {rec['e2e_ms']:.0f} ms | {len(boxes)} objects",
                    (10, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)
        vw.write(frame)
        n += 1
    vw.release()
    print(f"wrote {n} frames -> {out}")


if __name__ == "__main__":
    main()
