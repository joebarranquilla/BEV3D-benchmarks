#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


NAME_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

CAMERA_ALIAS_MAP = {
    "CAM_FRONT_LEFT": "CAMERA_LEFT_FRONT",
    "CAM_FRONT": "CAMERA_LEFT_FRONT",
    "CAM_FRONT_RIGHT": "CAMERA_RIGHT_FRONT",
    "CAM_BACK_LEFT": "CAMERA_LEFT_BACK",
    "CAM_BACK": "CAMERA_LEFT_BACK",
    "CAM_BACK_RIGHT": "CAMERA_RIGHT_BACK",
}

# Pick one top lidar channel as reference frame for BEV-style models.
TRUCK_REF_LIDAR = "LIDAR_TOP_FRONT"

# Reuse official mini_val names so nuScenes eval split filtering includes all scenes.
MINI_VAL_SCENE_NAMES = ["scene-0103", "scene-0916"]

NU_DET_ATTRIBUTE_NAMES = {
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
}


def stable_token(seed: str) -> str:
    return hashlib.md5(seed.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_metadata_version(root: Path, src_version: str, dst_version: str, ref_lidar_channel: str) -> Path:
    src_dir = root / src_version
    dst_dir = root / dst_version
    if not src_dir.exists():
        raise FileNotFoundError(f"Source metadata directory not found: {src_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    src_attributes = load_json(src_dir / "attribute.json")
    valid_attr_tokens = {
        row["token"]
        for row in src_attributes
        if row.get("name") in NU_DET_ATTRIBUTE_NAMES
    }

    for src_file in sorted(src_dir.glob("*.json")):
        payload = load_json(src_file)
        if src_file.name == "scene.json":
            log_token = stable_token(f"{dst_version}:truckscenes:log")
            for i, scene in enumerate(payload):
                scene["log_token"] = log_token
                scene["name"] = MINI_VAL_SCENE_NAMES[i % len(MINI_VAL_SCENE_NAMES)]
            dump_json(dst_dir / src_file.name, payload)
            continue

        if src_file.name == "sample_annotation.json":
            for row in payload:
                tokens = row.get("attribute_tokens", [])
                row["attribute_tokens"] = [t for t in tokens if t in valid_attr_tokens]
            dump_json(dst_dir / src_file.name, payload)
            continue

        if src_file.name == "sensor.json":
            for row in payload:
                if row.get("channel") == ref_lidar_channel:
                    row["channel"] = "LIDAR_TOP"
            dump_json(dst_dir / src_file.name, payload)
            continue

        dump_json(dst_dir / src_file.name, payload)

    log_token = stable_token(f"{dst_version}:truckscenes:log")
    map_token = stable_token(f"{dst_version}:truckscenes:map")

    log_json = [
        {
            "token": log_token,
            "logfile": "",
            "vehicle": "truckscenes-man",
            "date_captured": "",
            "location": "truckscenes",
        }
    ]
    map_json = [
        {
            "token": map_token,
            "category": "semantic_prior",
            "filename": "",
            "log_tokens": [log_token],
        }
    ]

    dump_json(dst_dir / "log.json", log_json)
    dump_json(dst_dir / "map.json", map_json)
    return dst_dir


def obtain_sensor2top(
    nusc: NuScenes,
    sensor_token: str,
    l2e_t: np.ndarray,
    l2e_r_mat: np.ndarray,
    e2g_t: np.ndarray,
    e2g_r_mat: np.ndarray,
    sensor_type: str,
) -> dict[str, Any]:
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))

    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }

    l2e_r_s = Quaternion(cs_record["rotation"]).rotation_matrix
    e2g_r_s = Quaternion(pose_record["rotation"]).rotation_matrix

    r = (l2e_r_s.T @ e2g_r_s.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    t = (cs_record["translation"] @ e2g_r_s.T + pose_record["translation"]) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    t -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T

    sweep["sensor2lidar_rotation"] = r.T
    sweep["sensor2lidar_translation"] = t
    return sweep


def build_bevformer_infos(
    nusc: NuScenes,
    camera_alias_map: dict[str, str],
    ref_lidar_channel: str,
    max_sweeps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    val_infos: list[dict[str, Any]] = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        if ref_lidar_channel not in sample["data"]:
            continue

        lidar_token = sample["data"][ref_lidar_channel]
        sd_rec = nusc.get("sample_data", lidar_token)
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        info: dict[str, Any] = {
            "lidar_path": lidar_path,
            "token": sample["token"],
            "prev": sample["prev"],
            "next": sample["next"],
            "can_bus": np.zeros(18, dtype=np.float64),
            "frame_idx": 0,
            "sweeps": [],
            "cams": {},
            "scene_token": sample["scene_token"],
            "lidar2ego_translation": cs_record["translation"],
            "lidar2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sample["timestamp"],
        }

        l2e_t = np.array(info["lidar2ego_translation"])
        l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        e2g_t = np.array(info["ego2global_translation"])
        e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix

        for cam_alias, truck_cam in camera_alias_map.items():
            if truck_cam not in sample["data"]:
                raise KeyError(f"Camera channel '{truck_cam}' not found for sample {sample['token']}")
            cam_token = sample["data"][truck_cam]
            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(
                nusc,
                cam_token,
                l2e_t,
                l2e_r_mat,
                e2g_t,
                e2g_r_mat,
                cam_alias,
            )
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info["cams"][cam_alias] = cam_info

        sweeps = []
        sweep_rec = sd_rec
        while len(sweeps) < max_sweeps and sweep_rec["prev"]:
            sweep = obtain_sensor2top(
                nusc,
                sweep_rec["prev"],
                l2e_t,
                l2e_r_mat,
                e2g_t,
                e2g_r_mat,
                "lidar",
            )
            sweeps.append(sweep)
            sweep_rec = nusc.get("sample_data", sweep_rec["prev"])
        info["sweeps"] = sweeps

        annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
        velocity = np.array([nusc.box_velocity(token)[:2] for token in sample["anns"]])
        valid_flag = np.array(
            [(anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0 for anno in annotations],
            dtype=bool,
        ).reshape(-1)

        for i in range(len(boxes)):
            velo = np.array([*velocity[i], 0.0])
            velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
            velocity[i] = velo[:2]

        names = [b.name for b in boxes]
        names = [NAME_MAPPING.get(name, name) for name in names]
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)

        info["gt_boxes"] = gt_boxes
        info["gt_names"] = np.array(names)
        info["gt_velocity"] = velocity.reshape(-1, 2)
        info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
        info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
        info["valid_flag"] = valid_flag

        val_infos.append(info)

    return [], val_infos


def build_bevdepth_infos(
    nusc: NuScenes,
    camera_alias_map: dict[str, str],
    ref_lidar_channel: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    val_infos: list[dict[str, Any]] = []

    for scene in nusc.scene:
        first_sample_token = scene["first_sample_token"]
        cur_sample = nusc.get("sample", first_sample_token)

        while True:
            info: dict[str, Any] = {
                "sample_token": cur_sample["token"],
                "timestamp": cur_sample["timestamp"],
                "scene_token": cur_sample["scene_token"],
                "cam_infos": {},
                "lidar_infos": {},
                "cam_sweeps": [],
                "lidar_sweeps": [],
            }

            for cam_alias, truck_cam in camera_alias_map.items():
                if truck_cam not in cur_sample["data"]:
                    raise KeyError(f"Camera channel '{truck_cam}' not found for sample {cur_sample['token']}")
                cam_data = nusc.get("sample_data", cur_sample["data"][truck_cam])
                info["cam_infos"][cam_alias] = {
                    "sample_token": cam_data["sample_token"],
                    "ego_pose": nusc.get("ego_pose", cam_data["ego_pose_token"]),
                    "timestamp": cam_data["timestamp"],
                    "is_key_frame": cam_data["is_key_frame"],
                    "height": cam_data["height"],
                    "width": cam_data["width"],
                    "filename": cam_data["filename"],
                    "calibrated_sensor": nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"]),
                }

            if ref_lidar_channel not in cur_sample["data"]:
                raise KeyError(f"Reference lidar '{ref_lidar_channel}' missing for sample {cur_sample['token']}")
            lidar_data = nusc.get("sample_data", cur_sample["data"][ref_lidar_channel])
            info["lidar_infos"]["LIDAR_TOP"] = {
                "sample_token": lidar_data["sample_token"],
                "ego_pose": nusc.get("ego_pose", lidar_data["ego_pose_token"]),
                "timestamp": lidar_data["timestamp"],
                "filename": lidar_data["filename"],
                "calibrated_sensor": nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"]),
            }

            ann_infos = []
            for ann_token in cur_sample["anns"]:
                ann_info = nusc.get("sample_annotation", ann_token)
                velocity = nusc.box_velocity(ann_info["token"])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info["velocity"] = velocity
                ann_infos.append(ann_info)
            info["ann_infos"] = ann_infos

            val_infos.append(info)

            if not cur_sample["next"]:
                break
            cur_sample = nusc.get("sample", cur_sample["next"])

    return [], val_infos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TruckScenes metadata and info PKLs for BEVDepth/BEVFormer")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/truckscenes/man-truckscenes"))
    parser.add_argument("--source-version", type=str, default="v1.2-mini")
    parser.add_argument("--target-version", type=str, default="v1.0-mini")
    parser.add_argument("--ref-lidar", type=str, default=TRUCK_REF_LIDAR)
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()

    normalized_dir = normalize_metadata_version(root, args.source_version, args.target_version, args.ref_lidar)
    if args.verbose:
        print(f"Normalized metadata written to {normalized_dir}")

    nusc = NuScenes(version=args.target_version, dataroot=str(root), verbose=args.verbose)
    normalized_ref_lidar = "LIDAR_TOP"

    bevdepth_train, bevdepth_val = build_bevdepth_infos(nusc, CAMERA_ALIAS_MAP, normalized_ref_lidar)
    mmcv.dump(bevdepth_train, str(root / "nuscenes_infos_train.pkl"))
    mmcv.dump(bevdepth_val, str(root / "nuscenes_infos_val.pkl"))

    bevformer_train, bevformer_val = build_bevformer_infos(
        nusc,
        CAMERA_ALIAS_MAP,
        normalized_ref_lidar,
        args.max_sweeps,
    )
    metadata = {"version": args.target_version}
    mmcv.dump({"infos": bevformer_train, "metadata": metadata}, str(root / "nuscenes_infos_temporal_train.pkl"))
    mmcv.dump({"infos": bevformer_val, "metadata": metadata}, str(root / "nuscenes_infos_temporal_val.pkl"))

    print(f"Wrote {len(bevdepth_train)} BEVDepth train infos and {len(bevdepth_val)} BEVDepth val infos")
    print(f"Wrote {len(bevformer_train)} BEVFormer temporal train infos and {len(bevformer_val)} BEVFormer temporal val infos")


if __name__ == "__main__":
    main()
