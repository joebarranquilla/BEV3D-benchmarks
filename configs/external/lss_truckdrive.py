"""TruckDrive single-scene dataset adapter for LSS (cab-mounted cameras, ego frame = annotation/velodyne frame)."""
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

DEFAULT_CAMS = [
    "forward_left_wide",
    "forward_center_medium",
    "forward_right_wide",
    "rearward_left_bottom_medium",
    "rearward_right_bottom_medium",
]
H, W = 2168, 3848
# Virtual pinhole focal for equidistant cameras; ~65 deg HFOV to match what LSS saw on nuScenes.
FISHEYE_VIRTUAL_F = 3000.0


def _extract(scene_dir: Path) -> Path:
    out = scene_dir / "extracted"
    for name in ("calibrations", "annotations"):
        if not (out / name).exists():
            with zipfile.ZipFile(scene_dir / f"{name}.zip") as z:
                z.extractall(out / name)
    return out


def _tf_matrix(tf: dict) -> np.ndarray:
    t, r = tf["transform"]["translation"], tf["transform"]["rotation"]
    m = np.eye(4)
    m[:3, :3] = Quaternion(r["w"], r["x"], r["y"], r["z"]).rotation_matrix
    m[:3, 3] = [t["x"], t["y"], t["z"]]
    return m


def _ts(path: Path) -> float:
    return int(path.stem.split("_")[1]) / 1e9


class TruckDriveData(torch.utils.data.Dataset):
    def __init__(self, scene_dir, data_aug_conf, grid_conf, cams=None, max_samples=0):
        from src.tools import gen_dx_bx, img_transform, normalize_img

        self._img_transform, self._normalize = img_transform, normalize_img
        self.scene_dir = Path(scene_dir)
        self.cams = list(cams or DEFAULT_CAMS)
        self.data_aug_conf = data_aug_conf
        dx, bx, nx = gen_dx_bx(grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        ex = _extract(self.scene_dir)
        tree = json.load(open(ex / "calibrations" / "calib_tf_tree_full.json"))
        cab_from_vel = _tf_matrix(tree["cab_velodyne"])
        vel_from_cab = np.linalg.inv(cab_from_vel)

        self.cam_info = {}
        self.cam_files = {}
        for cam in self.cams:
            calib = json.load(open(ex / "calibrations" / f"calib_camera_leopard_{cam}.json"))
            K = np.array(calib["K"]).reshape(3, 3)
            fisheye = calib["distortion_model"] == "equidistant"
            vel_from_cam = vel_from_cab @ _tf_matrix(tree[f"cab_camera_leopard_{cam}"])
            map1 = map2 = None
            if fisheye:
                K_new = K.copy()
                K_new[0, 0] = K_new[1, 1] = FISHEYE_VIRTUAL_F
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, np.zeros(4), np.eye(3), K_new, (W, H), cv2.CV_32FC1)
                K = K_new
            self.cam_info[cam] = dict(K=K, vel_from_cam=vel_from_cam, map1=map1, map2=map2)
            files = sorted((self.scene_dir / "camera" / "leopard" / cam / "images").glob("*.jpg"), key=_ts)
            self.cam_files[cam] = (np.array([_ts(f) for f in files]), files)

        ann_files = sorted((ex / "annotations" / "bounding_boxes").glob("*.json"), key=_ts)
        self.ann_files = ann_files[:max_samples] if max_samples else ann_files
        print(f"TruckDriveData: {len(self)} samples, cams={self.cams}")

    def __len__(self):
        return len(self.ann_files)

    def _sample_augmentation(self):
        fH, fW = self.data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * resize_dims[1]) - fH
        crop_w = int(max(0, resize_dims[0] - fW) / 2)
        return resize, resize_dims, (crop_w, crop_h, crop_w + fW, crop_h + fH)

    def _load_image(self, cam, ts):
        stamps, files = self.cam_files[cam]
        img = cv2.imread(str(files[int(np.abs(stamps - ts).argmin())]))
        info = self.cam_info[cam]
        if info["map1"] is not None:
            img = cv2.remap(img, info["map1"], info["map2"], cv2.INTER_LINEAR)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _binimg(self, boxes):
        img = np.zeros((self.nx[0], self.nx[1]))
        for b in boxes:
            cls = b["class-id"]
            if not cls.startswith("Vehicle") or cls.startswith("Vehicle-EgoVehicle"):
                continue
            g = lambda k: b.get(f"{k}_ego", b[k])  # some boxes lack *_ego; plain keys are in the same frame
            box = Box([g("x"), g("y"), g("z")], [g("w"), g("l"), g("h")],
                      Quaternion(axis=[0, 0, 1], angle=g("yaw")))
            pts = box.bottom_corners()[:2].T
            pts = np.round((pts - self.bx[:2] + self.dx[:2] / 2.0) / self.dx[:2]).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)
        return torch.Tensor(img).unsqueeze(0)

    def __getitem__(self, index):
        ann_file = self.ann_files[index]
        ts = _ts(ann_file)
        resize, resize_dims, crop = self._sample_augmentation()
        imgs, rots, trans, intrins, post_rots, post_trans = [], [], [], [], [], []
        for cam in self.cams:
            info = self.cam_info[cam]
            img, post_rot2, post_tran2 = self._img_transform(
                self._load_image(cam, ts), torch.eye(2), torch.zeros(2),
                resize=resize, resize_dims=resize_dims, crop=crop, flip=False, rotate=0)
            post_rot, post_tran = torch.eye(3), torch.zeros(3)
            post_rot[:2, :2], post_tran[:2] = post_rot2, post_tran2
            imgs.append(self._normalize(img))
            intrins.append(torch.Tensor(info["K"]))
            rots.append(torch.Tensor(info["vel_from_cam"][:3, :3]))
            trans.append(torch.Tensor(info["vel_from_cam"][:3, 3]))
            post_rots.append(post_rot)
            post_trans.append(post_tran)
        binimg = self._binimg(json.load(open(ann_file)))
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans), torch.stack(intrins),
                torch.stack(post_rots), torch.stack(post_trans), binimg)
