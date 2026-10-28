"""LSS vehicle-segmentation inference/eval on nuScenes-mini and TruckScenes-mini (nuScenes format)."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from efficientnet_pytorch import EfficientNet
from nuscenes.nuscenes import NuScenes

# Avoid downloading ImageNet weights; the LSS checkpoint already contains the trunk.
EfficientNet.from_pretrained = classmethod(lambda cls, name, *a, **k: cls.from_name(name))

CAMS = {
    "nuscenes": {
        "cams": ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
        "H": 900,
        "W": 1600,
    },
    "truckscenes": {
        "cams": ["CAMERA_LEFT_FRONT", "CAMERA_RIGHT_FRONT", "CAMERA_LEFT_BACK", "CAMERA_RIGHT_BACK"],
        "H": 943,
        "W": 1980,
    },
    "truckdrive": {
        "cams": [
            "forward_left_wide",
            "forward_center_medium",
            "forward_right_wide",
            "rearward_left_bottom_medium",
            "rearward_right_bottom_medium",
        ],
        "H": 2168,
        "W": 3848,
    },
}


def save_visualizations(model, valdata, device, cams, out_dir, num, denormalize_img):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(valdata) // num)
    model.eval()
    for idx in list(range(0, len(valdata), step))[:num]:
        imgs, rots, trans, intrins, post_rots, post_trans, binimg = valdata[idx]
        with torch.no_grad():
            pred = model(*(t.unsqueeze(0).to(device) for t in (imgs, rots, trans, intrins, post_rots, post_trans)))
        pred = pred.sigmoid().cpu()[0, 0]

        n = len(cams)
        fig = plt.figure(figsize=(4 * n, 9))
        gs = fig.add_gridspec(2, n, height_ratios=(1, 3))
        for i in range(n):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(denormalize_img(imgs[i]))
            ax.set_title(cams[i], fontsize=8)
            ax.axis("off")
        ax = fig.add_subplot(gs[1, : n // 2])
        ax.imshow(pred, vmin=0, vmax=1, cmap="Blues", origin="lower")
        ax.set_title("LSS predicted vehicle segmentation")
        ax = fig.add_subplot(gs[1, n // 2 :])
        ax.imshow(binimg[0], vmin=0, vmax=1, cmap="Greys", origin="lower")
        ax.set_title("Ground truth")
        fig.tight_layout()
        fig.savefig(out_dir / f"lss_{idx:04d}.jpg", dpi=80)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz-dir", default="")
    ap.add_argument("--viz-num", type=int, default=6)
    ap.add_argument("--lss-dir", required=True)
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--dataset", choices=list(CAMS), default="nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--nworkers", type=int, default=4)
    ap.add_argument("--gpuid", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.lss_dir)
    from src.data import SegmentationData
    from src.models import compile_model
    from src.tools import SimpleLoss, denormalize_img, get_val_info

    spec = CAMS[args.dataset]
    grid_conf = {
        "xbound": [-50.0, 50.0, 0.5],
        "ybound": [-50.0, 50.0, 0.5],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [4.0, 45.0, 1.0],
    }
    data_aug_conf = {
        "resize_lim": (0.193, 0.225),
        "final_dim": (128, 352),
        "rot_lim": (-5.4, 5.4),
        "rand_flip": True,
        "bot_pct_lim": (0.0, 0.22),
        "H": spec["H"],
        "W": spec["W"],
        "cams": spec["cams"],
        "Ncams": len(spec["cams"]),
    }

    if args.dataset == "truckdrive":
        from lss_truckdrive import TruckDriveData

        valdata = TruckDriveData(args.dataroot, data_aug_conf, grid_conf, cams=spec["cams"])
    else:

        class EvalData(SegmentationData):
            def fix_nuscenes_formatting(self):
                pass

        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
        valdata = EvalData(nusc, is_train=False, data_aug_conf=data_aug_conf, grid_conf=grid_conf)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=args.bsz, shuffle=False, num_workers=args.nworkers)

    device = torch.device("cpu") if args.gpuid < 0 else torch.device(f"cuda:{args.gpuid}")
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)

    loss_fn = SimpleLoss(1.0).to(device)
    start = time.perf_counter()
    info = get_val_info(model, valloader, loss_fn, device)
    summary = {
        "dataset": args.dataset,
        "num_samples": len(valdata),
        "cams": spec["cams"],
        "iou": float(info["iou"]),
        "loss": float(info["loss"]),
        "elapsed_sec": round(time.perf_counter() - start, 2),
    }
    print(summary)

    if args.viz_dir:
        save_visualizations(model, valdata, device, spec["cams"], Path(args.viz_dir), args.viz_num, denormalize_img)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
