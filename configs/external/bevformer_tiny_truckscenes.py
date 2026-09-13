_base_ = [
    "../../third_party/BEVFormer/projects/configs/bevformer/bevformer_tiny.py",
]

data_root = "data/truckscenes/man-truckscenes/"

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True,
)

data = dict(
    train=dict(
        data_root=data_root,
        ann_file=data_root + "nuscenes_infos_temporal_train.pkl",
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + "nuscenes_infos_temporal_val.pkl",
    ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + "nuscenes_infos_temporal_val.pkl",
    ),
)
