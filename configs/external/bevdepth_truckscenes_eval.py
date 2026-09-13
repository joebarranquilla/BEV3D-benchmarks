import os

from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.mv.bev_depth_lss_r50_256x704_128x128_24e_2key import (
    BEVDepthLightningModel as BaseBEVDepthLightningModel,
)


class BEVDepthTruckScenesLightningModel(BaseBEVDepthLightningModel):
    def __init__(self, **kwargs):
        kwargs.setdefault("data_root", "data/truckscenes/man-truckscenes")
        super().__init__(**kwargs)

        self.ida_aug_conf["cams"] = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
        self.ida_aug_conf["Ncams"] = len(self.ida_aug_conf["cams"])
        self.ida_aug_conf["H"] = 943
        self.ida_aug_conf["W"] = 1980

        self.train_info_paths = os.path.join(self.data_root, "nuscenes_infos_train.pkl")
        self.val_info_paths = os.path.join(self.data_root, "nuscenes_infos_val.pkl")


if __name__ == "__main__":
    run_cli(BEVDepthTruckScenesLightningModel, "truckscenes-mini-bevdepth")
