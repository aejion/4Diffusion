import math
import random
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.data.random_multiview import (
    RandomMultiviewCameraDataModuleConfig,
    RandomMultiviewCameraIterableDataset
)
from threestudio.utils.config import parse_structured
from threestudio.utils.base import Updateable
from threestudio.utils.typing import *
from threestudio.utils.ops import get_ray_directions, get_rays, get_projection_matrix, get_mvp_matrix

from threestudio.utils.misc import get_rank

class RandomSingleMultiViewCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg_single_view: Any, cfg_multi_view: Any, prob_multi_view: int = None) -> None:
        super().__init__()
        self.cfg_single = parse_structured(RandomCameraDataModuleConfig, cfg_single_view)
        self.cfg_multi = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg_multi_view)
        self.train_dataset_single = RandomCameraIterableDataset(self.cfg_single)
        self.train_dataset_multi = RandomMultiviewCameraIterableDataset(self.cfg_multi)
        self.idx = 0
        self.prob_multi_view = prob_multi_view

        self.rank = get_rank()
        self.default_elevation_deg = 0
        self.default_azimuth_deg = 0
        self.default_camera_distance = 1.1
        self.default_fovy_deg = 45
        self.height = 256
        self.width = 256
        self.n_view = 4
        self.azimuth_range = (-180, 180)

        # elevation_deg = torch.FloatTensor([self.default_elevation_deg])
        elevation_deg = torch.FloatTensor([self.default_elevation_deg]).repeat_interleave(self.n_view, dim=0)
        # azimuth_deg = torch.FloatTensor([self.default_azimuth_deg])

        azimuth_deg = (torch.arange(self.n_view).reshape(1, -1)).reshape(-1) / self.n_view * (
                              self.azimuth_range[1] - self.azimuth_range[0]
                      ) + self.azimuth_range[0]

        camera_distance = torch.FloatTensor([self.default_camera_distance]).repeat_interleave(self.n_view, dim=0)


        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "B 3"] = camera_position
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        self.c2w: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.default_fovy_deg]).repeat_interleave(self.n_view, dim=0))

        self.heights: List[int] = (
            [self.height] if isinstance(self.height, int) else self.height
        )
        self.widths: List[int] = (
            [self.width] if isinstance(self.width, int) else self.width
        )
        assert len(self.heights) == len(self.widths)

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]

        self.directions_unit_focal = self.directions_unit_focals[0]

        self.focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * self.fovy)
        self.directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
                                               None, :, :, :
                                               ].repeat(self.n_view, 1, 1, 1)
        self.directions[:, :, :, :2] = (
                self.directions[:, :, :, :2] / self.focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        self.rays_o, self.rays_d = get_rays(self.directions, self.c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.c2w, proj_mtx)


        self.image_path = self.cfg_multi.image_path
        self.anchor_view_num = self.cfg_multi.anchor_view_num


        self.length = self.cfg_multi.num_frames
        self.total_train_length = self.cfg_multi.num_frames

        timestamps = torch.linspace(0.0, 1.0, self.total_train_length)
        self.timestamps = timestamps
        self.mvp_mtx = mvp_mtx

        self.load_images()

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions, self.c2w, keepdim=True, noise_scale=0
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.total_train_length = self.cfg_multi.num_frames
        self.rays_o, self.rays_d = rays_o, rays_d
        # self.rays_o = self.rays_o.unsqueeze(1).repeat(1, self.length, 1, 1, 1)
        # self.rays_d = self.rays_d.unsqueeze(1).repeat(1, self.length, 1, 1, 1)

        timestamps = torch.linspace(0.0, 1.0, self.total_train_length)
        timestamps = timestamps[0::4]
        self.timestamps = timestamps
        self.mvp_mtx = mvp_mtx

    def load_images(self):
        # load image
        assert os.path.exists(
            self.image_path
        ), f"Could not find image {self.image_path}!"
        self.rgb = []
        self.mask = []
        self.rgb_path = []
        for j in range(4):
            for i in range(self.length):
                self.rgb_path.append(os.path.join(self.image_path, '{1}/{0}_rgba.png'.format(i, j)))
                rgba = cv2.cvtColor(
                    cv2.imread(os.path.join(self.image_path, '{1}/{0}_rgba.png'.format(i, j)), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
                )
                rgba = (
                    cv2.resize(
                        rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
                    ).astype(np.float32)
                    / 255.0
                )
                rgb = rgba[..., :3]
                alpha = rgba[..., 3:]
                rgb = rgb * alpha + 1.0 * (1 - alpha)
                self.rgb.append(
                    torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
                )
                self.mask.append(
                    torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
                )
        self.rgb = torch.cat(self.rgb, dim=0)
        self.mask = torch.cat(self.mask, dim=0)
        print(
            f"[INFO] single image dataset: load image {self.image_path} {self.rgb.shape}"
        )

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.train_dataset_single.update_step(epoch, global_step, on_load_weights)
        self.train_dataset_multi.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        if self.prob_multi_view is not None:
            multi = random.random() < self.prob_multi_view
        else:
            multi = False
        if multi:
            batch = self.train_dataset_multi.collate(batch)
            batch['single_view'] = False
            batch['is_video'] = True
        else:
            batch = self.train_dataset_single.collate(batch)
            batch['single_view'] = True

        index = (self.anchor_view_num + 2) % 4
        index2 = self.anchor_view_num
        if batch['stage_one'] == True:
            index3 = 0
        else:
            index3 = random.randint(0, self.length - 1)
        batch['gt_view'] = {
            'rgb_path': self.rgb_path[index2 * self.length + index3: index2 * self.length + index3 + 1],
            "rays_o": self.rays_o[index, None],
            "rays_d": self.rays_d[index, None],
            "mvp_mtx": self.mvp_mtx[index, None],
            "camera_positions": self.camera_position[index, None],
            "light_positions": self.light_position[index, None],
            "elevation": self.elevation_deg[index, None],
            "azimuth": self.azimuth_deg[index, None],
            "camera_distances": self.camera_distance[index, None],
            "rgb": self.rgb[index2 * self.length + index3: index2 * self.length + index3 + 1],
            "mask": self.mask[index2 * self.length + index3: index2 * self.length + index3 + 1],
            "height": self.height,
            "width": self.width,
            'frame_times': self.timestamps[index3, None],
            'train_dynamic_camera': False,
        }
        self.idx += 1
        return batch


@register("single-multiview-combined-camera-datamodule")
class SingleMultiviewCombinedCameraDataModule(pl.LightningDataModule):
    cfg: RandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg_multi = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg.multi_view)
        self.cfg_single = parse_structured(RandomCameraDataModuleConfig, cfg.single_view)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomSingleMultiViewCameraIterableDataset(self.cfg_single, self.cfg_multi, prob_multi_view=self.cfg.prob_multi_view)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg_single, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg_single, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

# @register("single-multiview-combined-camera-datamodule")
# class SingleMultiviewCombinedCameraDataModule(pl.LightningDataModule):
#     cfg: RandomMultiviewCameraDataModuleConfig

#     def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
#         super().__init__()
#         self.cfg_multi = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg.multi_view)
#         self.cfg_single = parse_structured(RandomCameraDataModuleConfig, cfg.single_view)

#     def setup(self, stage=None) -> None:
#         if stage in [None, "fit"]:
#             self.train_dataset_multi = RandomMultiviewCameraIterableDataset(self.cfg_multi)
#             self.train_dataset_single = RandomCameraIterableDataset(self.cfg_single)
#             self.train_single_view_loader = self.general_loader(
#             self.train_dataset_single, batch_size=None, collate_fn=self.train_dataset_single.collate
#             )
#             self.train_multi_view_loader = self.general_loader(
#                 self.train_dataset_multi, batch_size=None, collate_fn=self.train_dataset_multi.collate
#             )
#             breakpoint()
#         if stage in [None, "fit", "validate"]:
#             self.val_dataset = RandomCameraDataset(self.cfg_multi, "val")
#         if stage in [None, "test", "predict"]:
#             self.test_dataset = RandomCameraDataset(self.cfg_multi, "test")

#     def prepare_data(self):
#         pass

#     def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
#         return DataLoader(
#             dataset,
#             # very important to disable multi-processing if you want to change self attributes at runtime!
#             # (for example setting self.width and self.height in update_step)
#             num_workers=0,  # type: ignore
#             batch_size=batch_size,
#             collate_fn=collate_fn,
#         )

#     def train_dataloader(self) -> DataLoader:
#         breakpoint()
#         # self.train_single_view_loader = self.general_loader(
#         #     self.train_dataset_single, batch_size=None, collate_fn=self.train_dataset_single.collate
#         # )
#         # self.train_multi_view_loader = self.general_loader(
#         #     self.train_dataset_multi, batch_size=None, collate_fn=self.train_dataset_multi.collate
#         # )
#         # return {"single_view": self.train_single_view_loader, "multi_view": self.train_multi_view_loader}
#         return {"single_view": self.train_single_view_loader}

#     def val_dataloader(self) -> DataLoader:
#         return self.general_loader(
#             self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
#         )

#     def test_dataloader(self) -> DataLoader:
#         return self.general_loader(
#             self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
#         )

#     def predict_dataloader(self) -> DataLoader:
#         return self.general_loader(
#             self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
#         )
