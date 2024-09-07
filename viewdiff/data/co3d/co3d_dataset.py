# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import crop
from torch.nn.functional import interpolate

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils.camera_conversions import opencv_from_cameras_projection
from pytorch3d.io import IO
from tqdm import tqdm
import cv2

from .util import (
    get_dataset,
    has_pointcloud,
    get_crop_around_mask,
    adjust_crop_size,
)

def read_transparent_png(filename, bg_color=0):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255 * bg_color

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    final_image = (final_image).astype(np.uint8)
    final_image = cv2.resize(final_image, (256, 256))
    return final_image[:,:,::-1], alpha_channel

def read_bbox_file(file_path):
    with open(file_path, 'r') as file: 
        lines = file.readlines()

    values = lines[0].split() + lines[1].split()
    float_values = [float(value) for value in values]
    
    return float_values

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                d = {}
                d['R'] = qvec2rotmat(qvec)
                d['t'] = tvec
                images[image_name] = d
    return images


@dataclass
class DatasetArgsConfig:
    """Arguments for JsonIndexDataset. See here for a full list: pytorch3d/implicitron/dataset/json_index_dataset.py"""

    remove_empty_masks: bool = False
    """Removes the frames with no active foreground pixels
            in the segmentation mask after thresholding (see box_crop_mask_thr)."""

    load_point_clouds: bool = False
    """If pointclouds should be loaded from the dataset"""

    load_depths: bool = False
    """If depth_maps should be loaded from the dataset"""

    load_depth_masks: bool = False
    """If depth_masks should be loaded from the dataset"""

    load_masks: bool = False
    """If foreground masks should be loaded from the dataset"""

    box_crop: bool = False
    """Enable cropping of the image around the bounding box inferred
            from the foreground region of the loaded segmentation mask; masks
            and depth maps are cropped accordingly; cameras are corrected."""

    image_width: Optional[int] = None
    """The width of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing."""

    image_height: Optional[int] = None
    """The height of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing."""

    pick_sequence: Tuple[str, ...] = ()
    """A list of sequence names to restrict the dataset to."""

    exclude_sequence: Tuple[str, ...] = ()
    """a list of sequences to exclude"""

    n_frames_per_sequence: int = -1
    """If > 0, randomly samples #n_frames_per_sequence
        frames in each sequences uniformly without replacement if it has
        more frames than that; applied before other frame-level filters."""


@dataclass
class BatchConfig:
    """Arguments for how batches are constructed."""

    n_parallel_images: int = 5
    """How many images of the same sequence are selected in one batch (used for multi-view supervision)."""

    image_width: int = 512
    """The desired image width after applying all augmentations (e.g. crop) and resizing operations."""

    image_height: int = 512
    """The desired image height after applying all augmentations (e.g. crop) and resizing operations."""

    other_selection: Literal["random", "sequence", "mix", "fixed-frames"] = "random"
    """How to select the other frames for each batch.
        The mode 'random' selects the other frames at random from all remaining images in the dataset.
        The mode 'sequence' selects the other frames in the order as they appear after the first frame (==idx) in the dataset. Selects i-th other image as (idx + i * sequence_offset).
        The mode 'mix' decides at random which of the other two modes to choose. It also randomly samples sequence_offset when choosing the mode 'sequence'.
        The mode 'fixed-frames' gets as frame indices as input and directly uses them."""

    other_selection_frame_indices: Tuple[int, ...] = ()
    """The frame indices to use when --other_selection=fixed-frames. Must be as many indices as --n_parallel_images."""

    sequence_offset: int = 1
    """If other_selection='sequence', uses this offset to determine how many images to skip for each next frame.
    Allows to do short-range and long-range consistency tests by setting to a small or large number."""

    crop: Literal["random", "foreground", "resize", "center"] = "random"
    """Performs a crop on the original image such that the desired (image_height, image_width) is achieved. 
       The mode 'random' crops randomly in the image.
       The mode 'foreground' crops centered around the foreground (==object) mask.
       The mode 'resize' performs brute-force resizing which ignores the aspect ratio.
       The mode 'center' crops centered around the middle pixel (similar to DFM baseline)."""

    mask_foreground: bool = False
    """If true, will mask out the background and only keep the foreground."""

    prompt: str = "Editorial Style Photo, ${category}, 4k --ar 16:9"
    """The text prompt for generation. The string ${category} will be replaced with the actual category."""

    use_blip_prompt: bool = False
    """If True, will use blip2 generated prompts for the sequence instead of the prompt specified in --prompt."""

    load_recentered: bool = False
    """If True, will load the recentered poses/bbox from the dataset. Will skip all sequences for which this was not pre-computed."""

    replace_pose_with_spherical_start_phi: float = -400.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this many degrees. Default: -400, meaning do not replace."""

    replace_pose_with_spherical_end_phi: float = 360.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this many degrees. Default: -1, meaning do not replace."""

    replace_pose_with_spherical_phi_endpoint: bool = False
    """If True, will set endpoint=True for np.linspace, else False."""

    replace_pose_with_spherical_radius: float = 4.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this radius. Default: 3.0."""

    replace_pose_with_spherical_theta: float = 45.0
    """Replace the poses in each batch with spherical ones that go around the object in a partial circle of this elevation. Default: 45.0."""


@dataclass
class CO3DConfig:
    """Arguments for setup of the CO3Dv2_Dataset."""

    dataset_args: DatasetArgsConfig
    batch: BatchConfig

    co3d_root: str
    """Path to the co3dv2 root directory"""

    category: Optional[str] = None
    """If specified, only selects this category from the dataset. Can be a comma-separated list of categories as well."""

    subset: Optional[str] = None
    """If specified, only selects images corresponding to this subset. See https://github.com/facebookresearch/co3d for available options."""

    split: Optional[str] = None
    """Must be specified if --subset is specified. Tells which split to use from the subset."""

    max_sequences: int = -1
    """If >-1, randomly select max_sequence sequences per category. Only sequences _with pointclouds_ are selected. Mutually exclusive with --sequence."""

    seed: int = 42
    """Random seed for all rng objects"""


def spherical_to_cartesian(phi, theta, radius):
    # adapted from: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return torch.tensor([x, y, z], dtype=torch.float32)


def lookAt(eye, at, up):
    # adapted from: https://ksimek.github.io/2012/08/22/extrinsic/
    L = at - eye
    L = torch.nn.functional.normalize(L, dim=-1)
    s = torch.linalg.cross(L, up, dim=-1)
    s = torch.nn.functional.normalize(s, dim=-1)
    u = torch.linalg.cross(s, L, dim=-1)

    R = torch.stack([s, u, -L], dim=-2)
    t = torch.bmm(R, eye[..., None])
    w2c = torch.cat([R, t], dim=-1)
    hom = torch.zeros_like(w2c[..., 0:1, :])
    hom[..., -1] = 1
    w2c = torch.cat([w2c, hom], dim=-2)
    return w2c


def sample_uniform_poses_on_sphere(n_samples, radius=3.0, start_phi=0.0, end_phi=360.0, theta=30.0, endpoint: bool = False):
    eye = torch.stack([spherical_to_cartesian(phi=phi, theta=theta, radius=radius) for phi in np.linspace(start_phi, end_phi, n_samples, endpoint=endpoint)], dim=0)
    at = torch.tensor([[0, 0, 0]] * n_samples, dtype=torch.float32)
    up = torch.tensor([[0, 0, 1]] * n_samples, dtype=torch.float32)
    w2c = lookAt(eye, at, up)
    return w2c


def sample_random_poses_on_sphere(n_samples, radius=3.0, start_phi=0.0, end_phi=360.0, theta=30.0):
    phi_values = np.random.uniform(low=start_phi, high=end_phi, size=(n_samples,))
    eye = torch.stack([spherical_to_cartesian(phi=phi, theta=theta, radius=radius) for phi in phi_values], dim=0)
    at = torch.tensor([[0, 0, 0]] * n_samples, dtype=torch.float32)
    up = torch.tensor([[0, 0, 1]] * n_samples, dtype=torch.float32)
    w2c = lookAt(eye, at, up)
    return w2c


class CO3DDataset(Dataset):
    def __init__(self, config: CO3DConfig):
        self.config = config

        
        
        # obj_list_txt = '/nfs/tang.scratch.inf.ethz.ch/export/tang/cluster/yutongchen/code/nerfstudio/for_Shapenetcore/3d-denoiser/all.seen-categories_train.colmap.txt'
        obj_list_txt = '/nfs/tang.scratch.inf.ethz.ch/export/tang/cluster/yutongchen/data/OOD-benchmark/shapenetOOD/test1.txt'
        with open(obj_list_txt, 'r') as file:
            self.obj_list = file.read().splitlines()
        self.len = len(self.obj_list)
        # self.obj_paths = '/nfs/tang.scratch.inf.ethz.ch/export/tang/cluster/yutongchen/code/stanford-shapenet-renderer/results_256_sin-0-10x5_max2-20x1_1-4_rotate_sh0/'
        self.obj_paths = '/nfs/tang.scratch.inf.ethz.ch/export/tang/cluster/yutongchen/data/OOD-benchmark/shapenetOOD/test100/'
        self.valid_categories = []

    def get_all_sequences(self):
        sequences = ()
        for c in self.valid_categories:
            sequences += tuple(self.category_dict[c]["sequences"])
        return sequences
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # construct output
        output = {
            "images": [],
            "intensity_stats": [],
            "pose": [],
            "K": [],
            # "root": str(root),
            "file_names": [],
            "foreground_mask": [],
            "foreground_prob": []
        }

        output["prompt"] = ""

        obj = self.obj_list[idx]
        sample_dir = os.path.join(self.obj_paths, obj)
        image_dir = os.path.join(sample_dir, 'images')
        image_names = os.listdir(image_dir)
        image_names.sort()
        poses_colmap = read_images_text(f'{sample_dir}/sparse/0/images.txt')
        bbox = np.array(read_bbox_file(f'{sample_dir}/sparse/0/bbox.txt')).reshape(2,3)
        center = bbox.sum(axis=0)/2
        n = 5
        
        selected_indices = random.sample(range(0, 42), n)
        for i in selected_indices:
            image_name = image_names[i]
            w2c = np.eye(4)
            R = poses_colmap[image_name]['R']
            w2c[:3,:3] = R
            t = poses_colmap[image_name]['t'].reshape(3,1)
            t += R@center.reshape(3,1)
            w2c[:3,3] = t.reshape(3,) * (0.5 / (bbox - center).max())
            filename = f"{obj}_{image_names[i]}"
            
            image_rgb, mask = read_transparent_png(os.path.join(image_dir, image_names[i]))
            image_rgb = torch.from_numpy(image_rgb / 255).float().permute(2,0,1)
            # print(image_rgb.shape, mask.shape, mask.sum(), mask.max(), mask.min(), np.unique(mask))
            # exit()
            fg_prob = torch.from_numpy(mask / 255).float()
            fg_mask = torch.zeros_like(fg_prob)
            thr = 0.2
            while fg_mask.sum() <= 1.0:
                fg_mask = fg_prob > thr
                thr -= 0.05
            fg_mask = (~fg_mask)

            output["foreground_mask"].append(fg_mask)
            output["foreground_prob"].append(fg_prob)

            image_rgb = image_rgb * 2.0 - 1.0  # StableDiffusion expects images in [-1, 1]

            output["images"].append(image_rgb) # N, 3, H, W
            output["file_names"].append(filename) # 
            output["pose"].append(torch.tensor(w2c).float())
            K = torch.tensor([[800., 0., 127.5], [0., 800., 127.5], [0., 0., 1.]]).float()
            output["K"].append(K)
            output["bbox"] = torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]]).float()

            # get intensity stats
            var, mean = torch.var_mean(image_rgb)
            intensity_stats = torch.stack([mean, var], dim=0)
            output["intensity_stats"].append(intensity_stats)
        output["selected_indices"] = selected_indices
        # selected_indices_known = list(range(0, 42))
        # for i in selected_indices:
        #     selected_indices_known.remove(i)
        # known_indices = random.sample(selected_indices_known, 1)
        
        # for i in known_indices:
        #     image_name = image_names[i]
        #     image_rgb = torch.from_numpy(read_transparent_png(os.path.join(image_dir, image_names[i])) / 255).float().permute(2,0,1)
        #     image_rgb = image_rgb * 2.0 - 1.0  # StableDiffusion expects images in [-1, 1]
        #     filename = f"{obj}_{image_names[i]}"
        #     output["known_images"] = [image_rgb]
        #     output["known_images_filenames"] = [filename]

        # convert lists to tensors
        for k in output.keys():
            if isinstance(output[k], list) and isinstance(output[k][0], torch.Tensor):
                output[k] = torch.stack(output[k])
        output["selected_indices"] = torch.tensor(selected_indices)

        return output


class CO3DDreamboothDataset(Dataset):
    DEFAULT_MIN_FOCAL: float = 565.0
    DEFAULT_MAX_FOCAL: float = 590.0
    DEFAULT_CX: float = 255.5
    DEFAULT_CY: float = 255.5
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512

    @staticmethod
    def get_intrinsics_for_image_size(width: int = 512, height: int = 512):
        # get scaling factors
        scaling_factor_h = height / CO3DDreamboothDataset.DEFAULT_HEIGHT
        scaling_factor_w = width / CO3DDreamboothDataset.DEFAULT_WIDTH

        # fx
        min_focal_x = CO3DDreamboothDataset.DEFAULT_MIN_FOCAL * scaling_factor_w
        max_focal_x = CO3DDreamboothDataset.DEFAULT_MAX_FOCAL * scaling_factor_w

        # fy
        min_focal_y = CO3DDreamboothDataset.DEFAULT_MIN_FOCAL * scaling_factor_h
        max_focal_y = CO3DDreamboothDataset.DEFAULT_MAX_FOCAL * scaling_factor_h

        # We assume opencv-convention ((0, 0) refers to the center of the top-left pixel and (-0.5, -0.5) is the top-left corner of the image-plane):
        # We need to scale the principal offset with the 0.5 add/sub like here.
        # see this for explanation: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
        cx = (CO3DDreamboothDataset.DEFAULT_CX + 0.5) * scaling_factor_w - 0.5
        cy = (CO3DDreamboothDataset.DEFAULT_CY + 0.5) * scaling_factor_h - 0.5

        return min_focal_x, max_focal_x, min_focal_y, max_focal_y, cx, cy

    def __init__(
        self,
        co3d_root: str,
        selected_categories: Tuple[str, ...] = (),
        pose_spherical_min_radius: float = 2.0,
        pose_spherical_max_radius: float = 4.0,
        pose_spherical_min_theta: float = 30.0,
        pose_spherical_max_theta: float = 90.0,
        width: int = 512,
        height: int = 512,
    ):
        self.pose_spherical_min_radius = pose_spherical_min_radius
        self.pose_spherical_max_radius = pose_spherical_max_radius
        self.pose_spherical_min_theta = pose_spherical_min_theta
        self.pose_spherical_max_theta = pose_spherical_max_theta
        self.width = width
        self.height = height
        self.min_focal_x, self.max_focal_x, self.min_focal_y, self.max_focal_y, self.cx, self.cy = CO3DDreamboothDataset.get_intrinsics_for_image_size(width, height)
        self.bbox = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32)

        root = os.path.join(co3d_root, "dreambooth_prior_preservation_dataset")
        categories = os.listdir(root)
        self.images = []
        self.prompts = []
        for c in categories:
            if len(selected_categories) > 0 and c not in selected_categories:
                continue

            category_path = os.path.join(root, c)
            if not os.path.isdir(category_path):
                continue

            files = os.listdir(os.path.join(root, c))
            images = [os.path.join(category_path, f) for f in files if ".jpg" in f]
            image_to_prompt = [os.path.join(category_path, f) for f in files if f == "image_to_prompt.json"][0]
            with open(image_to_prompt, "r") as f:
                image_to_prompt = json.load(f)
            prompts = [image_to_prompt[os.path.splitext(os.path.basename(img))[0]] for img in images]
            assert len(images) == len(prompts)

            self.images.extend(images)
            self.prompts.extend(prompts)

        print("Constructed CO3DDreamboothDataset. Total #images:", len(self.images), "Selected Categories:", selected_categories, "Image Size:", height, "x", width)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get random pose
        pose = sample_random_poses_on_sphere(
            n_samples=1,
            radius=random.uniform(self.pose_spherical_min_radius, self.pose_spherical_max_radius),
            end_phi=360.0,
            theta=random.uniform(self.pose_spherical_min_theta, self.pose_spherical_max_theta)
        )

        # get random intrinsics
        w = random.uniform(0.0, 1.0)
        K = np.eye(3, 3, dtype=np.float32)
        K[0, 0] = w * self.min_focal_x + (1 - w) * self.max_focal_x
        K[0, 2] = self.cx
        K[1, 1] = w * self.min_focal_y + (1 - w) * self.max_focal_y
        K[1, 2] = self.cy
        K = K[None, ...]
        K = torch.from_numpy(K)

        # get next image/prompt
        prompt = self.prompts[idx]
        image_file = self.images[idx]
        with open(image_file, "rb") as f:
            image = Image.open(f)
            if image.width != self.width or image.height != self.height:
                image = image.resize((self.width, self.height), Image.BILINEAR)
            image = np.array(image)
            image = image / 255.0
            image = image * 2.0 - 1.0
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)

            # get intensity stats
            var, mean = torch.var_mean(image)
            intensity_stats = torch.stack([mean, var], dim=0)
            intensity_stats = intensity_stats.unsqueeze(0)

        # build output
        item = {
            "images": image,
            "intensity_stats": intensity_stats,
            "prompt": prompt,
            "pose": pose,
            "K": K,
            "bbox": self.bbox.clone(),
        }

        return item
