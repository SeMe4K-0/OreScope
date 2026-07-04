from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def get_train_transforms(cfg: DictConfig) -> A.Compose:
    aug = cfg.augmentation
    norm = aug.normalize
    size = cfg.data.image_size
    transforms: list[A.BasicTransform] = []

    if aug.train.rotate90:
        transforms.append(A.RandomRotate90(p=0.5))
    if aug.train.horizontal_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug.train.vertical_flip:
        transforms.append(A.VerticalFlip(p=0.5))
    if aug.train.brightness_contrast:
        bc = aug.train.brightness_contrast
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=bc.brightness_limit,
                contrast_limit=bc.contrast_limit,
                p=0.4,
            )
        )
    if aug.train.gauss_noise:
        gn = aug.train.gauss_noise
        # albumentations 2.x uses std_range (not var_limit); convert: std = sqrt(var)/255
        var_lo, var_hi = tuple(gn.var_limit)
        std_lo = (var_lo ** 0.5) / 255.0
        std_hi = (var_hi ** 0.5) / 255.0
        transforms.append(
            A.GaussNoise(std_range=(std_lo, std_hi), p=0.3)
        )
    if aug.train.motion_blur:
        mb = aug.train.motion_blur
        transforms.append(A.MotionBlur(blur_limit=mb.blur_limit, p=0.2))
    if aug.train.coarse_dropout:
        cd = aug.train.coarse_dropout
        # albumentations 2.x renamed params to *_range tuples
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, cd.max_holes),
                hole_height_range=(cd.max_height // 2, cd.max_height),
                hole_width_range=(cd.max_width // 2, cd.max_width),
                p=0.2,
            )
        )
    if aug.train.random_scale:
        rs = aug.train.random_scale
        transforms.append(
            A.RandomScale(scale_limit=rs.scale_limit, p=0.3)
        )

    transforms += [
        A.Resize(size, size),
        A.Normalize(mean=list(norm.mean), std=list(norm.std)),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_valid_transforms(cfg: DictConfig) -> A.Compose:
    norm = cfg.augmentation.normalize
    size = cfg.data.image_size
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=list(norm.mean), std=list(norm.std)),
        ToTensorV2(),
    ])


def get_tta_transforms(cfg: DictConfig) -> list[A.Compose]:
    norm = cfg.augmentation.normalize
    size = cfg.data.image_size
    base = [
        A.Resize(size, size),
        A.Normalize(mean=list(norm.mean), std=list(norm.std)),
        ToTensorV2(),
    ]
    angles = [0, 90, 180, 270]
    result: list[A.Compose] = []
    for angle in angles:
        if angle == 0:
            result.append(A.Compose(base))
        else:
            result.append(
                A.Compose([A.Rotate(limit=(angle, angle), p=1.0)] + base)
            )
    return result
