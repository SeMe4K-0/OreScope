"""Обучение U-Net сегментации талька на уточнённых масках (рассеянная фаза).

- Цель: тальковая ФАЗА (тёмная рассеянная в нерудной матрице) — маски
  data/talc_masks после уточнения make_talc_masks.py.
- data/talc_ignore — тёмная фаза вне экспертных зон (возможный неразмеченный
  тальк): помечается 255 и исключается из лосса (CE ignore_index + маска в Dice).
- Дополнительно: --extra-dir data/talc_masks_ch2 — проверенные псевдомаски ч2
  (пары {stem}.png + {stem}_ignore.png + {stem}_rgb.jpg с препроцессингом).
- Сплит train/val по изображениям (не по тайлам) — без утечки.
- Лосс: Dice + CE (по SulfideNet, IEEE JSTARS 2025).
- Аугментации: геометрия агрессивно, цвет умеренно + доменная симуляция
  «панорамного» профиля (затемнение/обесцвечивание) + царапины.

Запуск: py -3.11 scripts/train_talc.py [--epochs 40] [--resume models/talc_best.pt]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_rgb  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.segmentation.model import build_model, save_checkpoint  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402

ORIG_DIR = ROOT / "ore_data" / "Фото руд по сортам. ч1" / "Оталькованные руды"
MASK_DIR = ROOT / "data" / "talc_masks"
IGNORE_DIR = ROOT / "data" / "talc_ignore"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE = 255


def domain_shift(image: np.ndarray, **kwargs) -> np.ndarray:
    """Симуляция панорамного профиля: затемнение + нейтрализация цвета."""
    rng = np.random.default_rng()
    dark = rng.uniform(0.35, 0.85)
    desat = rng.uniform(0.3, 0.9)
    img = image.astype(np.float32) * dark
    gray = img.mean(axis=2, keepdims=True)
    img = img * (1 - desat) + gray * desat
    return np.clip(img, 0, 255).astype(np.uint8)


def add_scratches(image: np.ndarray, **kwargs) -> np.ndarray:
    rng = np.random.default_rng()
    img = image.copy()
    h, w = img.shape[:2]
    for _ in range(rng.integers(1, 5)):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        val = int(rng.integers(0, 60)) if rng.random() < 0.5 else int(rng.integers(180, 255))
        cv2.line(img, p1, p2, (val, val, val), int(rng.integers(1, 3)))
    return img


def get_transforms(size: int, train: bool) -> A.Compose:
    if not train:
        return A.Compose([A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(0.25, 0.25, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.25),
        A.Lambda(image=domain_shift, p=0.35),
        A.Lambda(image=add_scratches, p=0.25),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])


class TalcTiles(Dataset):
    """Тайлы из полных изображений; train — случайные кропы, val — сетка."""

    def __init__(self, items: list[tuple[np.ndarray, np.ndarray]], size: int,
                 train: bool, samples_per_image: int = 24):
        self.items = items
        self.size = size
        self.train = train
        self.spi = samples_per_image
        self.tf = get_transforms(size, train)
        if not train:
            self.grid = []
            for idx, (img, _) in enumerate(items):
                h, w = img.shape[:2]
                for y in range(0, h - size + 1, size):
                    for x in range(0, w - size + 1, size):
                        self.grid.append((idx, y, x))

    def __len__(self) -> int:
        return len(self.items) * self.spi if self.train else len(self.grid)

    def __getitem__(self, i: int):
        s = self.size
        if self.train:
            idx = i % len(self.items)
            img, msk = self.items[idx]
            h, w = img.shape[:2]
            rng = np.random.default_rng()
            # с вероятностью 0.5 сэмплируем кроп с тальком (борьба с дисбалансом)
            for _ in range(8):
                y = int(rng.integers(0, h - s + 1))
                x = int(rng.integers(0, w - s + 1))
                crop_m = msk[y:y + s, x:x + s]
                if rng.random() > 0.5 or (crop_m == 1).mean() > 0.02:
                    break
            crop_i = img[y:y + s, x:x + s]
        else:
            idx, y, x = self.grid[i]
            img, msk = self.items[idx]
            crop_i = img[y:y + s, x:x + s]
            crop_m = msk[y:y + s, x:x + s]
        out = self.tf(image=crop_i, mask=crop_m.astype(np.uint8))
        return out["image"], out["mask"].long()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    keep = (target != IGNORE).float()
    t = (target == 1).float()
    inter = (probs * t * keep).sum(dim=(1, 2))
    denom = (probs * keep).sum(dim=(1, 2)) + (t * keep).sum(dim=(1, 2))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def val_dice(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    inter_sum, denom_sum = 0.0, 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb).argmax(dim=1).cpu()
        keep = yb != IGNORE
        t = (yb == 1) & keep
        p = (pred == 1) & keep
        inter_sum += float((p & t).sum())
        denom_sum += float(p.sum() + t.sum())
    return 2 * inter_sum / max(denom_sum, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--negatives", type=int, default=0,
                        help="добавить N фото рядовых/тонких как чистые негативы (тальк=0)")
    parser.add_argument("--neg-domains", choices=["all", "ch1"], default="all",
                        help="ch1: негативы только из ч1 — пока нет ч2-позитивов, иначе сеть "
                             "учит шорткат «текстура ч2 -> не тальк» (наблюдалось на v2)")
    parser.add_argument("--extra-dir", type=str, default="",
                        help="папка проверенных псевдомасок ч2 (только в train)")
    args = parser.parse_args()

    seed_everything(42)
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    device = get_device()
    size = int(cfg.data.image_size)
    print(f"device={device}, tile={size}")

    # ── данные: препроцессированный оригинал + маска {0,1,255=ignore} ───────
    mask_files = sorted(MASK_DIR.glob("*.png"))
    items: list[tuple[np.ndarray, np.ndarray]] = []
    for mp in mask_files:
        orig = None
        for ext in (".JPG", ".jpg", ".jpeg", ".png"):
            cand = ORIG_DIR / (mp.stem + ext)
            if cand.exists():
                orig = cand
                break
        if orig is None:
            print(f"[SKIP] нет оригинала для {mp.name}")
            continue
        rgb = imread_rgb(orig)
        talc = cv2.imdecode(np.fromfile(str(mp), np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        if talc.shape != rgb.shape[:2]:
            talc = cv2.resize(talc.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        pre = preprocess(rgb, cfg)
        target = np.zeros(rgb.shape[:2], np.uint8)
        target[talc] = 1
        ign_p = IGNORE_DIR / mp.name
        if ign_p.exists():
            ign = cv2.imdecode(np.fromfile(str(ign_p), np.uint8), cv2.IMREAD_GRAYSCALE) > 127
            target[ign & ~talc] = IGNORE
        target[~pre.valid] = IGNORE
        items.append((pre.rgb, target))
    print(f"Изображений: {len(items)}")

    rng = np.random.default_rng(42)
    order = rng.permutation(len(items))
    n_val = max(1, int(len(items) * args.val_frac))
    val_idx = set(order[:n_val].tolist())
    train_items = [items[i] for i in range(len(items)) if i not in val_idx]
    val_items = [items[i] for i in range(len(items)) if i in val_idx]

    # негативные контроли (по SulfideNet): фото без талька, target=0 —
    # добавляются ТОЛЬКО в train, борьба с ложным тальком на рядовых/тонких
    if args.negatives > 0:
        neg_rng = np.random.default_rng(7)
        neg_sources = [
            ROOT / "ore_data" / "Фото руд по сортам. ч1" / "Рядовые руды",
            ROOT / "ore_data" / "Фото руд по сортам. ч1" / "Труднообогатимые руды",
        ]
        if args.neg_domains == "all":
            neg_sources += [
                ROOT / "ore_data" / "Фото руд по сортам. ч2" / "рядовые",
                ROOT / "ore_data" / "Фото руд по сортам. ч2" / "тонкие",
            ]
        per_src = max(1, args.negatives // len(neg_sources))
        n_neg = 0
        for folder in neg_sources:
            cand = sorted(p for p in folder.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            for p in neg_rng.choice(cand, size=min(per_src, len(cand)), replace=False):
                rgb = imread_rgb(p)
                h, w = rgb.shape[:2]
                if max(h, w) > 2272:
                    s = 2272 / max(h, w)
                    rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                pre = preprocess(rgb, cfg)
                target = np.zeros(pre.rgb.shape[:2], np.uint8)
                target[~pre.valid] = IGNORE
                train_items.append((pre.rgb, target))
                n_neg += 1
        print(f"негативов добавлено в train: {n_neg}")

    # проверенные псевдомаски ч2 (только train; val остаётся экспертным ч1)
    if args.extra_dir:
        extra_dir = ROOT / args.extra_dir
        n_extra = 0
        for mp in sorted(extra_dir.glob("*.png")):
            if mp.stem.endswith("_ignore"):
                continue
            rgb_p = extra_dir / f"{mp.stem}_rgb.jpg"
            if not rgb_p.exists():
                continue
            rgb = imread_rgb(rgb_p)  # уже препроцессирован при генерации
            talc = cv2.imdecode(np.fromfile(str(mp), np.uint8), cv2.IMREAD_GRAYSCALE) > 127
            target = np.zeros(rgb.shape[:2], np.uint8)
            target[talc] = 1
            ign_p = extra_dir / f"{mp.stem}_ignore.png"
            if ign_p.exists():
                ign = cv2.imdecode(np.fromfile(str(ign_p), np.uint8), cv2.IMREAD_GRAYSCALE) > 127
                target[ign & ~talc] = IGNORE
            train_items.append((rgb, target))
            n_extra += 1
        print(f"псевдомасок ч2 добавлено в train: {n_extra}")
    print(f"train={len(train_items)} img, val={len(val_items)} img")

    train_ds = TalcTiles(train_items, size, train=True, samples_per_image=24)
    val_ds = TalcTiles(val_items, size, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"train tiles/epoch={len(train_ds)}, val tiles={len(val_ds)}")

    model = build_model(cfg).to(device)
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"resume <- {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-7)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = 0.5 * ce(logits, yb) + 0.5 * dice_loss(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        vd = val_dice(model, val_dl, device)
        mark = ""
        if vd > best:
            best = vd
            save_checkpoint(model, opt, epoch, {"val_dice": vd}, ROOT / "models" / "talc_best.pt")
            mark = " *best*"
        print(f"epoch {epoch:02d}/{args.epochs} | loss {np.mean(losses):.4f} | "
              f"val_dice {vd:.4f} | {time.time() - t0:.0f}s{mark}", flush=True)

    save_checkpoint(model, opt, args.epochs, {"val_dice": best}, ROOT / "models" / "talc_last.pt")
    print(f"Готово. Лучший val Dice: {best:.4f} -> models/talc_best.pt")


if __name__ == "__main__":
    main()
