"""CNN-классификатор типа срастаний (обычные/тонкие) на weak labels.

Фаза prepare: из ~1034 фото ч1+ч2 (только рядовые/тонкие) нарезаются кропы
512px с долей сульфидов >= 3% (грубый Otsu), метка кропа = метка фото.
Фаза train: resnet18(ImageNet) -> 2 класса; сплит по изображениям;
image-level предсказание = среднее по кропам; метрика — macro-F1.

Запуск: py -3.11 scripts/train_grain_cnn.py --prepare
        py -3.11 scripts/train_grain_cnn.py --train [--epochs 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import imread_rgb, imwrite  # noqa: E402
from utils import get_device, seed_everything  # noqa: E402

DATA = ROOT / "ore_data"
SOURCES = [
    (DATA / "Фото руд по сортам. ч1" / "Рядовые руды", 0),
    (DATA / "Фото руд по сортам. ч1" / "Труднообогатимые руды", 1),
    (DATA / "Фото руд по сортам. ч2" / "рядовые", 0),
    (DATA / "Фото руд по сортам. ч2" / "тонкие", 1),
]
CROPS_DIR = ROOT / "data" / "grain_crops"
INDEX = CROPS_DIR / "index.json"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def prepare(crop: int = 512, per_image: int = 8, max_side: int = 2048) -> None:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    index = []
    t0 = time.time()
    n_img = 0
    for folder, label in SOURCES:
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for path in files:
            try:
                rgb = imread_rgb(path)
            except Exception as e:
                print(f"[ERR] {path.name}: {e}", flush=True)
                continue
            h, w = rgb.shape[:2]
            s = max_side / max(h, w)
            if s < 1.0:
                rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            sulf = gray > t
            H, W = gray.shape
            if H <= crop or W <= crop:
                continue
            saved = 0
            for _ in range(per_image * 6):
                if saved >= per_image:
                    break
                y = int(rng.integers(0, H - crop))
                x = int(rng.integers(0, W - crop))
                if sulf[y:y + crop, x:x + crop].mean() < 0.03:
                    continue
                name = f"{n_img:05d}_{saved}.jpg"
                imwrite(CROPS_DIR / name, rgb[y:y + crop, x:x + crop])
                index.append({"file": name, "label": label, "image": str(path)})
                saved += 1
            n_img += 1
            if n_img % 100 == 0:
                print(f"  {n_img} изображений, {len(index)} кропов, {time.time()-t0:.0f}s", flush=True)
    INDEX.write_text(json.dumps(index), encoding="utf-8")
    print(f"Готово: {n_img} изображений -> {len(index)} кропов", flush=True)


class CropDS(Dataset):
    def __init__(self, items: list[dict], train: bool):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = imread_rgb(CROPS_DIR / it["file"])
        rng = np.random.default_rng()
        if self.train:
            # масштабная инвариантность: случайный суб-кроп 45-100% -> 224
            f = rng.uniform(0.45, 1.0)
            cs = int(img.shape[0] * f)
            y = int(rng.integers(0, img.shape[0] - cs + 1))
            x = int(rng.integers(0, img.shape[1] - cs + 1))
            img = img[y:y + cs, x:x + cs]
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            k = int(rng.integers(0, 4))
            img = np.rot90(img, k).copy()
            if rng.random() < 0.5:
                img = np.fliplr(img).copy()
            img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.3), 0, 255)
            g = img.mean(axis=2, keepdims=True)
            img = img * (1 - rng.uniform(0, 0.6)) + g * rng.uniform(0, 0.6)
        else:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        arr = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1)), it["label"], it["image"]


def train(epochs: int, batch: int, lr: float) -> None:
    import torchvision

    seed_everything(42)
    device = get_device()
    items = json.loads(INDEX.read_text(encoding="utf-8"))
    images = sorted({it["image"] for it in items})
    rng = np.random.default_rng(42)
    val_images = set(rng.choice(images, size=int(len(images) * 0.15), replace=False).tolist())
    train_items = [it for it in items if it["image"] not in val_images]
    val_items = [it for it in items if it["image"] in val_images]
    print(f"crops: train={len(train_items)}, val={len(val_items)}; "
          f"images: train={len(images)-len(val_images)}, val={len(val_images)}", flush=True)

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)

    tdl = DataLoader(CropDS(train_items, True), batch_size=batch, shuffle=True, num_workers=0)
    vdl = DataLoader(CropDS(val_items, False), batch_size=batch, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for xb, yb, _ in tdl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()

        # image-level валидация: среднее prob по кропам изображения
        model.eval()
        probs: dict[str, list[float]] = {}
        labels: dict[str, int] = {}
        with torch.no_grad():
            for xb, yb, ims in vdl:
                p = torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy()
                for pi, yi, im in zip(p, yb.numpy(), ims):
                    probs.setdefault(im, []).append(float(pi))
                    labels[im] = int(yi)
        y_true = np.array([labels[im] for im in probs])
        y_pred = np.array([int(np.mean(v) >= 0.5) for v in probs.values()])
        f1s = []
        for cls in (0, 1):
            tp = ((y_pred == cls) & (y_true == cls)).sum()
            fp = ((y_pred == cls) & (y_true != cls)).sum()
            fn = ((y_pred != cls) & (y_true == cls)).sum()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        f1 = float(np.mean(f1s))
        acc = float((y_pred == y_true).mean())
        mark = ""
        if f1 > best:
            best = f1
            torch.save(model.state_dict(), ROOT / "models" / "grain_cnn.pt")
            mark = " *best*"
        print(f"epoch {ep}/{epochs} | loss {np.mean(losses):.4f} | img-F1 {f1:.4f} | "
              f"img-acc {acc:.4f} | {time.time()-t0:.0f}s{mark}", flush=True)
    print(f"Лучший image-level macro-F1: {best:.4f} -> models/grain_cnn.pt", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.train:
        train(args.epochs, args.batch, args.lr)
