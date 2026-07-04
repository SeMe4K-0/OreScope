"""Массовая загрузка ore_data в работающий дашборд OreScope через API.

Загружает все фото классов ч1/ч2 (кроме копий с разметкой «Области оталькования»)
и панорамы; очередь дашборда обрабатывает их последовательно.

Запуск: py -3.11 scripts/upload_all.py [--url http://127.0.0.1:7860] [--batch 10]
"""
from __future__ import annotations

import argparse
import mimetypes
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ore_data"

FOLDERS = [
    DATA / "Фото руд по сортам. ч1" / "Оталькованные руды",
    DATA / "Фото руд по сортам. ч1" / "Рядовые руды",
    DATA / "Фото руд по сортам. ч1" / "Труднообогатимые руды",
    DATA / "Фото руд по сортам. ч2" / "оталькованные",
    DATA / "Фото руд по сортам. ч2" / "рядовые",
    DATA / "Фото руд по сортам. ч2" / "тонкие",
    DATA / "Панорамы",
]
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def post_batch(url: str, files: list[Path]) -> int:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for p in files:
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
                 f"filename=\"{p.name}\"\r\nContent-Type: {ctype}\r\n\r\n").encode("utf-8")
        body += p.read_bytes()
        body += b"\r\n"
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"um_per_px\"\r\n\r\n0"
             ).encode("utf-8")
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        f"{url}/api/samples", data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()

    all_files: list[Path] = []
    for folder in FOLDERS:
        if not folder.exists():
            print(f"[SKIP] нет папки {folder}")
            continue
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in EXTS)
        all_files += files
        print(f"{folder.name}: {len(files)} файлов")
    print(f"Всего к загрузке: {len(all_files)}")

    sent = 0
    for i in range(0, len(all_files), args.batch):
        chunk = all_files[i:i + args.batch]
        # панорамы (>40 МБ) шлём по одной
        singles = [p for p in chunk if p.stat().st_size > 40 * 1024 * 1024]
        rest = [p for p in chunk if p not in singles]
        for group in ([rest] if rest else []) + [[s] for s in singles]:
            for attempt in range(3):
                try:
                    post_batch(args.url, group)
                    break
                except Exception as e:
                    print(f"[RETRY {attempt + 1}] {e}", flush=True)
                    time.sleep(5)
            else:
                print(f"[FAIL] {[p.name for p in group]}", flush=True)
                continue
            sent += len(group)
        if sent and sent % 100 < args.batch:
            print(f"  отправлено {sent}/{len(all_files)}", flush=True)
    print(f"Готово: отправлено {sent}/{len(all_files)}")


if __name__ == "__main__":
    main()
