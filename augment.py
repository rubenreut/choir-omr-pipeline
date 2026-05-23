#!/usr/bin/env python3
"""Augment rendered MusicXML→PNG pages to look like real phone photos.

v2 augmentations (added for ChoirSMT v2 training):
  - Rotation ±5° (was ±3)
  - Perspective warp up to ±8% (was 2.5%)
  - Brightness/contrast jitter
  - Gaussian blur (motion-like)
  - Paper tint
  - Additive noise
  - Vignetting (corners darker)
  - Lens distortion (barrel curve)
  - Random shadow (gradient across the page)
  - Uneven lighting (radial gradient)
  - Slight frame crop / margin variation

For each page in ./images/<piece_id>/page_N.png we produce K augmented variants
in ./augmented/<piece_id>/page_N_aug_K.webp.

Usage:
    python3 augment.py [--variants K] [--limit N]
"""
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

SRC_DIR = Path(__file__).parent / "images"
OUT_DIR = Path(__file__).parent / "augmented"


def random_perspective(img: Image.Image, max_shift: float = 0.08) -> Image.Image:
    """Stronger perspective warp simulating phone photo angle."""
    w, h = img.size
    def jitter(x: int, y: int) -> tuple[float, float]:
        return (x + random.uniform(-max_shift, max_shift) * w,
                y + random.uniform(-max_shift, max_shift) * h)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [jitter(*p) for p in src]
    matrix = []
    for s, d in zip(dst, src):
        matrix.append([s[0], s[1], 1, 0, 0, 0, -d[0]*s[0], -d[0]*s[1]])
        matrix.append([0, 0, 0, s[0], s[1], 1, -d[1]*s[0], -d[1]*s[1]])
    A = np.array(matrix, dtype=float)
    B = np.array([p for pt in src for p in pt], dtype=float)
    try:
        coeffs = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return img
    return img.transform((w, h), Image.PERSPECTIVE, coeffs.tolist(),
                          resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def vignette(img: Image.Image, strength: float = 0.5) -> Image.Image:
    """Apply vignetting — darken corners."""
    w, h = img.size
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    cx, cy = w / 2, h / 2
    max_d = math.sqrt(cx * cx + cy * cy)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_d
    mask = 1.0 - strength * (d ** 2)
    arr *= mask[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def lens_barrel(img: Image.Image, k: float = 0.05) -> Image.Image:
    """Apply slight barrel distortion."""
    w, h = img.size
    arr = np.array(img.convert("RGB"))
    cx, cy = w / 2, h / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - cx) / cx
    dy = (yy - cy) / cy
    r2 = dx * dx + dy * dy
    factor = 1 + k * r2
    src_x = (cx + dx * cx * factor).astype(np.int32)
    src_y = (cy + dy * cy * factor).astype(np.int32)
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    out = arr[src_y, src_x]
    return Image.fromarray(out)


def random_shadow(img: Image.Image, strength: float = 0.4) -> Image.Image:
    """Apply a soft shadow gradient across the page (simulates hand shadow)."""
    w, h = img.size
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    # Random shadow direction (angle 0–2π) and offset
    angle = random.uniform(0, 2 * math.pi)
    dx_dir = math.cos(angle)
    dy_dir = math.sin(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Distance along shadow direction, normalized
    t = (xx * dx_dir + yy * dy_dir)
    t_norm = (t - t.min()) / max(1, (t.max() - t.min()))
    # Shadow falls on one side, with a soft edge
    shadow_pos = random.uniform(0.3, 0.7)  # transition point
    softness = random.uniform(0.1, 0.3)
    mask = 1.0 - strength * np.clip((shadow_pos - t_norm) / softness, 0, 1)
    arr *= mask[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def uneven_lighting(img: Image.Image, strength: float = 0.3) -> Image.Image:
    """Apply a radial brightness gradient — one side brighter than the other."""
    w, h = img.size
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    # Light source position (random somewhere on the page)
    lx = random.uniform(0, w)
    ly = random.uniform(0, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2)
    d_norm = d / d.max()
    # Brighter near light, darker far away
    factor = 1.0 + strength * (0.5 - d_norm)
    arr *= factor[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def paper_tint(img: Image.Image) -> Image.Image:
    r = random.randint(245, 255)
    g = random.randint(240, 252)
    b = random.randint(225, 245)
    tint = Image.new("RGB", img.size, (r, g, b))
    return Image.blend(img.convert("RGB"), tint, alpha=random.uniform(0.05, 0.15))


def random_crop_margin(img: Image.Image) -> Image.Image:
    """Slightly crop a margin off the page randomly (simulates page not perfectly framed)."""
    w, h = img.size
    crop_pct = random.uniform(0.0, 0.04)
    cl = int(w * random.uniform(0, crop_pct))
    cr = int(w * random.uniform(0, crop_pct))
    ct = int(h * random.uniform(0, crop_pct))
    cb = int(h * random.uniform(0, crop_pct))
    return img.crop((cl, ct, w - cr, h - cb))


def augment_one(src_png: Path, n_variants: int, out_dir: Path) -> list[Path]:
    """Produce N augmented variants of a single page."""
    img = Image.open(src_png).convert("RGB")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    base_name = src_png.stem

    for k in range(n_variants):
        x = img

        # 1) rotation (stronger)
        angle = random.uniform(-5, 5)
        x = x.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255), expand=False)

        # 2) perspective warp (stronger, 70% of variants)
        if random.random() < 0.7:
            x = random_perspective(x, max_shift=random.uniform(0.03, 0.08))

        # 3) lens distortion (40%)
        if random.random() < 0.4:
            x = lens_barrel(x, k=random.uniform(0.02, 0.08))

        # 4) random crop margin (60%)
        if random.random() < 0.6:
            x = random_crop_margin(x)

        # 5) brightness/contrast jitter
        x = ImageEnhance.Brightness(x).enhance(random.uniform(0.80, 1.15))
        x = ImageEnhance.Contrast(x).enhance(random.uniform(0.80, 1.20))

        # 6) uneven lighting (50%)
        if random.random() < 0.5:
            x = uneven_lighting(x, strength=random.uniform(0.15, 0.35))

        # 7) shadow (35%)
        if random.random() < 0.35:
            x = random_shadow(x, strength=random.uniform(0.2, 0.5))

        # 8) blur (40%)
        if random.random() < 0.4:
            x = x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.5)))

        # 9) paper tint
        x = paper_tint(x)

        # 10) noise (60%)
        if random.random() < 0.6:
            arr = np.array(x, dtype=np.int16)
            noise = np.random.normal(0, random.uniform(4, 12), arr.shape).astype(np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            x = Image.fromarray(arr)

        # 11) vignetting (50%)
        if random.random() < 0.5:
            x = vignette(x, strength=random.uniform(0.2, 0.5))

        # Save as WebP
        out_path = out_dir / f"{base_name}_aug_{k:02d}.webp"
        x.save(out_path, "WEBP", quality=random.randint(70, 92))
        results.append(out_path)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=10,
                    help="Number of augmented variants per page (default: 10)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of pieces processed")
    args = ap.parse_args()

    if not SRC_DIR.exists():
        sys.exit(f"No source dir: {SRC_DIR}. Run render.py first.")

    pieces = sorted(p for p in SRC_DIR.iterdir() if p.is_dir())
    if args.limit:
        pieces = pieces[:args.limit]

    total_in, total_out = 0, 0
    for i, piece_dir in enumerate(pieces, start=1):
        pages = sorted(piece_dir.glob("page_*.png"))
        if not pages:
            continue
        out_piece = OUT_DIR / piece_dir.name
        for page in pages:
            augments = augment_one(page, args.variants, out_piece)
            total_in += 1
            total_out += len(augments)
        if i % 50 == 0:
            print(f"[augment] [{i}/{len(pieces)}] {piece_dir.name}: "
                  f"{len(pages)} page(s) → {len(pages) * args.variants} variants", flush=True)

    print(f"\n[augment] done. {total_in} source pages → {total_out} augmented variants", flush=True)


if __name__ == "__main__":
    main()
