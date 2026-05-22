#!/usr/bin/env python3
"""Apply phone-photo-style augmentations to each page PNG.

For each page in ./images/<piece_id>/page_N.png we produce K augmented variants
in ./augmented/<piece_id>/page_N_aug_K.webp (WebP for ~8x smaller files).

Augmentations (random per variant):
  - rotation ±3°
  - perspective warp (slight, simulating phone angle)
  - brightness/contrast jitter
  - Gaussian blur (mild)
  - paper-color overlay (off-white tone)
  - additive noise
  - JPEG-like artifacts (re-encode at lower quality)

Usage:
    python3 augment.py [--variants K] [--limit N]
"""
import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SRC_DIR = Path(__file__).parent / "images"
OUT_DIR = Path(__file__).parent / "augmented"


def random_perspective(img: Image.Image, max_shift: float = 0.04) -> Image.Image:
    """Slight perspective warp simulating off-angle phone photo."""
    w, h = img.size
    # Shift each corner by up to max_shift × dim
    def jitter(x: int, y: int) -> tuple[float, float]:
        return (x + random.uniform(-max_shift, max_shift) * w,
                y + random.uniform(-max_shift, max_shift) * h)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [jitter(*p) for p in src]
    # Solve for perspective coefficients (from PIL docs)
    import numpy as np
    matrix = []
    for s, d in zip(dst, src):
        matrix.append([s[0], s[1], 1, 0, 0, 0, -d[0]*s[0], -d[0]*s[1]])
        matrix.append([0, 0, 0, s[0], s[1], 1, -d[1]*s[0], -d[1]*s[1]])
    A = np.array(matrix, dtype=float)
    B = np.array([p for pt in src for p in pt], dtype=float)
    coeffs = np.linalg.solve(A, B)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs.tolist(),
                          resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def paper_tint(img: Image.Image) -> Image.Image:
    """Multiply with a slight off-white tint to simulate paper color."""
    r = random.randint(245, 255)
    g = random.randint(240, 252)
    b = random.randint(225, 245)
    tint = Image.new("RGB", img.size, (r, g, b))
    return Image.blend(img.convert("RGB"), tint, alpha=random.uniform(0.05, 0.15))


def augment_one(src_png: Path, n_variants: int, out_dir: Path) -> list[Path]:
    """Produce N augmented variants of a single page."""
    img = Image.open(src_png).convert("RGB")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    base_name = src_png.stem  # e.g. "page_1"

    for k in range(n_variants):
        x = img

        # 1) rotation
        angle = random.uniform(-3, 3)
        x = x.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255), expand=False)

        # 2) perspective (50% of the time)
        if random.random() < 0.5:
            x = random_perspective(x, max_shift=0.025)

        # 3) brightness/contrast jitter
        x = ImageEnhance.Brightness(x).enhance(random.uniform(0.85, 1.10))
        x = ImageEnhance.Contrast(x).enhance(random.uniform(0.85, 1.15))

        # 4) mild blur (30% of the time)
        if random.random() < 0.3:
            x = x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))

        # 5) paper tint
        x = paper_tint(x)

        # 6) additive noise (mild)
        if random.random() < 0.5:
            import numpy as np
            arr = np.array(x, dtype=np.int16)
            noise = np.random.normal(0, random.uniform(3, 10), arr.shape).astype(np.int16)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            x = Image.fromarray(arr)

        # Save as WebP (much smaller than PNG)
        out_path = out_dir / f"{base_name}_aug_{k:02d}.webp"
        x.save(out_path, "WEBP", quality=random.randint(75, 92))
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
        print(f"[augment] [{i}/{len(pieces)}] {piece_dir.name}: "
              f"{len(pages)} page(s) → {len(pages) * args.variants} variants", flush=True)

    print(f"\n[augment] done. {total_in} source pages → {total_out} augmented variants", flush=True)


if __name__ == "__main__":
    main()
