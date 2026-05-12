#!/usr/bin/env python
"""Create a controlled LR input from an HR image for SR smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter


def parse_size(value: str) -> tuple[int, int]:
    clean = value.lower().replace(' ', '')
    if 'x' not in clean:
        side = int(clean)
        return side, side
    width, height = clean.split('x', 1)
    return int(width), int(height)


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a deterministic LR/HR pair for SR tests.')
    parser.add_argument('--input_image', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--hr_size', default='1024x1024')
    parser.add_argument('--lr_size', default='256x256')
    parser.add_argument('--blur_radius', type=float, default=0.0)
    parser.add_argument('--jpeg_quality', type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hr_size = parse_size(args.hr_size)
    lr_size = parse_size(args.lr_size)
    source = Image.open(args.input_image).convert('RGB')
    cropped = center_crop_square(source)
    hr = cropped.resize(hr_size, Image.Resampling.LANCZOS)
    lr_source = hr
    if args.blur_radius > 0:
        lr_source = lr_source.filter(ImageFilter.GaussianBlur(args.blur_radius))
    lr = lr_source.resize(lr_size, Image.Resampling.BICUBIC)

    hr_path = output_dir / 'hr_reference.png'
    lr_path = output_dir / 'lr_input.png'
    hr.save(hr_path)
    if args.jpeg_quality > 0:
        lr_path = output_dir / 'lr_input.jpg'
        lr.save(lr_path, quality=args.jpeg_quality)
    else:
        lr.save(lr_path)

    metadata = {
        'source_image': str(Path(args.input_image)),
        'source_size': list(source.size),
        'hr_reference': str(hr_path),
        'hr_size': list(hr.size),
        'lr_input': str(lr_path),
        'lr_size': list(lr.size),
        'scale_factor': hr.size[0] / lr.size[0],
        'blur_radius': args.blur_radius,
        'jpeg_quality': args.jpeg_quality,
    }
    metadata_path = output_dir / 'sr_pair_metadata.json'
    with metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
