from __future__ import annotations

import io
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


def apply_stressor(img: Image.Image, stressor: dict) -> Image.Image:
    name = stressor.get('type', 'none')
    if name == 'none':
        return img.copy()
    if name == 'resize_down_up':
        factor = float(stressor.get('factor', 0.5))
        w, h = img.size
        small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))))
        return small.resize((w, h))
    if name == 'gaussian_blur':
        return img.filter(ImageFilter.GaussianBlur(radius=float(stressor.get('radius', 1.0))))
    if name == 'jpeg_compression':
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=int(stressor.get('quality', 50)))
        buffer.seek(0)
        return Image.open(buffer).convert(img.mode)
    if name == 'brightness':
        delta = float(stressor.get('delta', 0))
        factor = max(0.0, 1.0 + delta / 100.0)
        return ImageEnhance.Brightness(img).enhance(factor)
    if name == 'contrast':
        return ImageEnhance.Contrast(img).enhance(float(stressor.get('factor', 1.0)))
    if name == 'random_occlusion':
        out = img.copy()
        draw = ImageDraw.Draw(out)
        w, h = img.size
        frac = float(stressor.get('area_fraction', 0.1))
        rect_w = int(w * frac ** 0.5)
        rect_h = int(h * frac ** 0.5)
        x0 = random.randint(0, max(0, w - rect_w))
        y0 = random.randint(0, max(0, h - rect_h))
        draw.rectangle([x0, y0, x0 + rect_w, y0 + rect_h], fill=0)
        return out
    raise ValueError(f"Unknown stressor type: {name}")
