"""Write the synthetic frames/images referenced by examples/data/*.json (no real data involved).

    python examples/make_synthetic_media.py      # creates examples/toy_media/
"""

import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = np.random.default_rng(0)
    frames_dir = os.path.join(HERE, "toy_media", "frames", "clip_000")
    views_dir = os.path.join(HERE, "toy_media", "views")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)
    for t in range(16):  # a red square moving to the right
        image = Image.fromarray((rng.random((360, 640, 3)) * 60 + 100).astype(np.uint8))
        x = 40 + t * 30
        ImageDraw.Draw(image).rectangle([x, 150, x + 60, 210], fill=(220, 30, 30))
        image.save(os.path.join(frames_dir, f"{t:04d}.jpg"))
    for v in range(3):
        image = Image.fromarray((rng.random((480, 640, 3)) * 255).astype(np.uint8))
        image.save(os.path.join(views_dir, f"view_{v}.jpg"))
    print(f"wrote synthetic media to {os.path.join(HERE, 'toy_media')}")


if __name__ == "__main__":
    main()
