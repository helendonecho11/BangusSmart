"""Draw SYNTHETIC silhouettes and a known-size board; never real-fish validation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def make_scene(lengths_cm=(15.0,), pixels_per_cm=24.0, square_size_cm=1.0,
               angle_degrees=0.0, reference=True, blur=0):
    """Return image and exact drawing landmarks (snout to tail fork)."""
    ppcm = float(pixels_per_cm)
    square = int(round(square_size_cm * ppcm))
    width = max(1000, int(max(lengths_cm) * ppcm + 200))
    height = max(650, 320 + 180 * len(lengths_cm))
    image = np.full((height, width, 3), 245, np.uint8)
    if reference:
        for row in range(6):
            for col in range(8):
                x, y = 50 + col * square, 35 + row * square
                cv2.rectangle(image, (x, y), (x + square - 1, y + square - 1),
                              (15, 15, 15) if (row + col) % 2 == 0 else (255, 255, 255), -1)
    truths = []
    # Body and forked tail. Coordinates normalized to exact snout/fork length 1.
    shape = np.array([[0, 0], [.05, -.07], [.18, -.12], [.42, -.115],
                      [.62, -.08], [.80, -.03], [.88, -.025], [1.1, -.13],
                      [1, 0], [1.1, .13], [.88, .025], [.80, .03], [.62, .08],
                      [.42, .115], [.18, .12], [.05, .07]], float)
    theta = np.deg2rad(angle_degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for index, length in enumerate(lengths_cm):
        origin = np.array([120., 340. + index * 180])
        points = np.rint((shape * length * ppcm) @ rotation.T + origin).astype(np.int32)
        cv2.fillPoly(image, [points], (48, 63, 70))
        truths.append({"length_cm": float(length), "snout_px": origin.tolist(),
                       "fork_px": ((np.array([length * ppcm, 0.]) @ rotation.T) + origin).tolist()})
    if blur:
        image = cv2.GaussianBlur(image, (blur | 1, blur | 1), 0)
    return image, {"synthetic": True, "pixels_per_cm": ppcm, "square_size_cm": square_size_cm,
                   "fish": truths, "note": "Drawn geometry only; not real bangus accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/synthetic_media'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (length, angle, scale) in enumerate([(10, 0, 24), (15, 0, 24), (18, 15, 24), (15, -10, 30)]):
        image, truth = make_scene((length,), scale, angle_degrees=angle)
        cv2.imwrite(str(args.output / f'synthetic_{index}.png'), image)
        (args.output / f'synthetic_{index}.json').write_text(json.dumps(truth, indent=2))
    print(json.dumps({"synthetic": True, "generated_images": 4, "folder": str(args.output)}))


if __name__ == '__main__':
    main()
