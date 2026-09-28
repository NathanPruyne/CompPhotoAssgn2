import argparse
import os

import numpy as np

from cp_hw2 import readHDR, writeHDR, read_colorchecker_gm
from q1 import save_preview

def apply_transform(averages, colorchecker, full_image):

    X, *_ = np.linalg.lstsq(averages, colorchecker, rcond=None)
    transform = X.T.astype(np.float32)

    corrected = full_image @ transform[:, :3].T + transform[:, 3]
    return np.maximum(corrected, 0)

def white_balance(image, patches, white_patch=(3, 0)):
    r0, r1, c0, c1 = patches[white_patch]
    white = image[r0:r1, c0:c1].mean(axis=(0, 1))
    return image * (white[1] / white).astype(np.float32)

def patch_averages(hdr, patches):
    #Keep the last as ones
    averages = np.ones((*patches.shape[:2], 4), dtype=np.float32)
    for row in range(patches.shape[0]):
        for col in range(patches.shape[1]):
            r0, r1, c0, c1 = patches[row, col]
            averages[row, col, :-1] = hdr[r0:r1, c0:c1].mean(axis=(0, 1))
    return averages

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input HDR image")
    parser.add_argument("patches", type=str, help="Input .npy file for the patch coordinates")
    parser.add_argument("output", type=str, help="Output HDR image")
    parser.add_argument("--preview-scale", type=float, help="Scale applied to the median-normalized HDR for the preview image", default=0.18)

    args = parser.parse_args()

    hdr = readHDR(args.input)

    patches = np.load(args.patches)

    averages = patch_averages(hdr, patches).reshape(-1, 4)

    colorchecker_colors = np.stack(read_colorchecker_gm(), axis=-1).reshape(-1, 3)

    corrected = apply_transform(averages, colorchecker_colors, hdr)

    balanced = white_balance(corrected, patches)

    writeHDR(args.output, balanced)

    save_preview(balanced, args.preview_scale / np.median(balanced), os.path.splitext(args.output)[0] + "_preview.png")