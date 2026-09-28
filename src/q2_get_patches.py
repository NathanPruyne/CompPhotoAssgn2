import argparse

import numpy as np
import matplotlib.pyplot as plt

from cp_hw2 import readHDR, read_colorchecker_gm

def to_display(hdr, scale):
    img = np.clip(hdr * scale, 0, 1)
    return np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (1 / 2.4) - 0.055) # sRGB gamma

def click_box(ax, prompt):
    ax.set_title(prompt + "\n(right click undoes a point)")
    plt.draw()
    while True:
        points = plt.ginput(2, timeout=0)
        if len(points) == 2:
            break
    (x0, y0), (x1, y1) = points
    return int(min(y0, y1)), int(max(y0, y1)), int(min(x0, x1)), int(max(x0, x1))

def click_patches(hdr, preview_scale, preview_downsample):
    gt = np.stack(read_colorchecker_gm(), axis=-1)  # (4, 6, 3) linear sRGB
    display = to_display(hdr[::preview_downsample, ::preview_downsample], preview_scale)

    fig, (ax_img, ax_ref) = plt.subplots(1, 2, figsize=(14, 8), width_ratios=[4, 1])
    ax_img.imshow(display)
    ax_ref.imshow(to_display(gt, 1))
    ax_ref.set_title("Reference patches")
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])

    # zoom the view in on the checker first, redoing it until the whole checker is in view
    full_xlim, full_ylim = ax_img.get_xlim(), ax_img.get_ylim()
    while True:
        ax_img.set_xlim(full_xlim)
        ax_img.set_ylim(full_ylim)
        r0, r1, c0, c1 = click_box(ax_img, "Zoom: click the top-left and bottom-right corners of the ENTIRE color checker (all 24 patches)")
        ax_img.set_xlim(c0, c1)
        ax_img.set_ylim(r1, r0)
        ax_img.set_title("Press Enter if all 24 patches are visible, or click anywhere to redo the zoom")
        plt.draw()
        if not plt.ginput(1, timeout=0):
            break

    # coords[row, col] = [row0, row1, col0, col1] in full resolution pixels
    coords = np.zeros((4, 6, 4), dtype=np.int64)
    highlight = None

    for row in range(4):
        for col in range(6):
            if highlight is not None:
                highlight.remove()
            highlight = ax_ref.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=3))

            box = click_box(ax_img, f"Patch {row * 6 + col + 1}/24: click two corners of a square inside the patch outlined in red")
            coords[row, col] = np.array(box) * preview_downsample

            b_r0, b_r1, b_c0, b_c1 = box
            ax_img.add_patch(plt.Rectangle((b_c0, b_r0), b_c1 - b_c0, b_r1 - b_r0, fill=False, edgecolor="red"))

    plt.close(fig)

    return coords

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input HDR image")
    parser.add_argument("output", type=str, help="Output .npy file for the patch coordinates")
    parser.add_argument("--preview-scale", type=float, help="Scale applied to the HDR for display", default=0.25)
    parser.add_argument("--preview-downsample", type=int, help="Factor to downsample the displayed image by", default=4)

    args = parser.parse_args()

    hdr = readHDR(args.input)

    coords = click_patches(hdr, args.preview_scale, args.preview_downsample)

    np.save(args.output, coords)
    print(f"Saved patch coordinates to {args.output}")
