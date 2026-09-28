import argparse

import numpy as np
import skimage
import matplotlib.pyplot as plt

def make_ramp(steps=255):
    return np.tile(np.linspace(0, 1, steps), (steps, 1))

def rescale_ramp(ramp, height, width):
    # nearest neighbour so each gray level stays a flat band
    rows = np.arange(height) * ramp.shape[0] // height
    cols = np.arange(width) * ramp.shape[1] // width
    return ramp[rows][:, cols]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=str, help="File to save a page-sized ramp to for printing (if None won't save)", default=None)
    parser.add_argument("--page-width", type=float, help="Page width in inches", default=11)
    parser.add_argument("--page-height", type=float, help="Page height in inches", default=8.5)
    parser.add_argument("--dpi", type=int, help="Print resolution", default=300)
    parser.add_argument("--no-display", action="store_true", help="Don't show the ramp on screen")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen (the window can't be moved until 'f' toggles it off)")

    args = parser.parse_args()

    ramp = make_ramp()

    if args.output is not None:
        page = rescale_ramp(ramp, int(args.page_height * args.dpi), int(args.page_width * args.dpi))
        skimage.io.imsave(args.output, (page * 255).round().astype(np.uint8))
        print(f"Saved {page.shape[1]}x{page.shape[0]} ramp to {args.output}")

    if not args.no_display:
        fig, ax = plt.subplots()
        ax.imshow(ramp, cmap="gray", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        if args.fullscreen:
            fig.canvas.manager.full_screen_toggle()
        else:
            print("Move the window to the screen you want, then press 'f' to toggle fullscreen")
        plt.show()
