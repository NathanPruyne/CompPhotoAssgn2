import argparse
import glob
import time

import numpy as np
import skimage
from tqdm import tqdm
import matplotlib.pyplot as plt

def dark_frame(paths):
    # running sum so only one image is in memory at a time
    total = None
    for path in tqdm(paths, desc="Averaging dark frames"):
        img = skimage.io.imread(path)
        if total is None:
            total = np.zeros(img.shape, dtype=np.float64)
        total += img
    return (total / len(paths)).astype(np.float32)

def ramp_statistics(paths, dark, sample_pixels):
    total = np.zeros(dark.shape, dtype=np.float64)
    total_sq = np.zeros(dark.shape, dtype=np.float64)
    rows, cols, chans = np.array(sample_pixels).T
    samples = np.zeros((len(paths), len(sample_pixels)), dtype=np.float32)

    for n, path in enumerate(tqdm(paths, desc="Ramp statistics")):
        img = skimage.io.imread(path).astype(np.float64) - dark
        total += img
        total_sq += img ** 2
        samples[n] = img[rows, cols, chans]

    N = len(paths)
    mean = total / N
    variance = (total_sq - N * mean ** 2) / (N - 1)
    return mean, variance, samples

def mean_variance_curve(mean, variance):
    # round the means and average the variance of every pixel sharing a rounded mean
    keys = np.rint(mean).astype(np.int64).ravel()
    offset = keys.min()
    counts = np.bincount(keys - offset)
    sums = np.bincount(keys - offset, weights=variance.ravel())
    present = counts > 0
    unique_means = np.nonzero(present)[0] + offset
    return unique_means, sums[present] / counts[present], counts[present]

def plot_histograms(samples, sample_pixels, path):
    fig, axes = plt.subplots(1, len(sample_pixels), figsize=(4 * len(sample_pixels), 3.5))
    axes = np.atleast_1d(axes)
    for ax, values, (row, col, chan) in zip(axes, samples.T, sample_pixels):
        ax.hist(values, bins=15)
        ax.set_title(f"pixel ({row}, {col})")
        ax.set_xlabel("dark-subtracted value")
    axes[0].set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_mean_variance(unique_means, avg_variance, fit_mask, gain, additive, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(unique_means[~fit_mask], avg_variance[~fit_mask], s=2, color="lightgray", label="excluded from fit")
    ax.scatter(unique_means[fit_mask], avg_variance[fit_mask], s=2, label="mean-variance points")
    x = np.array([unique_means[fit_mask].min(), unique_means[fit_mask].max()])
    ax.plot(x, gain * x + additive, color="red", label=f"fit: g = {gain:.3f}, $\\sigma^2_{{add}}$ = {additive:.1f}")
    ax.set_xlabel("mean $\\mu$")
    ax.set_ylabel("variance $\\sigma^2$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Dark frame directory")
    parser.add_argument("output", type=str, help="Output .npy file for the dark frame")
    parser.add_argument("--filetype", type=str, help="File extension to get images from", default="tiff")
    parser.add_argument("--ramp", type=str, help="Ramp image directory for noise calibration (if None only computes the dark frame)", default=None)
    parser.add_argument("--histogram-filename", type=str, help="File to save the pixel histograms to", default="noise_histograms.png")
    parser.add_argument("--mean-var-filename", type=str, help="File to save the mean-variance plot to", default="mean_variance.png")
    parser.add_argument("--fit-min", type=float, help="Smallest rounded mean used in the line fit", default=0)
    parser.add_argument("--fit-max", type=float, help="Largest rounded mean used in the line fit", default=np.inf)
    parser.add_argument("--min-count", type=int, help="Ignore rounded means shared by fewer pixels than this", default=10)
    parser.add_argument("--channel", type=str, choices=["R", "G", "B", "all"], help="Channel used for the mean-variance fit (white balancing scales each channel differently)", default="G")

    args = parser.parse_args()

    frames = sorted(glob.glob(args.input + "/*." + args.filetype))

    dark = dark_frame(frames)

    np.save(args.output, dark)

    print(f"Averaged {len(frames)} frames, shape {dark.shape}")
    print(f"Dark frame mean {dark.mean():.3f}, min {dark.min():.3f}, max {dark.max():.3f}")

    if args.ramp is not None:
        ramp_frames = sorted(glob.glob(args.ramp + "/*." + args.filetype))

        # a few pixels spread along the ramp (which varies left to right), green channel
        height, width = dark.shape[:2]
        sample_pixels = [(height // 2, int(width * f), 1) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]

        start = time.time()
        mean, variance, samples = ramp_statistics(ramp_frames, dark, sample_pixels)
        print(f"Ramp statistics over {len(ramp_frames)} frames took {time.time() - start:.1f}s")

        plot_histograms(samples, sample_pixels, args.histogram_filename)

        if args.channel != "all":
            c = "RGB".index(args.channel)
            mean, variance = mean[..., c], variance[..., c]

        unique_means, avg_variance, counts = mean_variance_curve(mean, variance)

        fit_mask = (unique_means >= args.fit_min) & (unique_means <= args.fit_max) & (counts >= args.min_count)
        gain, additive = np.polyfit(unique_means[fit_mask], avg_variance[fit_mask], 1)

        print(f"Estimated gain g = {gain:.4f}")
        print(f"Estimated additive noise variance = {additive:.4f}")

        plot_mean_variance(unique_means, avg_variance, fit_mask, gain, additive, args.mean_var_filename)
