import argparse
import glob
import os
from fractions import Fraction

import numpy as np
import skimage
from tqdm import tqdm

from cp_hw2 import writeHDR
from q1 import save_preview

def optimal_weights(z, exposure, gain, additive_variance, zmin=0.05, zmax=0.95):
    # eq 15: w = t^2 / (g z + sigma^2_add) on well exposed pixels, z in the same units as the calibration
    z_norm = z / 65535
    mask = (zmin <= z_norm) & (z_norm <= zmax)
    # a negative fitted additive variance can make the noise variance non-positive for small z
    noise_variance = np.maximum(gain * z + additive_variance, 1)
    return np.where(mask, exposure ** 2 / noise_variance, 0).astype(np.float32)

def merge_raw_stack(paths, dark, dark_exposure, gain, additive_variance, lowest_inv_expos=256, logarithmic=False, zmin=0.05, zmax=0.95):

    numerator = np.zeros(dark.shape, np.float32)
    denominator = np.zeros(dark.shape, np.float32)

    epsilon = np.finfo(np.float32).eps

    exposures = 2.0 ** np.arange(len(paths)) / lowest_inv_expos

    for k, path in enumerate(tqdm(paths, desc="Merging")):
        # dark frame scaled to this image's exposure time
        img = skimage.io.imread(path).astype(np.float32) - np.float32(exposures[k] / dark_exposure) * dark

        if k == 0:
            shortest = img

        w = optimal_weights(img, exposures[k], gain, additive_variance, zmin, zmax)

        I_lin = img / 65535

        if logarithmic:
            numerator += w * (np.log(np.maximum(I_lin, 0) + epsilon) - np.float32(np.log(exposures[k])))
        else:
            numerator += w * I_lin / np.float32(exposures[k])

        denominator += w

    valid = denominator > 0
    hdr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid)
    if logarithmic:
        hdr = np.exp(hdr)

    # pixels with no well-exposed values: over-exposed even in the shortest
    # exposure get the max valid value, the rest (always under-exposed) get the min
    invalid = ~valid
    over_exposed = invalid & (shortest / 65535 > zmax)
    hdr[over_exposed] = hdr[valid].max()
    hdr[invalid & ~over_exposed] = hdr[valid].min()

    return hdr

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input directory of the RAW (.tiff) exposure stack")
    parser.add_argument("dark", type=str, help="Dark frame .npy from q5_darkframe.py")
    parser.add_argument("output", type=str, help="Output HDR file")
    parser.add_argument("--gain", type=float, required=True, help="Camera gain g from noise calibration")
    parser.add_argument("--additive-variance", type=float, required=True, help="Additive noise variance from noise calibration")
    parser.add_argument("--dark-shutter", type=str, required=True, help="Shutter speed the dark frame was captured at, e.g. 1/60")
    parser.add_argument("--filetype", type=str, help="File extension to get images from", default="tiff")
    parser.add_argument("--lowest-shutter", type=int, help="Inverse of the lowest shutter speed used", default=256)
    parser.add_argument("--logarithmic", help="Whether to use logarithmic merging", action="store_true")
    parser.add_argument("--preview-scale", type=float, help="Scale applied to the median-normalized HDR for the preview image", default=0.18)

    args = parser.parse_args()

    exposures = sorted(glob.glob(args.input + "/*." + args.filetype))

    dark = np.load(args.dark)

    hdr_result = merge_raw_stack(exposures, dark, float(Fraction(args.dark_shutter)), args.gain, args.additive_variance,
                                 lowest_inv_expos=args.lowest_shutter, logarithmic=args.logarithmic)

    hdr_result = hdr_result / np.median(hdr_result) #Scale

    writeHDR(args.output, hdr_result)

    save_preview(hdr_result, args.preview_scale, os.path.splitext(args.output)[0] + "_preview.png")
