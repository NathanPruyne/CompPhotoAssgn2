import argparse
import glob
import os
from fractions import Fraction

import numpy as np
import skimage
from tqdm import tqdm
import matplotlib.pyplot as plt

from cp_hw2 import writeHDR

def downsample(image, scale):
    return image[..., ::scale, ::scale, :]

def save_preview(hdr, scale, path):
    img = np.clip(hdr * scale, 0, 1)
    img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * img ** (1 / 2.4) - 0.055) # sRGB gamma
    skimage.io.imsave(path, (img * 255).astype(np.uint8))

def weighted(arr, weight_scheme, exposure=None, scale=1, zmin=0.05, zmax=0.95, gain=None, additive_variance=None, raw_scale=65535):

    if scale != 1:
        arr = arr.astype(np.float32)
        arr = arr / scale

    mask = (zmin <= arr) & (arr <= zmax)

    if weight_scheme == "uniform":
        weight = 1
    elif weight_scheme == "tent":
        weight = np.minimum(arr, 1 - arr)
    elif weight_scheme == "gaussian":
        weight = np.exp(-4 * ((arr - 0.5) ** 2) / (0.5 ** 2))
    elif weight_scheme == "photon":
        if exposure is None:
            raise ValueError("Can't do photon weighting with exposure None")
        weight = np.broadcast_to(exposure, arr.shape).astype(np.float32)
    elif weight_scheme == "optimal":
        if exposure is None or gain is None or additive_variance is None:
            raise ValueError("Optimal weighting needs exposure, gain, and additive_variance")
        noise_variance = np.maximum(gain * arr * raw_scale + additive_variance, 1)
        weight = (exposure ** 2 / noise_variance).astype(np.float32)
    else:
        raise ValueError(f"Bad weight scheme {weight_scheme}")

    weighted_arr = np.where(mask, weight, 0)

    return weighted_arr
        

def linearize(image_stack, weight_scheme, lambda_val=100, lowest_inv_expos=2048, g_plot=None):

    num_expos = image_stack.shape[0]
    expos_stack = image_stack.reshape(num_expos, -1)
    pix_per_expo = expos_stack.shape[1]

    exposures = (2.0 ** np.arange(num_expos) / lowest_inv_expos)[:, None]

    num_data = num_expos * pix_per_expo
    A = np.zeros((num_data + 255, 256 + pix_per_expo), dtype=np.float32)
    b = np.zeros(A.shape[0], dtype=np.float32)

    w_Ikij = weighted(expos_stack, weight_scheme, exposure=exposures, scale=255).ravel()
    rows = np.arange(num_data)
    pixels = np.tile(np.arange(pix_per_expo), num_expos)
    A[rows, expos_stack.ravel()] = w_Ikij
    A[rows, 256 + pixels] = w_Ikij * -1
    b[:num_data] = w_Ikij * np.broadcast_to(np.log(exposures), expos_stack.shape).ravel()

    z = np.arange(1, 255)
    if weight_scheme == "photon":
        w_z = np.ones(254)
    else:
        w_z = weighted(z, weight_scheme, scale=255, zmin=0, zmax=1)
    smoothness_weight = np.sqrt(lambda_val) * w_z
    rows = num_data + np.arange(254)
    A[rows, z - 1] = smoothness_weight
    A[rows, z] = -2 * smoothness_weight
    A[rows, z + 1] = smoothness_weight

    A[-1, 128] = 1

    v, *_ = np.linalg.lstsq(A, b)

    g = v[:256]

    if g_plot is not None:
        plt.plot(np.arange(256), g)
        plt.title(f"g curve for weighting scheme {weight_scheme}")
        plt.savefig(g_plot, dpi=150)
        plt.close()

    return g

def merge_stack(exposure_stack, weighting_scheme, type, linearizer=None, lowest_inv_expos=2048, logarithmic=False, zmin=0.05, zmax=0.95,
                dark=None, dark_exposure=None, gain=None, additive_variance=None):

    final_shape = exposure_stack.shape[1:]
    numerator = np.zeros(final_shape, np.float32)
    denominator = np.zeros(final_shape, np.float32)

    epsilon = np.finfo(np.float32).eps

    exposures = (2.0 ** np.arange(exposure_stack.shape[0]) / lowest_inv_expos)[:, None]
    is_jpg = type == "jpg"

    for k, img in enumerate(tqdm(exposure_stack)):
        if is_jpg:
            I_ldr = img / 255
            I_lin = np.exp(linearizer[img])
        else:
            if dark is not None:
                # dark frame scaled to this image's exposure time
                img = img.astype(np.float32) - np.float32(exposures[k, 0] / dark_exposure) * dark
            I_ldr = img / 65535
            I_lin = I_ldr

        w_Idlr = weighted(I_ldr, weighting_scheme, exposure=exposures[k], zmin=zmin, zmax=zmax, gain=gain, additive_variance=additive_variance)

        if logarithmic:
            # dark subtraction can leave negative values, which get zero weight but would still make NaNs
            numerator += w_Idlr * (np.log(np.maximum(I_lin, 0) + epsilon) - np.log(exposures[k]))
        else:
            numerator += w_Idlr * I_lin / exposures[k]
        
        denominator += w_Idlr

    valid = denominator > 0
    hdr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=valid)
    if logarithmic:
        hdr = np.exp(hdr)

    invalid = ~valid
    over_exposed = invalid & (exposure_stack[0] / (255 if is_jpg else 65535) > zmax)
    hdr[over_exposed] = hdr[valid].max()
    hdr[invalid & ~over_exposed] = hdr[valid].min()

    return hdr
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input directory of HDR image stack")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument("--filetype", type=str, help="File extension to get images from", default="jpg")
    parser.add_argument("--weight", type=str, help="Weighting scheme to use", default="uniform")
    parser.add_argument("--downsample", type=int, help="Factor to downsample by", default=200)
    parser.add_argument("--g-plot-filename", type=str, help="File to save plot of g curve to (if None won't save curve)", default=None)
    parser.add_argument("--linearize-lambda", type=int, help="Lambda value for linearization", default=100)
    parser.add_argument("--lowest-shutter", type=int, help="Inverse of the lowest shutter speed used (defaults to 2048 for sample photo)", default=2048)
    parser.add_argument("--logarithmic", help="Whether to use logarithmic merging", action="store_true")
    parser.add_argument("--preview-scale", type=float, help="Scale applied to the median-normalized HDR for the preview image", default=0.18)
    parser.add_argument("--dark", type=str, help="Dark frame .npy from q5_darkframe.py to subtract from TIFFs (if None won't subtract)", default=None)
    parser.add_argument("--dark-shutter", type=str, help="Shutter speed the dark frame was captured at, e.g. 1/60", default=None)
    parser.add_argument("--gain", type=float, help="Camera gain g from noise calibration (for optimal weights)", default=None)
    parser.add_argument("--additive-variance", type=float, help="Additive noise variance from noise calibration (for optimal weights)", default=None)

    args = parser.parse_args()

    if args.dark is not None and args.dark_shutter is None:
        parser.error("--dark needs --dark-shutter")
    if args.weight == "optimal" and (args.gain is None or args.additive_variance is None):
        parser.error("--weight optimal needs --gain and --additive-variance")

    exposures = sorted(glob.glob(args.input + "/*." + args.filetype))

    stackarr = None
    

    for i, exposure in enumerate(tqdm(exposures, desc="Loading images")):

        imarr = skimage.io.imread(exposure)

        if stackarr is None:
            stackarr = np.empty((len(exposures), *imarr.shape), dtype=np.uint16)

        stackarr[i] = imarr

    print(stackarr.shape)

    if args.filetype == "jpg":
        print("Linearizing rendered images")
        g = linearize(downsample(stackarr, args.downsample), args.weight, lambda_val=args.linearize_lambda, lowest_inv_expos=args.lowest_shutter, g_plot=args.g_plot_filename)
    else:
        g = None

    print("Merging")
    dark = np.load(args.dark) if args.dark is not None else None
    dark_exposure = float(Fraction(args.dark_shutter)) if args.dark_shutter is not None else None

    hdr_result = merge_stack(stackarr, args.weight, args.filetype, linearizer=g, lowest_inv_expos=args.lowest_shutter, logarithmic=args.logarithmic,
                             dark=dark, dark_exposure=dark_exposure, gain=args.gain, additive_variance=args.additive_variance)

    hdr_result = hdr_result / np.median(hdr_result) #Scale

    writeHDR(args.output, hdr_result)

    save_preview(hdr_result, args.preview_scale, os.path.splitext(args.output)[0] + "_preview.png")
