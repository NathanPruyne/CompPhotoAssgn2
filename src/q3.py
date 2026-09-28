import argparse

import numpy as np
import matplotlib.pyplot as plt

from cp_hw2 import readHDR, lRGB2XYZ, XYZ2lRGB
from q1 import save_preview

def tonemap(hdrimage, luminance=False, burn=0.95, key=0.15):

    if luminance:
        luminance_image = lRGB2XYZ(hdrimage)
        tonemap_chans = luminance_image[:, :, 1:2]
    else:
        tonemap_chans = hdrimage

    epsilon = 1e-4

    I_m = np.exp(np.mean(np.log(tonemap_chans + epsilon), dtype=np.float64))

    I_tilde = key / I_m * tonemap_chans

    I_tildewhite = burn * np.max(I_tilde)

    I_TM = I_tilde * (1 + I_tilde / (I_tildewhite) ** 2) / (1 + I_tilde)

    if luminance:
        X, Y, Z = luminance_image[..., 0], luminance_image[..., 1], luminance_image[..., 2]
        s = X + Y + Z
        valid = s > 0
        x = np.where(valid, X / np.where(valid, s, 1), 0)
        y = np.where(valid, Y / np.where(valid, s, 1), 0)

        Y_TM = I_TM[..., 0]
        y_safe = np.where(y > 0, y, 1)
        X_TM = np.where(y > 0, x * Y_TM / y_safe, 0)
        Z_TM = np.where(y > 0, (1 - x - y) * Y_TM / y_safe, 0)

        return XYZ2lRGB(np.stack([X_TM, Y_TM, Z_TM], axis=-1))
    else:
        return I_TM

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input HDR image")
    parser.add_argument("output", type=str, help="Output tonemapped image")
    parser.add_argument("--luminance", action="store_true", help="Use luminance-based tonemapping (else RGB)")
    parser.add_argument("--burn", type=float, default=0.95, help="Burn value")
    parser.add_argument("--key", type=float, default=0.15, help="Key value")

    args = parser.parse_args()

    hdr = readHDR(args.input)

    output_image = tonemap(hdr, luminance=args.luminance, burn=args.burn, key=args.key)

    save_preview(output_image, 1, args.output)