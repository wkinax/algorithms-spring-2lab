import numpy as np

def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:

    rgb = rgb.astype(np.float32)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    Y  =  0.299  * R + 0.587  * G + 0.114  * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5    * B + 128
    Cr =  0.5    * R - 0.4187 * G - 0.0813 * B + 128

    ycbcr = np.stack([Y, Cb, Cr], axis=-1)
    return ycbcr.astype(np.float32)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    Y  = ycbcr[..., 0]
    Cb = ycbcr[..., 1] - 128.0
    Cr = ycbcr[..., 2] - 128.0

    R = Y + 1.402   * Cr
    G = Y - 0.34414 * Cb - 0.71414 * Cr
    B = Y + 1.772   * Cb

    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb
