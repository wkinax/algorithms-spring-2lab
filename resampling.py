import numpy as np

def downsample2(img: np.ndarray) -> np.ndarray:
    return img[::2, ::2, ...]


def upsample2_nearest(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    return out

def linear_interp(x1: float, x2: float, y1: float, y2: float, x: float) -> float:
    if x2 == x1:
        return y1
    t = (x - x1) / (x2 - x1)
    return (1 - t) * y1 + t * y2


def linear_spline(xs: np.ndarray, ys: np.ndarray, x: float) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    i = np.searchsorted(xs, x) - 1
    x1, x2 = xs[i], xs[i+1]
    y1, y2 = ys[i], ys[i+1]
    return linear_interp(x1, x2, y1, y2, x)


def bilinear_interp(
    x1: float, x2: float, y1: float, y2: float,
    z11: float, z12: float, z21: float, z22: float,
    x: float, y: float
) -> float:

    if x2 == x1:
        x2 = x1 + 1e-6
    if y2 == y1:
        y2 = y1 + 1e-6

    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)

    z1 = (1 - tx) * z11 + tx * z21
    z2 = (1 - tx) * z12 + tx * z22
    z = (1 - ty) * z1 + ty * z2
    return z


def resize_bilinear(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:

    img = img.astype(np.float32)
    h, w = img.shape[:2]

    if img.ndim == 2:
        c = 1
        img_ = img[..., None]
    else:
        c = img.shape[2]
        img_ = img

    out = np.zeros((new_h, new_w, c), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            y = (i + 0.5) * h / new_h - 0.5
            x = (j + 0.5) * w / new_w - 0.5

            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            y1 = min(y0 + 1, h - 1)
            x1 = min(x0 + 1, w - 1)

            dy = y - y0
            dx = x - x0

            for ch in range(c):
                z11 = img_[y0, x0, ch]
                z21 = img_[y0, x1, ch]
                z12 = img_[y1, x0, ch]
                z22 = img_[y1, x1, ch]

                val = (
                    z11 * (1 - dx) * (1 - dy) +
                    z21 * dx       * (1 - dy) +
                    z12 * (1 - dx) * dy       +
                    z22 * dx       * dy
                )
                out[i, j, ch] = val

    if c == 1:
        out = out[..., 0]
    return np.clip(out, 0, 255).astype(np.uint8)