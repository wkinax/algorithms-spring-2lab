import numpy as np


def pad_to_block(img: np.ndarray, block_size: int = 8) -> np.ndarray:
    h, w = img.shape[:2]
    ph = (h + block_size - 1) // block_size * block_size
    pw = (w + block_size - 1) // block_size * block_size
    padded = np.zeros((ph, pw), dtype=img.dtype)
    padded[:h, :w] = img
    return padded


def blocks_8x8(img: np.ndarray):
    h, w = img.shape[:2]
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            yield i, j, img[i:i+8, j:j+8]



def _dct_matrix(n: int = 8) -> np.ndarray:
    C = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        for i in range(n):
            if k == 0:
                alpha = np.sqrt(1 / n)
            else:
                alpha = np.sqrt(2 / n)
            C[k, i] = alpha * np.cos((np.pi * (2 * i + 1) * k) / (2 * n))
    return C


_C8 = _dct_matrix(8)


def dct2(block: np.ndarray) -> np.ndarray:
    block = block.astype(np.float32)
    return _C8 @ block @ _C8.T


def idct2(coeffs: np.ndarray) -> np.ndarray:
    coeffs = coeffs.astype(np.float32)
    return _C8.T @ coeffs @ _C8



def quantize(coeffs: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    return np.round(coeffs / qtable).astype(np.int32)


def dequantize(qblock: np.ndarray, qtable: np.ndarray) -> np.ndarray:
    return (qblock * qtable).astype(np.float32)


def scale_quant_table(qtable: np.ndarray, quality: int) -> np.ndarray:
    # стандартная JPEG‑формула
    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality
    q = np.floor((qtable * S + 50) / 100)
    q[q < 1] = 1
    q[q > 255] = 255
    return q.astype(np.int32)
