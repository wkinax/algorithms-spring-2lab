import numpy as np
from PIL import Image

from dct_utils import pad_to_block, blocks_8x8, dct2, idct2, quantize, dequantize
from zigzag_rle_vlc import zigzag_8x8, diff_dc, rle_ac, ZIGZAG_IDX


img = Image.open("data/grey.png").convert("L")
arr = np.array(img)

arr_padded = pad_to_block(arr)
h, w = arr_padded.shape
restored = np.zeros_like(arr_padded)

Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
], dtype=np.int32)

all_dc = []
all_ac = []

for i, j, block in blocks_8x8(arr_padded):

    coeffs = dct2(block)

    qblock = quantize(coeffs, Q)

    zz = zigzag_8x8(qblock)

    dc = zz[0]
    ac = zz[1:]

    all_dc.append(dc)
    all_ac.append(ac)

dc_diffs = diff_dc(all_dc)

ac_rle = [rle_ac(ac) for ac in all_ac]

idx = 0
for i, j, block in blocks_8x8(arr_padded):

    dc = all_dc[idx]
    ac = all_ac[idx]

    zz = np.zeros(64, dtype=np.int32)
    zz[0] = dc
    zz[1:] = ac

    block_q = np.zeros((8, 8), dtype=np.int32)
    for k, (x, y) in enumerate(ZIGZAG_IDX):
        block_q[x, y] = zz[k]

    coeffs = dequantize(block_q, Q)

    block_restored = idct2(coeffs)

    restored[i:i+8, j:j+8] = block_restored
    idx += 1

Image.fromarray(restored[:arr.shape[0], :arr.shape[1]]).save("data/grey_reconstructed.png")

print("DC коэффициенты:", all_dc[:10])
print("DC разности:", dc_diffs[:10])
print("Пример RLE AC:", ac_rle[0][:10])
