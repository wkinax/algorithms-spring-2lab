import numpy as np
from PIL import Image
from colorspaces import rgb_to_ycbcr, ycbcr_to_rgb
import struct
import pickle

from dct_utils import (
    pad_to_block, blocks_8x8,
    dct2, idct2,
    quantize, dequantize,
    scale_quant_table,
)
from zigzag_rle_vlc import (
    ZIGZAG_IDX,
    zigzag_8x8,
    diff_dc,
    rle_ac,
    vlc_dc,
    vlc_ac,
    DC_LUMA_HUFF,
    AC_LUMA_HUFF,
)


def bits_to_bytes(bitstring: str) -> bytes:
    if len(bitstring) % 8 != 0:
        bitstring += "0" * (8 - len(bitstring) % 8)
    return int(bitstring, 2).to_bytes(len(bitstring) // 8, "big")


def bytes_to_bits(data: bytes) -> str:
    return "".join(f"{b:08b}" for b in data)


Q_LUMA_STD = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99],
], dtype=np.int32)

def encode_channel_8x8(img_arr: np.ndarray, Q: np.ndarray):
    padded = pad_to_block(img_arr)
    ph, pw = padded.shape

    all_dc = []
    all_ac_rle = []

    for i, j, block in blocks_8x8(padded):
        block_f = block.astype(np.float32) - 128.0
        coeffs = dct2(block_f)
        qblock = quantize(coeffs, Q)
        zz = zigzag_8x8(qblock)

        dc = int(zz[0])
        ac = zz[1:].astype(int)

        all_dc.append(dc)
        all_ac_rle.append(rle_ac(ac))

    dc_diffs = diff_dc(all_dc)

    bitstream = ""
    for diff in dc_diffs:
        bitstream += vlc_dc(int(diff), DC_LUMA_HUFF)
    for rle_pairs in all_ac_rle:
        bitstream += vlc_ac(rle_pairs, AC_LUMA_HUFF)

    return padded.shape, all_dc, all_ac_rle, bitstream

def decode_channel_8x8(ph: int, pw: int, Q: np.ndarray,
                       num_blocks: int, bits: str) -> np.ndarray:
    DC_INV = invert_huff_table_dc(DC_LUMA_HUFF)
    AC_INV = invert_huff_table_ac(AC_LUMA_HUFF)

    def read_huff_dc(pos):
        code = ""
        while True:
            code += bits[pos]
            pos += 1
            if code in DC_INV:
                return DC_INV[code], pos

    def read_huff_ac(pos):
        code = ""
        while True:
            code += bits[pos]
            pos += 1
            if code in AC_INV:
                run, size = AC_INV[code]
                return run, size, pos

    def read_magnitude(size, pos):
        if size == 0:
            return 0, pos
        mag_bits = bits[pos:pos + size]
        pos += size
        if mag_bits[0] == "1":
            val = int(mag_bits, 2)
        else:
            val = int(mag_bits, 2) - ((1 << size) - 1)
        return val, pos

    dc_vals = []
    pos = 0
    for _ in range(num_blocks):
        size, pos = read_huff_dc(pos)
        diff, pos = read_magnitude(size, pos)
        if not dc_vals:
            dc_vals.append(diff)
        else:
            dc_vals.append(dc_vals[-1] + diff)

    ac_all = []
    for _ in range(num_blocks):
        ac = []
        while len(ac) < 63:
            run, size, pos = read_huff_ac(pos)
            if (run, size) == (0, 0):
                ac.extend([0] * (63 - len(ac)))
                break
            if (run, size) == (15, 0):
                ac.extend([0] * 16)
                continue
            val, pos = read_magnitude(size, pos)
            ac.extend([0] * run)
            ac.append(val)
        ac_all.append(ac[:63])

    restored = np.zeros((ph, pw), dtype=np.float32)
    idx_block = 0

    for i, j, _ in blocks_8x8(restored):
        dc = dc_vals[idx_block]
        ac = ac_all[idx_block]
        idx_block += 1

        zz = np.zeros(64, dtype=np.int32)
        zz[0] = dc
        zz[1:] = np.array(ac, dtype=np.int32)

        qblock = np.zeros((8, 8), dtype=np.int32)
        for k, (x, y) in enumerate(ZIGZAG_IDX):
            qblock[x, y] = zz[k]

        coeffs = dequantize(qblock, Q)
        block = idct2(coeffs) + 128.0

        restored[i:i+8, j:j+8] = block

    restored = np.clip(restored, 0, 255).astype(np.uint8)
    return restored, pos


def encode_grey(img_arr: np.ndarray, quality: int = 50):
    h, w = img_arr.shape
    padded = pad_to_block(img_arr)
    ph, pw = padded.shape

    Q = scale_quant_table(Q_LUMA_STD, quality)

    all_dc = []
    all_ac_rle = []

    for i, j, block in blocks_8x8(padded):
        block_f = block.astype(np.float32) - 128.0
        coeffs = dct2(block_f)
        qblock = quantize(coeffs, Q)
        zz = zigzag_8x8(qblock)

        dc = int(zz[0])
        ac = zz[1:].astype(int)

        all_dc.append(dc)
        all_ac_rle.append(rle_ac(ac))

    dc_diffs = diff_dc(all_dc)

    bitstream = ""
    for diff in dc_diffs:
        bitstream += vlc_dc(int(diff), DC_LUMA_HUFF)
    for rle_pairs in all_ac_rle:
        bitstream += vlc_ac(rle_pairs, AC_LUMA_HUFF)

    bit_bytes = bits_to_bytes(bitstream)

    metadata = {
        "width": w,
        "height": h,
        "padded_width": pw,
        "padded_height": ph,
        "quality": quality,
        "qtable": Q.astype(np.int32),
        "num_blocks": len(all_dc),
    }

    return metadata, bit_bytes


def invert_huff_table_dc(dc_huff: dict) -> dict:
    return {v: k for k, v in dc_huff.items()}


def invert_huff_table_ac(ac_huff: dict) -> dict:
    return {v: k for k, v in ac_huff.items()}


def decode_grey(metadata: dict, bit_bytes: bytes) -> np.ndarray:
    w = metadata["width"]
    h = metadata["height"]
    ph = metadata["padded_height"]
    pw = metadata["padded_width"]
    Q = metadata["qtable"].astype(np.int32)

    bits = bytes_to_bits(bit_bytes)

    DC_INV = invert_huff_table_dc(DC_LUMA_HUFF)
    AC_INV = invert_huff_table_ac(AC_LUMA_HUFF)

    def read_huff_dc(pos):
        code = ""
        while True:
            code += bits[pos]
            pos += 1
            if code in DC_INV:
                return DC_INV[code], pos

    def read_huff_ac(pos):
        code = ""
        while True:
            code += bits[pos]
            pos += 1
            if code in AC_INV:
                run, size = AC_INV[code]
                return run, size, pos

    def read_magnitude(size, pos):
        if size == 0:
            return 0, pos
        mag_bits = bits[pos:pos + size]
        pos += size
        if mag_bits[0] == "1":
            val = int(mag_bits, 2)
        else:
            val = int(mag_bits, 2) - ((1 << size) - 1)
        return val, pos

    dc_vals = []
    pos = 0
    for _ in range(metadata["num_blocks"]):
        size, pos = read_huff_dc(pos)
        diff, pos = read_magnitude(size, pos)
        if not dc_vals:
            dc_vals.append(diff)
        else:
            dc_vals.append(dc_vals[-1] + diff)

    ac_all = []
    for _ in range(metadata["num_blocks"]):
        ac = []
        while len(ac) < 63:
            run, size, pos = read_huff_ac(pos)
            if (run, size) == (0, 0):
                ac.extend([0] * (63 - len(ac)))
                break
            if (run, size) == (15, 0):
                ac.extend([0] * 16)
                continue
            val, pos = read_magnitude(size, pos)
            ac.extend([0] * run)
            ac.append(val)
        ac_all.append(ac[:63])

    restored = np.zeros((ph, pw), dtype=np.float32)
    idx_block = 0

    for i, j, _ in blocks_8x8(restored):
        dc = dc_vals[idx_block]
        ac = ac_all[idx_block]
        idx_block += 1

        zz = np.zeros(64, dtype=np.int32)
        zz[0] = dc
        zz[1:] = np.array(ac, dtype=np.int32)

        qblock = np.zeros((8, 8), dtype=np.int32)
        for k, (x, y) in enumerate(ZIGZAG_IDX):
            qblock[x, y] = zz[k]

        coeffs = dequantize(qblock, Q)
        block = idct2(coeffs) + 128.0

        restored[i:i+8, j:j+8] = block

    restored = np.clip(restored, 0, 255).astype(np.uint8)
    return restored[:h, :w]

def encode_color(rgb_arr: np.ndarray, quality: int = 50):
    h, w, _ = rgb_arr.shape

    ycbcr = rgb_to_ycbcr(rgb_arr)
    Y = ycbcr[..., 0]
    Cb = ycbcr[..., 1]
    Cr = ycbcr[..., 2]

    Q = scale_quant_table(Q_LUMA_STD, quality)

    (phY, pwY), dcY, acY, bitsY = encode_channel_8x8(Y, Q)
    (phCb, pwCb), dcCb, acCb, bitsCb = encode_channel_8x8(Cb, Q)
    (phCr, pwCr), dcCr, acCr, bitsCr = encode_channel_8x8(Cr, Q)

    meta = {
        "width": w,
        "height": h,
        "padded_width_Y": pwY,
        "padded_height_Y": phY,
        "padded_width_Cb": pwCb,
        "padded_height_Cb": phCb,
        "padded_width_Cr": pwCr,
        "padded_height_Cr": phCr,
        "quality": quality,
        "qtable": Q.astype(np.int32),
        "num_blocks_Y": len(dcY),
        "num_blocks_Cb": len(dcCb),
        "num_blocks_Cr": len(dcCr),
    }

    bits_all = bitsY + bitsCb + bitsCr
    bit_bytes = bits_to_bytes(bits_all)

    meta["len_bits_Y"] = len(bitsY)
    meta["len_bits_Cb"] = len(bitsCb)
    meta["len_bits_Cr"] = len(bitsCr)

    return meta, bit_bytes

def decode_color(meta: dict, bit_bytes: bytes) -> np.ndarray:
    w = meta["width"]
    h = meta["height"]

    Q = meta["qtable"].astype(np.int32)

    bits = bytes_to_bits(bit_bytes)

    lenY = meta["len_bits_Y"]
    lenCb = meta["len_bits_Cb"]
    lenCr = meta["len_bits_Cr"]

    bitsY = bits[:lenY]
    bitsCb = bits[lenY:lenY+lenCb]
    bitsCr = bits[lenY+lenCb:lenY+lenCb+lenCr]

    phY = meta["padded_height_Y"]
    pwY = meta["padded_width_Y"]
    phCb = meta["padded_height_Cb"]
    pwCb = meta["padded_width_Cb"]
    phCr = meta["padded_height_Cr"]
    pwCr = meta["padded_width_Cr"]

    numY = meta["num_blocks_Y"]
    numCb = meta["num_blocks_Cb"]
    numCr = meta["num_blocks_Cr"]

    Y_rec, _ = decode_channel_8x8(phY, pwY, Q, numY, bitsY)
    Cb_rec, _ = decode_channel_8x8(phCb, pwCb, Q, numCb, bitsCb)
    Cr_rec, _ = decode_channel_8x8(phCr, pwCr, Q, numCr, bitsCr)

    Y_rec = Y_rec[:h, :w].astype(np.float32)
    Cb_rec = Cb_rec[:h, :w].astype(np.float32)
    Cr_rec = Cr_rec[:h, :w].astype(np.float32)

    ycbcr_rec = np.stack([Y_rec, Cb_rec, Cr_rec], axis=-1)
    rgb_rec = ycbcr_to_rgb(ycbcr_rec)
    return rgb_rec

def write_encoded_file(path: str, metadata: dict, bit_bytes: bytes):
    with open(path, "wb") as f:
        meta_bytes = pickle.dumps(metadata)

        f.write(struct.pack("I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(struct.pack("I", len(bit_bytes)))
        f.write(bit_bytes)


def read_encoded_file(path: str):
    with open(path, "rb") as f:
        meta_len = struct.unpack("I", f.read(4))[0]
        meta_bytes = f.read(meta_len)
        metadata = pickle.loads(meta_bytes)

        bit_len = struct.unpack("I", f.read(4))[0]
        bit_bytes = f.read(bit_len)

    return metadata, bit_bytes

def encode_image(img_arr: np.ndarray, quality: int):

    if img_arr.ndim == 2:
        return encode_grey(img_arr, quality)
    elif img_arr.ndim == 3:
        return encode_color(img_arr, quality)
    else:
        raise ValueError("Unsupported image format")

def decode_image(metadata: dict, bit_bytes: bytes):

    if "num_blocks_Y" in metadata:
        return decode_color(metadata, bit_bytes)
    else:
        return decode_grey(metadata, bit_bytes)