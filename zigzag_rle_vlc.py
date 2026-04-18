import numpy as np

ZIGZAG_IDX = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]


def zigzag_8x8(block: np.ndarray) -> np.ndarray:
    out = np.zeros(64, dtype=np.int32)
    for k, (i, j) in enumerate(ZIGZAG_IDX):
        out[k] = block[i, j]
    return out



def diff_dc(dc_list):
    diffs = []
    prev = 0
    for dc in dc_list:
        diffs.append(dc - prev)
        prev = dc
    return diffs



def magnitude_size(val: int) -> int:
    if val == 0:
        return 0
    return int(np.floor(np.log2(abs(val))) + 1)


def rle_ac(ac: np.ndarray):

    res = []
    run = 0
    for v in ac:
        if v == 0:
            run += 1
            if run == 16:
                res.append((15, 0, 0))  # ZRL
                run = 0
        else:
            size = magnitude_size(v)
            res.append((run, size, v))
            run = 0
    # EOB
    if run > 0 or len(res) == 0:
        res.append((0, 0, 0))
    return res



def magnitude_bits(val: int, size: int) -> str:
    if size == 0:
        return ""
    if val >= 0:
        return f"{val:0{size}b}"
    max_val = (1 << size) - 1
    code = max_val + val  # val < 0
    return f"{code:0{size}b}"


DC_LUMA_HUFF = {
    0: "00",
    1: "010",
    2: "011",
    3: "100",
    4: "101",
    5: "110",
    6: "1110",
    7: "11110",
    8: "111110",
    9: "1111110",
    10: "11111110",
    11: "111111110",
}

AC_LUMA_HUFF = {
    (0,0): "1010",            # EOB
    (15,0): "11111111001",    # ZRL

    (0,1): "00",
    (0,2): "01",
    (0,3): "100",
    (0,4): "1011",
    (0,5): "11010",
    (0,6): "1111000",
    (0,7): "11111000",
    (0,8): "1111110110",
    (0,9): "1111111110000010",
    (0,10): "1111111110000011",

    (1,1): "1100",
    (1,2): "11011",
    (1,3): "1111001",
    (1,4): "111110110",
    (1,5): "11111110110",
    (1,6): "1111111110000100",
    (1,7): "1111111110000101",
    (1,8): "1111111110000110",
    (1,9): "1111111110000111",
    (1,10): "1111111110001000",
}

for run in range(0, 16):
    for size in range(1, 11):
        if (run, size) not in AC_LUMA_HUFF:
            AC_LUMA_HUFF[(run, size)] = "11111111111" + f"{run:04b}{size:04b}"



def vlc_dc(diff: int, dc_huff: dict) -> str:
    size = magnitude_size(diff)
    huff = dc_huff[size]
    mag = magnitude_bits(diff, size)
    return huff + mag


def vlc_ac(rle_pairs, ac_huff: dict) -> str:
    bits = ""
    for run, size, val in rle_pairs:
        if (run, size) == (0, 0):
            bits += ac_huff[(0, 0)]
            break
        if (run, size) == (15, 0):
            bits += ac_huff[(15, 0)]
            continue
        bits += ac_huff[(run, size)]
        bits += magnitude_bits(val, size)
    return bits
