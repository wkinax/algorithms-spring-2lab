import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from jpeg_codec import (encode_grey, decode_grey, encode_color, decode_color)

DATA = "data"
TEST_IMAGES = [
    ("Lena.png", "LENA2"),
    ("color_image.jpg", "COLOR2"),
    ("grey.png", "GREY2"),
    ("bw_no_dither.png", "BW_NO2"),
    ("bw_dither.png", "BW_DITHER2"),
]
QUALITIES = list(range(10, 100, 10))

def is_color(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[2] == 3

def main():
    results = {}

    for filename, tag in TEST_IMAGES:
        path = os.path.join(DATA, filename)
        img = Image.open(path)

        if img.mode == "RGB":
            arr = np.array(img)
            mode = "COLOR"
        else:
            arr = np.array(img.convert("L"))
            mode = "GREY"

        print(f"\n {tag} ({mode})")

        sizes = []

        for q in QUALITIES:
            print(f"  Q={q}")

            if mode == "COLOR":
                meta, bitstream = encode_color(arr, quality=q)
                rec = decode_color(meta, bitstream)
            else:
                meta, bitstream = encode_grey(arr, quality=q)
                rec = decode_grey(meta, bitstream)

            out_name = f"{tag}_Q{q}.png"
            Image.fromarray(rec).save(os.path.join(DATA, out_name))
            sizes.append(len(bitstream))

        results[tag] = sizes

    plt.figure(figsize=(10, 6))
    for tag, sizes in results.items():
        plt.plot(QUALITIES, sizes, marker="o", label=tag)

    plt.title("Размер битстрима JPEG-like vs качество")
    plt.xlabel("Quality")
    plt.ylabel("Размер битстрима (байты)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DATA, "compression_graph.png"))
    plt.close()

    with open(os.path.join(DATA, "compression_sizes.csv"), "w") as f:
        f.write("Quality;" + ";".join([tag for _, tag in TEST_IMAGES]) + "\n")
        for i, q in enumerate(QUALITIES):
            row = [str(q)]
            for _, tag in TEST_IMAGES:
                row.append(str(results[tag][i]))
            f.write(";".join(row) + "\n")



if __name__ == "__main__":
    main()
