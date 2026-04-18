from PIL import Image
import os

DATA = "data"

def save_raw(img_path, out_name, img_type, colorspace="RGB"):
    img = Image.open(img_path)
    w, h = img.size
    data = img.tobytes()
    header = f"{img_type};{w};{h};{colorspace};".encode("ascii")

    with open(os.path.join(DATA, out_name), "wb") as f:
        f.write(header)
        f.write(data)

def file_size(path):
    return os.path.getsize(path)

def main():
    images = [
        ("color_image.jpg", "color_image.raw", "RGB"),
        ("grey.png", "grey.raw", "GREY"),
        ("bw_no_dither.png", "bw_no.raw", "BW"),
        ("bw_dither.png", "bw_dither.raw", "BW_DITHER"),
    ]

    for src, raw, t in images:
        src_path = os.path.join(DATA, src)
        raw_path = os.path.join(DATA, raw)

        save_raw(src_path, raw, t)

        s1 = file_size(src_path)
        s2 = file_size(raw_path)

        k = s2 / s1

        print(f"{src:20s} → {s1:10d} байт")
        print(f"{raw:20s} → {s2:10d} байт")
        print(f"Коэф. сжатия (RAW/PNG) = {k:.2f}")
        print()

if __name__ == "__main__":
    main()
