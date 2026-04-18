from PIL import Image
import numpy as np
import os

from jpeg_codec import (
    encode_image, decode_image,
    write_encoded_file, read_encoded_file
)

DATA = "data"

# Все тестовые изображения
images = [
    "data/Lena.png",
    "data/grey.png",
    "data/color_image.jpg",
    "data/bw_no_dither.png",
    "data/bw_dither.png",
]

# Одно качество для пункта 7
QUALITY = 50

for img_path in images:
    name = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\nProcessing: {name}")

    pil_img = Image.open(img_path)

    # Нормализуем режим
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("L")

    img = np.array(pil_img)

    # Кодирование
    metadata, bit_bytes = encode_image(img, QUALITY)

    # Запись в .bin
    out_bin = f"{DATA}/{name}_q{QUALITY}.bin"
    write_encoded_file(out_bin, metadata, bit_bytes)

    # Чтение
    meta2, bits2 = read_encoded_file(out_bin)

    # Декодирование
    decoded = decode_image(meta2, bits2)

    # Сохранение результата
    out_png = f"{DATA}/{name}_q{QUALITY}.png"
    Image.fromarray(decoded).save(out_png)

    print(f"OK: {name} Q={QUALITY}")
