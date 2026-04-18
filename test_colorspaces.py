import numpy as np
from PIL import Image
from colorspaces import rgb_to_ycbcr, ycbcr_to_rgb

img = Image.open("data/color_image.jpg").convert("RGB")
arr = np.array(img)

ycbcr = rgb_to_ycbcr(arr)

rgb_restored = ycbcr_to_rgb(ycbcr)

Image.fromarray(rgb_restored).save("data/color_roundtrip.png")

print("файл color_roundtrip.png")
