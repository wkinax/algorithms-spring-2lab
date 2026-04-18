import numpy as np
from PIL import Image

color = Image.open("data/color_image.jpg")

gray = color.convert("L")
gray.save("data/grey.png")

arr = np.array(gray)
bw_arr = np.round(arr / 255) * 255
bw_no_dither = Image.fromarray(bw_arr.astype("uint8"))
bw_no_dither.save("data/bw_no_dither.png")

bw_dither = gray.convert("1")
bw_dither.save("data/bw_dither.png")
