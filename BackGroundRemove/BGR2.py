from rembg import remove
from PIL import Image

img_path = Image.open('BackGroundRemove/Sporty_test_2.png')

out = remove(img_path)

out.save("Sporty_test2.png")