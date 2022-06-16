import torch

print(torch.__version__, torch.cuda.is_available())

from PIL import Image
import os

basewidth = 1024
path = 'data/content'
flist = os.listdir(path)

for index1 in range(0, len(flist)):
    image_file = path + os.sep + flist[index1]
    base_name = os.path.basename(image_file)
    img = Image.open(image_file)
    hsize = min(img.width, img.height)
    if img.width > img.height:
        padding = int((img.width - img.height) / 2)
        img = img.crop((padding, 0, padding + hsize, hsize))
    else:
        padding = int((img.height - img.width) / 2)
        img = img.crop((0, padding, hsize, padding + hsize))

    print(base_name, hsize, img.width, img.height)
    img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
    img.save(image_file)
