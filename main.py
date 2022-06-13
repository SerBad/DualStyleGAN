import torch

print(torch.__version__, torch.cuda.is_available())

from PIL import Image
import os

basewidth = 1024
path = 'data/head2/images/train'
flist = os.listdir(path)

for index1 in range(0, len(flist)):
    image_file = path + os.sep + flist[index1]
    base_name = os.path.basename(image_file)
    img = Image.open(image_file)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(image_file)



