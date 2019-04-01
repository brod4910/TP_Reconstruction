import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

image_datas = ['000000180792.jpg', '000000180792.png', './val2017_gt/000000222094.jpg', '000000222094.png']

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(Image.open(image_datas[0]))
axarr[0,1].imshow(Image.open(image_datas[1]))
axarr[1,0].imshow(Image.open(image_datas[2]))
axarr[1,1].imshow(Image.open(image_datas[3]))

f.show()
plt.pause(10)