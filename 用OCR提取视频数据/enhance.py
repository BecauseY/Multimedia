from PIL import Image
from PIL import ImageEnhance

img = Image.open('./extract_result/4216.jpg')
# img.show()

# 对比度增强
enh_con = ImageEnhance.Contrast(img)
contrast = 2
img_contrasted = enh_con.enhance(contrast)
# img_contrasted.show()
img_contrasted.save("7975enhance.jpg")
