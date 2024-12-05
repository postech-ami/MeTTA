# import cv2
# from PIL import Image

# img = cv2.imread("img/circle_chair_rgba.png", cv2.IMREAD_UNCHANGED)
# img = img / 255.
# _img = img[..., :3] * img[..., [3]] + (1 - img[..., [3]])
# cv2.imwrite("wb.png", _img * 255)
# import pdb; pdb.set_trace()
# print(img)


from PIL import Image

png = Image.open("img/circle_chair_rgba.png")
png.load() # required for png.split()

background = Image.new("RGB", png.size, (255, 255, 255))
background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

background.save('foo.jpg', 'JPEG', quality=80)