import os
import cv2

img = cv2.imread("./img/teddy_rgba.png", cv2.IMREAD_UNCHANGED)
mask = img[..., 3]
mask[mask > 0.5] = 255
new_img = img[..., :3] * (mask[..., None] / 255) + (255 - mask)[..., None]
cv2.imwrite("teddy_wb.png", new_img)
