import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))  # SITTO

import numpy as np
from render.util import *
import matplotlib.pyplot as plt


def rotate_image(image, rotation_degrees):
    """Rotate an image by the specified number of degrees."""
    rotation_pixels = int((rotation_degrees / 360) * image.shape[1])
    return np.roll(image, rotation_pixels, axis=1)


if __name__ == '__main__':
    # Load the HDR image.
    hdr_image = imageio.imread('data/irrmaps/mud_road_puresky_4k.hdr')

    # Split the image into top and bottom halves.
    # height, width, _ = hdr_image.shape
    # mid = height // 2
    # top_half = hdr_image[:mid, :, :]
    # bottom_half = hdr_image[mid:, :, :]

    # Rotate the top half by 90 degrees.
    # rotated_top_half = rotate_image(top_half, 270)
    save_path = './data/irrmaps/rot_hdr'
    os.makedirs(save_path, exist_ok=True)
    for angle in np.linspace(0.0, 360.0, num=11):
        print(angle)
        rotated_image = rotate_image(hdr_image, angle)
        imageio.imsave(os.path.join(save_path, f'mud_road_puresky_{int(angle):03}.hdr'), rotated_image)
    # rotated_top_half = np.flipud(rotated_top_half)

    # Recombine the image.
    # rotated_hdr_image = np.concatenate([top_half, rotated_top_half], axis=0)

    # Save the manipulated HDR image.
    # imageio.imsave('./data/irrmaps/kloofendal_overcast_2k_edit.hdr', rotated_hdr_image)
    # imageio.imsave('./data/irrmaps/mountain_90.hdr', rotated_image)
