import os
from PIL import Image
import cv2
import numpy as np

red_channel_index = 2
green_channel_index = 1
blue_channel_index = 0


def load_depth(image_path):
    png = cv2.imread(image_path)

    red_value = png[:, :, red_channel_index]
    green_value = png[:, :, green_channel_index]
    blue_value = png[:, :, blue_channel_index]

    normalized = (red_value + green_value * 2 ** 8 + blue_value * 2 ** 16) / (2 ** 24 - 1)

    in_meters = normalized * 256.

    return in_meters


def save_depth(depth_image_in_meters, write_path):
    normalized = depth_image_in_meters / 256.
    normalized[normalized >= 1.] = 1.

    encoded = normalized * (2 ** 24 - 1)

    values = np.round(encoded).astype(dtype = np.uint32)

    converted = np.empty(shape = (*values.shape, 3), dtype = np.uint8)
    for i in range(len(values)):
        for j in range(len(values[i])):
            str_bin = bin(values[i, j])[2:].zfill(24)
            converted[i, j, 0] = int(str_bin[-8:], 2)
            converted[i, j, 1] = int(str_bin[8:-8], 2)
            converted[i, j, 2] = int(str_bin[:8], 2)

    cv2.imwrite(write_path, converted)


def load_kitti_depth(filename):
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype = int)
    img_file.close()

    depth = depth_png.astype(np.float) / 256.
    # depth = np.expand_dims(depth, -1)
    return depth


# todo check
def save_kitti_depth(filename, image):
    cv2.imwrite(filename, image * 256.)


def convert_2d_image_to_2d_points(image_2d):
    if len(image_2d.shape) == 2:
        image_2d = image_2d.reshape((*image_2d.shape, 1))
    new_shape = (image_2d.shape[0], image_2d.shape[1], 3)

    indexed_2d = np.empty(new_shape, dtype = np.float32)

    for u in range(image_2d.shape[0]):
        indexed_2d[u, :, 1] = u
    for v in range(image_2d.shape[1]):
        indexed_2d[:, v, 0] = v

    indexed_2d[..., 2] = image_2d[..., 0]
    indexed_2d = indexed_2d.reshape(image_2d.shape[0] * image_2d.shape[1], 3)

    indexed_2d = indexed_2d[indexed_2d[..., 2] != 0]
    return indexed_2d

def convert_points_2d_to_image_2d(filtered_points_2d, image_height, image_width):
    image = np.full((image_height, image_width, 1), dtype = np.float32, fill_value = 0.0)
    for point_2d in filtered_points_2d:
        u = int(round((point_2d[0])))
        v = int(round((point_2d[1])))
        z = point_2d[2]
        image[v, u] = z
    return image
