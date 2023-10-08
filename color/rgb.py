import cv2
import numpy as np


def to_rgb_image(x, im_size=224, lw=4):

    if x.shape == (2,):
        x = x[0]
    x[:, :2] /= 224 / 200
    x_min, y_min, _, _ = np.min(x, axis=0)
    x_max, y_max, _, _ = np.max(x, axis=0)
    dx = (x_max + x_min)/2
    dy = (y_max + y_min)/2
    for xx in x[:, :1]:
        if xx >= 0:
            xx += 112 - dx
    for yy in x[:, 1:2]:
        if yy >= 0:
            yy += 112 - dy

    ones = np.nonzero(x[:, 2] == 0)[0]
    ones = np.append(ones, 99)
    ones = [x + 1 for x in ones]
    ones = np.insert(ones, 0, 0)

    img = np.zeros((im_size, im_size, 3), dtype=np.uint8)

    img[:, :, 0] = np.zeros([im_size, im_size]) + 255
    img[:, :, 1] = np.zeros([im_size, im_size]) + 255
    img[:, :, 2] = np.zeros([im_size, im_size]) + 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    color = np.array([255, 0, 0])

    for i in range(len(ones) - 1):
        start = ones[i]
        end = ones[i + 1]

        color_step = 51
        if color[0] > 0 and color[2] == 0:
            color[0] -= color_step
            color[1] += color_step

        elif 0.0 < color[1] <= 255:
            color[1] -= color_step
            color[2] += color_step

        elif 0.0 < color[2] <= 255 and color[1] == 0:
            color[0] += color_step
            color[2] -= color_step

        for j in range(start, end-1):

            cv2.line(img, tuple(np.round(x[j, :2]).astype(int)), tuple(np.round(x[j + 1, :2]).astype(int)),
                     tuple(np.round(color).astype(np.uint8).tolist()), lw)

    return img