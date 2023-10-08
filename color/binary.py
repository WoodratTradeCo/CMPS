import cv2
import numpy as np


def to_binary_image(x, im_size=224, lw=4):

    # 二维数组所有行和前两列
    if x.shape == (2,):
        x = x[0]

    x[:, :2] /= 224 / 200

    ones = np.nonzero(x[:, 2] == 0)[0]

    ones = np.append(ones, 99)
    ones = [x + 1 for x in ones]
    ones = np.insert(ones, 0, 0)
    img = np.zeros((im_size, im_size), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img[:, :, 0] = np.zeros([im_size, im_size]) + 255
    img[:, :, 1] = np.zeros([im_size, im_size]) + 255
    img[:, :, 2] = np.zeros([im_size, im_size]) + 255

    print(ones)
    for i in range(len(ones)-1):
        polys = [x[ones[i]: ones[i + 1], 0: 2].astype(np.int32)]

        cv2.polylines(img, polys, False, (0, 0, 0), lw)

    return img

