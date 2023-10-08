from color.binary import *
from color.rgb import *
# from color.rgb_tuberlin import *

data = np.load('./xxx/x.npy', allow_pickle=True, encoding='latin1')
print(data.shape)
data = to_rgb_image(data)
cv2.imshow('img', data)
cv2.waitKey(0)

