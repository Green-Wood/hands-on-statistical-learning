from PIL import Image
import numpy as np
from sklearn.mixture import GaussianMixture

im = Image.open('step3.png')

img = np.array(im)
shape = img.shape
img_reshape = img.reshape((-1, 3))

gmm = GaussianMixture(3)
pred = gmm.fit_predict(img_reshape)

img_reshape[pred == 0, :] = [255, 255, 0]
img_reshape[pred == 1, :] = [0, 0, 255]
img_reshape[pred == 2, :] = [0, 255, 0]

img_reshape = img_reshape.reshape((shape[0], shape[1], 3))

im = Image.fromarray(img_reshape.astype('uint8'))

im.save('result.png')

