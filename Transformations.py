import numpy as np


# Transform an RGB image to YCbCr image
def RGB2YCBCR(img):
	imageY = np.zeros((img.shape[0], img.shape[1]), dtype=np.int16)
	imageCb = np.zeros((img.shape[0], img.shape[1]), dtype=np.int16)
	imageCr = np.zeros((img.shape[0], img.shape[1]), dtype=np.int16)
	imageY = np.int16(((0.299 * img[:, :, 2]) + (0.587 * img[:, :, 1]) + (0.114 * img[:, :, 0])))
	imageCb = np.int16((0.564 * img[:, :, 0]) - (0.564 * imageY[:, :]) + 128)
	imageCr = np.int16((0.713 * img[:, :, 2]) - (0.713 * imageY[:, :]) + 128)
	img[:, :, 0] = imageY
	img[:, :, 1] = imageCr
	img[:, :, 2] = imageCb
	return img
