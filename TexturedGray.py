import cv2
import pywt
from PIL import Image
import Transformations as tr
import halftone as hf
from skimage.transform import downscale_local_mean



def main():

	img = cv2.imread('Lenna.jpg', 3)
	img = tr.RGB2YCBCR(img)
	Y, Cr, Cb = cv2.split(img)
	print(Y.shape)

	cv2.imwrite("OldY.jpg", Y)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	(Sl, (Sh, Sv, Sd)) = pywt.dwt2(Y, 'haar')


	ReducedCb = downscale_local_mean(Cb, (2, 2))
	ReducedCr = downscale_local_mean(Cr, (2, 2))

	# Sh = ReducedCb
	# Sv = ReducedCr

	NewY = pywt.idwt2((Sl, (ReducedCb, ReducedCr, Sd)), 'haar')

	# scale_percent = 200  # percent of original size
	# width = int(NewY.shape[1] * scale_percent / 100)
	# height = int(NewY.shape[0] * scale_percent / 100)
	# dim = (width, height)

	cv2.imwrite("NewYFirstTry.jpg", NewY)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Resize image
	# resized = cv2.resize(NewY, dim, interpolation=cv2.INTER_AREA)
	# print(resized.shape)

	h = hf.Halftone('NewYFirstTry.jpg')
	h.make(angles=[0, 15, 30, 45], antialias=True, percentage=10, sample=1, scale=2, style='grayscale')


if __name__ == '__main__':
	main()
