import cv2
import pywt
import Transformations as tr
import Halftone as hf
import pathlib
from skimage.transform import downscale_local_mean



def main():

	img = cv2.imread('Lenna.jpg', 3)
	img = tr.RGB2YCBCR(img)
	Y, Cr, Cb = cv2.split(img)
	print(Y.shape)

	pathlib.Path('./FirstAttempt').mkdir(parents=True, exist_ok=True)
	cv2.imwrite("./FirstAttempt/OldY.jpg", Y)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	(Sl, (Sh, Sv, Sd)) = pywt.dwt2(Y, 'haar')


	ReducedCb = downscale_local_mean(Cb, (2, 2))
	ReducedCr = downscale_local_mean(Cr, (2, 2))

	# Sh = ReducedCb
	# Sv = ReducedCr

	NewY = pywt.idwt2((Sl, (ReducedCb, ReducedCr, Sd)), 'haar')

	cv2.imwrite("./FirstAttempt/NewYFirstTry.jpg", NewY)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	h = hf.Halftone('./FirstAttempt/NewYFirstTry.jpg')
	h.make(angles=[0, 15, 30, 45], antialias=True, percentage=10, sample=1, scale=2, style='grayscale')


if __name__ == '__main__':
	main()
