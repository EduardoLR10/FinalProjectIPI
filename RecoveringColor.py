import cv2
import pywt
import Transformations as tr
import halftone as hf
from skimage.transform import rescale, downscale_local_mean


def main():
	img = cv2.imread('Lenna.jpg', 3)
	print(img.shape)
	img = tr.RGB2YCBCR(img)
	Y, Cr, Cb = cv2.split(img)

	# Color Embedding

	(Sl, (Sh1, Sv1, Sd1), (Sh2, Sv2, Sd2)) = pywt.wavedec2(Y, 'db1', level=2)

	ReducedCb = downscale_local_mean(Cb, (2, 2))
	ReducedCr = downscale_local_mean(Cr, (2, 2))

	CbPlus = ReducedCb
	CbMinus = ReducedCb

	for i in range(0, ReducedCb.shape[0]):
		for j in range(0, ReducedCb.shape[1]):
			if ReducedCb[i, j] < 0:
				CbPlus[i, j] = 0
			elif ReducedCb[i, j] > 0:
				CbMinus[i, j] = 0

	CrPlus = ReducedCr
	CrMinus = ReducedCr

	for i in range(0, ReducedCr.shape[0]):
		for j in range(0, ReducedCr.shape[1]):
			if ReducedCr[i, j] < 0:
				CrPlus[i, j] = 0
			elif ReducedCb[i, j] > 0:
				CrMinus[i, j] = 0

	ReducedCbMinus = downscale_local_mean(CbMinus, (2, 2))

	# Sd1 = ReducedCbMinus
	# Sh2 = CrPlus
	# Sv2 = CbPlus
	# Sd2 = CrMinus

	NewY = pywt.waverec2((Sl, (Sh1, Sv1, ReducedCbMinus), (CrPlus, CbPlus, CrMinus)), 'db1')
	h = hf.Halftone('NewY.jpg')
	h.make(angles=[0, 15, 30, 45], antialias=True, percentage=10, sample=1, scale=2, style='grayscale')
	cv2.imwrite("NewY2.jpg", NewY)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Color Recovery

	img = cv2.imread('NewY2.jpg')
	print(img.shape)


if __name__ == '__main__':
	main()
