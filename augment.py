import os
from scipy import misc
from PIL import Image

def swap(a,b):
	return b,a

def flip(img):
#	print img
	for i in range(len(img[0])):
		for j in range(len(img[0])/2):
			img[i,j],img[i][len(img[0])-j-1]=swap(img[i][j],img[i][len(img[0])-j-1])
#	print img
	return img

def transpose(img):
#	print img
	for i in range(len(img[0])-1):
		for j in range(len(img[0])-i-1):
			img[i+j+1,i], img[i][i+j+1]=swap(img[i+j+1][i],img[i][i+j+1])
#	print img
	return img

path='crops'
image= misc.imread(os.path.join(path,'b_01.bmp'), flatten= 0)
image_=image
## flatten=0 if image is required as it is 
## flatten=1 to flatten the color layers into a single gray-scale layer
image1=flip(image)
im = Image.fromarray(image)
im.save("crops/b_01_1.bmp")

image2=transpose(image_)
im = Image.fromarray(image)
im.save("crops/b_01_2.bmp")

image3=transpose(image1)
im = Image.fromarray(image)
im.save("crops/b_01_3.bmp")

image4=flip(image3)
im = Image.fromarray(image)
im.save("crops/b_01_4.bmp")

image5=transpose(image4)
im = Image.fromarray(image)
im.save("crops/b_01_5.bmp")

image6=flip(image5)
im = Image.fromarray(image)
im.save("crops/b_01_6.bmp")

image7=transpose(image6)
im = Image.fromarray(image)
im.save("crops/b_01_7.bmp")

#im = Image.fromarray(imglist[0])		
#im.save("crops/why.bmp")
#for i in range(7):
#	im = Image.fromarray(imageArr[i])
#	im.save("crops/b_01_%s.bmp" % i)
