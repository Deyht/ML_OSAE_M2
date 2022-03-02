
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageOps
from tqdm import tqdm


def make_square(im, min_size=128, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom*2
    img = img.crop((x - w / zoom2, y - h / zoom2, 
                    x + w / zoom2, y + h / zoom2))
    return img.resize((w, h), Image.LANCZOS)

def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def im_transform(im):
	im2 = im.copy()
	if(np.random.randint(0,2)):
		im2 = ImageOps.mirror(im2)
	im2 = im2.transform(im2.size, Image.AFFINE, (1.0-0.2*np.random.rand(),0.0, (np.random.rand()*2.0-1.0)*(im2.size[0]*0.1),
		0.0,1.0-0.2*np.random.rand(), (np.random.rand()*2.0-1.0)*(im2.size[1]*0.1)))
	#im2 = im2.rotate(np.random.rand()*40.0-20.0)
	im2 = im2.resize((128,128))
	
	return im2


orig_nb_images = 11500
test_size = 1000
image_size = 128
augm_fact = 6

train_im = np.zeros((orig_nb_images*augm_fact*2, 3, image_size, image_size), dtype="uint8")

for i in tqdm(range(0, orig_nb_images)):
	im = Image.open("/obs/dcornu/CIANNA/data_asirra/PetImages/Cat/"+str(i)+".jpg")
	width, height = im.size

	im = make_square(im)
	width2, height2 = im.size

	x_offset = int((width2 - width)*0.5)
	y_offset = int((height2 - height)*0.5)

	im = im.resize((image_size,image_size))
	
	for k in range(0,augm_fact):
		im_array = np.asarray(im_transform(im))
		#plt.imshow(im_array)
		#plt.show()
		for depth in range(0,3):
			train_im[i*augm_fact+k,depth,:,:] = im_array[:,:,depth]
	#if(i == 1):
	#	exit() 

for i in tqdm(range(0,orig_nb_images)):
	im = Image.open("/obs/dcornu/CIANNA/data_asirra/PetImages/Dog/"+str(i)+".jpg")
	width, height = im.size

	im = make_square(im)
	width2, height2 = im.size

	x_offset = int((width2 - width)*0.5)
	y_offset = int((height2 - height)*0.5)

	im = im.resize((image_size,image_size))
	
	for k in range(0,augm_fact):
		im_array = np.asarray(im_transform(im))
		for depth in range(0,3):
			train_im[orig_nb_images*augm_fact+i*augm_fact+k,depth,:,:] = im_array[:,:,depth]
		


train_im.tofile("train_im.dat")



test_im = np.zeros((test_size*2, 3, image_size, image_size), dtype="uint8")

for i in tqdm(range(0, test_size)):
	im = Image.open("/obs/dcornu/CIANNA/data_asirra/PetImages/Cat/"+str(orig_nb_images+i)+".jpg")
	width, height = im.size

	im = make_square(im)
	width2, height2 = im.size

	x_offset = int((width2 - width)*0.5)
	y_offset = int((height2 - height)*0.5)

	im = im.resize((image_size,image_size))
	
	im_array = np.asarray((im))
	for depth in range(0,3):
		test_im[i,depth,:,:] = im_array[:,:,depth]

for i in tqdm(range(0,test_size)):
	im = Image.open("/obs/dcornu/CIANNA/data_asirra/PetImages/Dog/"+str(orig_nb_images+i)+".jpg")
	width, height = im.size

	im = make_square(im)
	width2, height2 = im.size

	x_offset = int((width2 - width)*0.5)
	y_offset = int((height2 - height)*0.5)

	im = im.resize((image_size,image_size))
	
	im_array = np.asarray((im))
	for depth in range(0,3):
		test_im[test_size+i,depth, :,:] = im_array[:,:,depth]


test_im.tofile("test_im.dat")

