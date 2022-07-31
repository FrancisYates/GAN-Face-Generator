import math
import random

from PIL import Image, ImageFilter, ImageEnhance
from matplotlib import cm
import numpy as np

def img_brightness(img, amount):
    new_img = img.copy()
    enhancer = ImageEnhance.Brightness(new_img)
    new_img = enhancer.enhance(amount)
    return new_img

def img_salt_and_pepper(img, amount):
    new_img = np.copy(np.array(img, dtype=np.uint8))
    # add salt
    salt = np.ceil(amount * new_img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(salt)) for i in new_img.shape]
    new_img[coords] = 1

    # add pepper
    pepper = np.ceil(amount* new_img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(pepper)) for i in new_img.shape]
    new_img[coords] = 0

    #new_img = (new_img * 255).astype(np.uint8)
    #new_img = Image.fromarray(np.uint8(cm.gist_earth(new_img)*255))
    #new_img = Image.fromarray(np.uint8(new_img)).convert('RGB')
    return Image.fromarray(new_img)


def img_crop(img):
    width, height = img.size
    new_size = random.uniform(0.4, 0.9)
    new_width = math.ceil(width * new_size)
    new_height = math.ceil(height * new_size)

    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    right = left + new_width
    bot = top + new_height

    new_img = img.crop((left, top, right, bot))
    return new_img


def img_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def img_rotate(img, degrees):
    new_img = img.rotate(degrees)
    return new_img


def save_img(img, img_name, type, target_dir):

    new_name = img_name.replace('.JPG', '') + type + '.jpg'
    path = target_dir + '/' + new_name
    img.save(path)

    return new_name


def modify_imgs(source_dir, target_dir):
    img_paths = np.loadtxt(source_dir + 'labels.csv', delimiter=',', dtype=str, usecols=0)
    labels = np.loadtxt(source_dir + 'labels.csv', delimiter=',', usecols=[1,2,3])
    new_names = []
    new_labels = []

    #print(labels)
    count = 0
    for name in img_paths:
        path = target_dir + '/' + name
        img = Image.open(source_dir + name).convert('RGB')
        img.save(path)
        new_names.append(img_paths[count])
        vect = labels[count]
        new_labels.append(vect)

        """
        #Extends dataset by adding salt and pepper noise
        sp_amt = 0.02
        for i in range(0, 3, 1):
            new_img = img_salt_and_pepper(img, sp_amt*(i+1))

            name_ext = '_sp_' + str(100*sp_amt*(i+1))
            new_name = save_img(new_img, name, name_ext, target_dir)

            new_names.append(new_name)
            new_labels.append(labels[count])
        """

        for i in range(1, 2, 1):
        #Extends dataset by bluring image 
            new_img = img.filter(ImageFilter.GaussianBlur(i))

            name_ext = '_blur_' + str(i)
            new_name = save_img(new_img, name.replace(".jpg", ""), name_ext, target_dir)

            new_names.append(new_name)
            new_labels.append(labels[count])
            
        for i in range(1, 2, 1):
        #Extends dataset by increasing/decreasing image brightness
            brightness_change = 1 - 0.1 * i
            new_img = img_brightness(img, brightness_change)
            name_ext = '_brightness_' + str(brightness_change)
            new_name = save_img(new_img, name.replace(".jpg", ""), name_ext, target_dir)

            new_names.append(new_name)
            new_labels.append(labels[count])

            brightness_change = 1 + 0.1 * i
            new_img = img_brightness(img, brightness_change)
            name_ext = '_brightness_' + str(brightness_change)
            new_name = save_img(new_img, name.replace(".jpg", ""), name_ext, target_dir)

            new_names.append(new_name)
            new_labels.append(labels[count])

        """
        #Extends dataset by random cropping of image
        for i in range(0, 3, 1):
            new_img = img_crop(img)

            name_ext = '_crop_' + str(i + 1)
            new_name = save_img(new_img, name, name_ext, target_dir)

            new_names.append(new_name)
            new_labels.append(labels[count])
        """
        """
        #Extends dataset by flipping image
        new_img = img_flip(img)

        name_ext = '_flip'
        new_name = save_img(new_img, name, name_ext, target_dir)

        new_names.append(new_name)
        new_labels.append(labels[count])
        """

        """
        #Extends dataset by rotating image
        rotate_amt = 1
        for i in range(0, 2, 1):
            new_img = img_rotate(img, rotate_amt*(i+1))

            name_ext = '_rotateCC_' + str(rotate_amt*(i+1))
            new_name = save_img(new_img, name.replace(".jpg", ""), name_ext, target_dir)

            new_names.append(new_name)
            new_labels.append(labels[count])

            new_img = img_rotate(img, 360-(rotate_amt*(i+1)))

            name_ext = '_rotateC_' + str(rotate_amt*(i+1))
            new_name = save_img(new_img, name.replace(".jpg", ""), name_ext, target_dir)

            new_names.append(new_name)
            new_labels.append(labels[count])
        """
        count += 1

    new_data = []
    for i, img_name in enumerate(new_names):
        name = img_name
        x = new_labels[i][0]
        y = new_labels[i][1]
        z = new_labels[i][2]
        row = [name, x, y, z]
        new_data.append(row)

    np.savetxt(target_dir + '/labels.csv', new_data, fmt='%s', delimiter=',')


def main():
    source_dir = "C:/Users/Jane/Documents/Projects/DGAN/data/train/"
    target_dir = "C:/Users/Jane/Documents/Projects/DGAN/data/train_ex"
    modify_imgs(source_dir, target_dir)
    source_dir = "C:/Users/Jane/Documents/Projects/DGAN/data/validate/"
    target_dir = "C:/Users/Jane/Documents/Projects/DGAN/data/validate_ex"
    modify_imgs(source_dir, target_dir)


if __name__ == '__main__':
    main()
