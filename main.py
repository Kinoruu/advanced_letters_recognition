import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import os

Letters = []
letters = []

l_a = cv2.imread(filename = 'alphabet_arial/a.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_a)
letters.append ("a")
l_b = cv2.imread(filename = 'alphabet_arial/b.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_b)
letters.append ("b")
l_c = cv2.imread(filename = 'alphabet_arial/c.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_c)
letters.append ("c")
l_d = cv2.imread(filename = 'alphabet_arial/d.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_d)
letters.append ("d")
l_e = cv2.imread(filename = 'alphabet_arial/e.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_e)
letters.append ("e")
l_f = cv2.imread(filename = 'alphabet_arial/f.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_f)
letters.append ("f")
l_g = cv2.imread(filename = 'alphabet_arial/g.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_g)
letters.append ("g")
l_h = cv2.imread(filename = 'alphabet_arial/h.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_h)
letters.append ("h")
l_i = cv2.imread(filename = 'alphabet_arial/i.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_i)
letters.append ("i")
l_j = cv2.imread(filename = 'alphabet_arial/j.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_j)
letters.append ("j")
l_k = cv2.imread(filename = 'alphabet_arial/k.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_k)
letters.append ("k")
l_l = cv2.imread(filename = 'alphabet_arial/l.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_l)
letters.append ("l")
l_m = cv2.imread(filename = 'alphabet_arial/m.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_m)
letters.append ("m")
l_n = cv2.imread(filename = 'alphabet_arial/n.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_n)
letters.append ("n")
l_o = cv2.imread(filename = 'alphabet_arial/o.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_o)
letters.append ("o")
l_p = cv2.imread(filename = 'alphabet_arial/p.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_p)
letters.append ("p")
l_q = cv2.imread(filename = 'alphabet_arial/q.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_q)
letters.append ("q")
l_r = cv2.imread(filename = 'alphabet_arial/r.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_r)
letters.append ("r")
l_s = cv2.imread(filename = 'alphabet_arial/s.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_s)
letters.append ("s")
l_t = cv2.imread(filename = 'alphabet_arial/t.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_t)
letters.append ("t")
l_u = cv2.imread(filename = 'alphabet_arial/u.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_u)
letters.append ("u")
l_v = cv2.imread(filename = 'alphabet_arial/v.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_v)
letters.append ("v")
l_w = cv2.imread(filename = 'alphabet_arial/w.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_w)
letters.append ("w")
l_x = cv2.imread(filename = 'alphabet_arial/x.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_x)
letters.append ("x")
l_y = cv2.imread(filename = 'alphabet_arial/y.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_y)
letters.append ("y")
l_z = cv2.imread(filename = 'alphabet_arial/z.png', flags = cv2.IMREAD_GRAYSCALE)
Letters.append (l_z)
letters.append ("z")

Letters_pad = []
for letter in Letters:
    h, w = letter.shape
    letter_not_array = Image.fromarray(letter)
    def resize_with_pad(im, target_width, target_height):
        target_ratio = target_height / target_width
        im_ratio = im.height / im.width
        if target_ratio > im_ratio:
            resize_width = target_width - im.width
            resize_height = target_height - im.height
        else:
            resize_height = target_height
            resize_width = round(resize_height / im_ratio)
        background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
        offset = (round((target_width - im.width) / 2), round((target_height - im.height) / 2))
        background.paste(im, offset)
        background = np.array(background)
        background=cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        Letters_pad.append(background)


    resize_with_pad(letter_not_array, w + 40, h + 40)

gray = cv2.imread(filename = 'letters_cn.png', flags = cv2.IMREAD_GRAYSCALE)
ret, binary = cv2.threshold(gray, 100, 255,                          cv2.THRESH_OTSU)
inverted_binary = ~binary
contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

k = 0
g = []
iter1 = 0
for c in contours:
    iter1 = iter1 + 1
    iter2 = 0
    x, y, w, h = cv2.boundingRect(c)
    g.append(c)
    for d in contours:
        iter2 = iter2 + 1
        if(iter1 >= iter2):
            pass
        else:
            x2, y2, w2, h2 = cv2.boundingRect(d)
            if ((x >= x2) and (y >= y2) and ((x + w) <= (x2 + w2)) and ((y + h) <= (y2 + h2))):
                g.pop()

try:
    os.mkdir('found')
except FileExistsError:
    pass
try:
    os.mkdir('found_pad')
except FileExistsError:
    pass
for c in g:
    x, y, w, h = cv2.boundingRect(c)
    k=k+1
    im = gray[y-5:y + h+5, x-5:x + w+5]
    cv2.imwrite('found/found_letter_' + str(k) + '.png', im)
    im2 = cv2.imread('found/found_letter_' + str(k) + '.png')
    im = Image.fromarray(im)

    def resize_with_pad(im, target_width, target_height):

        target_ratio = target_height / target_width
        im_ratio = im.height / im.width
        if target_ratio > im_ratio:
            resize_width = target_width - im.width
            resize_height = target_height - im.height
        else:
            resize_height = target_height
            resize_width = round(resize_height / im_ratio)

        background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
        offset = (round((target_width - im.width) / 2), round((target_height - im.height) / 2))
        background.paste(im, offset)
        background.save('found_pad/found_letter_pad_' + str(k) + '.png')

        z = 0
        for j in Letters_pad:
            i = cv2.imread('found_pad/found_letter_pad_' + str(k) + '.png')

            #KAZE
            kaze = cv2.KAZE_create()
            keypoints1, descriptors1 = kaze.detectAndCompute(i, None)
            keypoints2, descriptors2 = kaze.detectAndCompute(j, None)
            if descriptors1 is None or descriptors2 is None:
                pass
            else:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=100)
                descriptors1 = np.float32(descriptors1)
                descriptors2 = np.float32(descriptors2)
                FLANN = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
                matches = FLANN.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
                ratio_thresh = 0.7
                good_matches = []
                matches2 = []
                for m, n in matches:
                    matches2.append(m)
                    if m.distance <= ratio_thresh * n.distance:
                        good_matches.append(m)
                if (len(matches2) > 0):
                    max = matches2[0]
                    for m in matches2:
                        now = m
                        if (max.distance > now.distance):
                            max = now
                    if ((max.distance <= 0.091)):
                        output3 = cv2.drawMatches(img1=i, keypoints1=keypoints1, img2=j,
                                                  keypoints2=keypoints2, matches1to2=matches2, outImg=None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        #cv2.imwrite('image_kaze_best_' + str(z) + " " + str(k) + '.jpg', output3)
                        #cv2.imwrite('image_kaze_best_' + str(z) + " " + str(k) + ' i.jpg', i)
                        #cv2.imwrite('image_kaze_best_' + str(z) + " " + str(k) + ' j.jpg', j)


            #BRISK
            BRISK = cv2.BRISK_create()
            keypoints1, descriptors1 = BRISK.detectAndCompute(i, None)
            keypoints2, descriptors2 = BRISK.detectAndCompute(j, None)
            if descriptors1 is None or descriptors2 is None:
                pass
            else:
                BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
                matches = BFMatcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
                matches = sorted(matches, key=lambda x: x.distance)
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=100)
                descriptors1 = np.float32(descriptors1)
                descriptors2 = np.float32(descriptors2)
                FLANN = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
                matches = FLANN.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
                ratio_thresh = 0.7
                good_matches = []
                matches2 = []
                for m, n in matches:
                    matches2.append(m)
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                if (len(matches2) > 0):
                    max = matches2[0]
                    for m in matches2:
                        now = m
                        if (max.distance > now.distance):
                            max = now
                    # print(max.distance)
                    if ((max.distance <= 490)):
                        output3 = cv2.drawMatches(img1=i, keypoints1=keypoints1, img2=j, keypoints2=keypoints2,
                                                  matches1to2=matches2, outImg=None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        #cv2.imwrite('image_brisk_best_' + str(z) + " " + str(k) + '.jpg', output3)
                        #cv2.imwrite('image_brisk_best_' + str(z) + " " + str(k) + ' i.jpg', i)
                        #cv2.imwrite('image_brisk_best_' + str(z) + " " + str(k) + ' j.jpg', j)

            z = z + 1
        return background.convert('RGB')

    resize_with_pad(im, w + 40, h + 40)
