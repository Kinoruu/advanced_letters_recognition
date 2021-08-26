import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import os
import letters_import
import operator



Alphabets = []
Alphabets.append(letters_import.Arial_Letters)
Alphabets.append(letters_import.Times_NR_Letters)
Alphabets.append(letters_import.Courier_N_Letters)

alphabets = []
alphabets.append("Arial_Letters")
alphabets.append("Times_NR_Letters")
alphabets.append("Courier_N_Letters")

names = []

Alphabets_pad = []
Arial_Letters = []
Alphabets_pad.append(Arial_Letters)
Times_NR_Letters = []
Alphabets_pad.append(Times_NR_Letters)
Courier_N_Letters = []
Alphabets_pad.append(Courier_N_Letters)

'''
name0 = "_pad"
for alphabet in alphabets:
    name = alphabet
    name2 = name + name0
    names.append(name2)
    exec("%s = %d" % (name2, []))

names = []
'''

for alphabet in Alphabets_pad:
    for alphabet2 in Alphabets:
        temp = []
        for letter in alphabet2:
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
                background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                alphabet.append(background)
                #temp.append(background)
                #Alphabets_pad.append(exec("%s = %d" % (name2, temp)))


            resize_with_pad(letter_not_array, w + 40, h + 40)

gray = cv2.imread(filename = 'letters_cn.png', flags = cv2.IMREAD_GRAYSCALE)
ret, binary = cv2.threshold(gray, 100, 255,                          cv2.THRESH_OTSU)
inverted_binary = ~binary
contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

k = 0
found_letter = []
iter1 = 0
for c in contours:
    iter1 = iter1 + 1
    iter2 = 0
    x, y, w, h = cv2.boundingRect(c)
    found_letter.append(c)
    for d in contours:
        iter2 = iter2 + 1
        if(iter1 >= iter2):
            pass
        else:
            x2, y2, w2, h2 = cv2.boundingRect(d)
            if ((x >= x2) and (y >= y2) and ((x + w) <= (x2 + w2)) and ((y + h) <= (y2 + h2))):
                found_letter.pop()

try:
    os.mkdir('found')
except FileExistsError:
    pass
try:
    os.mkdir('found_pad')
except FileExistsError:
    pass
try:
    os.mkdir('kaze')
except FileExistsError:
    pass
try:
    os.mkdir('brisk')
except FileExistsError:
    pass
for c in found_letter:
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
        Distances_kaze = []
        Distances_brisk = []
        for iterator in range(26):
            print(chr(iterator+97))
            for alphabet in Alphabets_pad:
                i = cv2.imread('found_pad/found_letter_pad_' + str(k) + '.png')
                j = alphabet[iterator]
                # KAZE
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
                            cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + '.jpg', output3)
                            cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + ' i.jpg', i)
                            cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + ' j.jpg', j)
                    Distances_kaze.append(max.distance)
                    #print(max.distance)

                # BRISK
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
                            cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + '.jpg', output3)
                            cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + ' i.jpg', i)
                            cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + ' j.jpg', j)
                    Distances_brisk.append(max.distance)

            Distances_kaze.sort()
            Distances_brisk.sort()
            print(Distances_kaze[0])
            print(Distances_brisk[0])
            z = z + 1
        return background.convert('RGB')

    resize_with_pad(im, w + 40, h + 40)
