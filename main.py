import numpy as np
import cv2
from PIL import Image
import os
# import glob
import letters_import
# import operator
# import time


# creating or cleaning directories
try:
    os.mkdir('found')
except FileExistsError:
    for root, dirs, files in os.walk('found'):
        for f in files:
            os.unlink(os.path.join(root, f))
try:
    os.mkdir('found_v2')
except FileExistsError:
    for root, dirs, files in os.walk('found_v2'):
        for f in files:
            os.unlink(os.path.join(root, f))
try:
    os.mkdir('found_pad')
except FileExistsError:
    for root, dirs, files in os.walk('found_pad'):
        for f in files:
            os.unlink(os.path.join(root, f))
try:
    os.mkdir('found_pad_v2')
except FileExistsError:
    for root, dirs, files in os.walk('found_pad_v2'):
        for f in files:
            os.unlink(os.path.join(root, f))
try:
    os.mkdir('kaze')
except FileExistsError:
    for root, dirs, files in os.walk('kaze'):
        for f in files:
            os.unlink(os.path.join(root, f))
try:
    os.mkdir('brisk')
except FileExistsError:
    for root, dirs, files in os.walk('brisk'):
        for f in files:
            os.unlink(os.path.join(root, f))
try:
    os.mkdir('segmentation')
except FileExistsError:
    for root, dirs, files in os.walk('segmentation'):
        for f in files:
            os.unlink(os.path.join(root, f))
try:
    os.mkdir('croped')
except FileExistsError:
    for root, dirs, files in os.walk('croped'):
        for f in files:
            os.unlink(os.path.join(root, f))
# flags and changeable elements
flag_i = 0  # italic flag
flag_b = 0  # bold flag
mean_range = 4  # best possible choosed
flag_geometric_mean = 1
flag_mean = 0
flag_minimum = 0
height_pad = 5  # best possible choosed
width_pad = 2  # best possible choosed
threshold = 245  # best possible choosed
# creating vectors for alphabets and importing from "import_letters.py"
Alphabets = [letters_import.Arial_Letters, letters_import.Times_NR_Letters, letters_import.Courier_N_Letters,
             letters_import.Calibri_Letters, letters_import.Comic_S_Letters, letters_import.Alef_Letters]
if flag_i == 1:
    Alphabets.append(letters_import.Arial_Letters_italic)
    Alphabets.append(letters_import.Times_NR_Letters_italic)
    Alphabets.append(letters_import.Courier_N_Letters_italic)
    Alphabets.append(letters_import.Calibri_Letters_italic)
    Alphabets.append(letters_import.Comic_S_Letters_italic)
    Alphabets.append(letters_import.Alef_Letters_italic)
if flag_b == 1:
    Alphabets.append(letters_import.Arial_Letters_bold)
    Alphabets.append(letters_import.Times_NR_Letters_bold)
    Alphabets.append(letters_import.Courier_N_Letters_bold)
    Alphabets.append(letters_import.Calibri_Letters_bold)
    Alphabets.append(letters_import.Comic_S_Letters_bold)
    Alphabets.append(letters_import.Alef_Letters_bold)
Alphabet_Letters = letters_import.letters
alphabets_names = ["Arial_Letters", "Times_NR_Letters", "Courier_N_Letters", "Calibri_Letters", "Comic_S_Letters",
                   "Alef_Letters"]
if flag_i == 1:
    alphabets_names.append("Arial_Letters_italic")
    alphabets_names.append("Times_NR_Letters_italic")
    alphabets_names.append("Courier_N_Letters_italic")
    alphabets_names.append("Calibri_Letters_italic")
    alphabets_names.append("Comic_S_Letters_italic")
    alphabets_names.append("Alef_Letters_italic")
if flag_b == 1:
    alphabets_names.append("Arial_Letters_bold")
    alphabets_names.append("Times_NR_Letters_bold")
    alphabets_names.append("Courier_N_Letters_bold")
    alphabets_names.append("Calibri_Letters_bold")
    alphabets_names.append("Comic_S_Letters_bold")
    alphabets_names.append("Alef_Letters_bold")
Alphabets_pad = []
Arial_Letters_no_pad = []
Times_NR_Letters_no_pad = []
Courier_N_Letters_no_pad = []
Comic_S_Letters_no_pad = []
Calibri_Letters_no_pad = []
Alef_Letters_no_pad = []
Alphabets_pad = [Arial_Letters_no_pad, Times_NR_Letters_no_pad, Courier_N_Letters_no_pad, Comic_S_Letters_no_pad,
                 Calibri_Letters_no_pad, Alef_Letters_no_pad]
if flag_i == 1:
    Arial_Letters_italic_no_pad = []
    Alphabets_pad.append(Arial_Letters_italic_no_pad)
    Times_NR_Letters_italic_no_pad = []
    Alphabets_pad.append(Times_NR_Letters_italic_no_pad)
    Courier_N_Letters_italic_no_pad = []
    Alphabets_pad.append(Courier_N_Letters_italic_no_pad)
    Comic_S_Letters_italic_no_pad = []
    Alphabets_pad.append(Comic_S_Letters_italic_no_pad)
    Calibri_Letters_italic_no_pad = []
    Alphabets_pad.append(Calibri_Letters_italic_no_pad)
    Alef_Letters_italic_no_pad = []
    Alphabets_pad.append(Alef_Letters_italic_no_pad)
if flag_b == 1:
    Arial_Letters_bold_no_pad = []
    Alphabets_pad.append(Arial_Letters_bold_no_pad)
    Times_NR_Letters_bold_no_pad = []
    Alphabets_pad.append(Times_NR_Letters_bold_no_pad)
    Courier_N_Letters_bold_no_pad = []
    Alphabets_pad.append(Courier_N_Letters_bold_no_pad)
    Comic_S_Letters_bold_no_pad = []
    Alphabets_pad.append(Comic_S_Letters_bold_no_pad)
    Calibri_Letters_bold_no_pad = []
    Alphabets_pad.append(Calibri_Letters_bold_no_pad)
    Alef_Letters_bold_no_pad = []
    Alphabets_pad.append(Alef_Letters_bold_no_pad)
# adding pads to libraries alphabets letters
iterator_empty = 0
for alphabet in Alphabets_pad:
    iterator_full = 0
    for alphabet2 in Alphabets:
        temp = []
        if iterator_empty == iterator_full:
            for letter in alphabet2:
                h, w = letter.shape
                letter_not_array = Image.fromarray(letter)

                def resize_with_pad(image, target_width, target_height):
                    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
                    offset = (round((target_width - image.width) / 2), round((target_height - image.height) / 2))
                    background.paste(image, offset)
                    background = np.array(background)
                    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                    alphabet.append(background)
                resize_with_pad(letter_not_array, w + 40, h + 40)
        iterator_full += 1
    iterator_empty += 1
# searching all shapes in the input image
no_gray = cv2.imread(filename='test3.png')
pic2 = cv2.imread(filename='test3.png')
gray = cv2.imread(filename='test3.png', flags=cv2.IMREAD_GRAYSCALE)
gray = np.array(gray)
height, width = gray.shape
white = 0
for h in range(height):
    for w in range(width):
        if gray[h, w] == [255]:
            white = white + 1
if white < (height * width) / 2:
    gray = cv2.cv2.bitwise_not(gray)
ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_OTSU)
inverted_binary = ~binary
contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# deleting shapes included in bigger ones
k = 0
found_letter = []
found_best = []
found_best_r = []
for c in contours:
    found_letter.append(c)


def selection(contours_sel):
    iter1 = 0
    for cs in contours_sel:
        iter1 = iter1 + 1
        iter2 = 0
        xs, ys, ws, hs = cv2.boundingRect(cs)
        found_best.append(cs)
        # if w < 10 or h < 5:
            # found_letter[iter2]
        for ds in contours_sel:
            iter2 = iter2 + 1
            if iter1 == iter2:
                pass
            else:
                xs2, ys2, ws2, hs2 = cv2.boundingRect(ds)
                if (xs >= xs2) and ((xs + ws) <= (xs2 + ws2)) and (ys >= ys2) and ((ys + hs) <= (ys2 + hs2)):
                    found_best.pop()
                    pass


def rev_selection(r_contours_sel):
    iter1 = 0
    for cs in reversed(r_contours_sel):
        iter1 = iter1 + 1
        iter2 = 0
        xs, ys, ws, hs = cv2.boundingRect(cs)
        found_best_r.append(cs)
        # if w < 10 or h < 5:
            # found_letter.pop()
        for ds in r_contours_sel:
            iter2 = iter2 + 1
            if iter1 == iter2:
                pass
            else:
                xs2, ys2, ws2, hs2 = cv2.boundingRect(ds)
                if (xs >= xs2) and ((xs + ws) <= (xs2 + ws2)) and (ys >= ys2) and ((ys + hs) <= (ys2 + hs2)):
                    found_best_r.pop()
                    pass


pic1 = no_gray
selection(found_letter)
number_of_found_letters = len(found_best)  # number of found shapes
print("OCR 1 have found ", number_of_found_letters, " letters")
found_better = []

for c in found_best:
    x, y, w, h = cv2.boundingRect(c)
    found_better.append(c)
    if (cv2.contourArea(c)) > 100:
        cv2.rectangle(pic1, (x, y), (x + w, y + h), (77, 22, 174), 2)
    else:
        found_better.pop()
cv2.imwrite('All contours with bounding box.png', pic1)
number_of_found_letters = len(found_better)  # number of found shapes
print("OCR 2 have found ", number_of_found_letters, " letters")

image_height, image_width = gray.shape

x_min = 0
x_max = image_width
y_min = 0
y_max = image_height
info = [[], [], []]
counter_of_contours_in_line = 0
counter_of_lines = 0


for c in found_better:
    (x, y, w, h) = cv2.boundingRect(c)
    print('x:' + str(x) + ' y:' + str(y) + ' w:' + str(w) + ' h:' + str(h))
    x_l = x
    x_r = x + w
    y_d = y
    y_u = y + h
    if ((y_min >= y_d and y_max <= y_u) or (y_min <= y_d and y_max >= y_u) or
            (y_min >= y_d and (y_max >= y_u >= y_min)) or (y_min <= y_d and (y_u >= y_max >= y_d)) or
            (y_max <= y_u and (y_max >= y_d >= y_min)) or (y_max >= y_u and (y_u >= y_min >= y_d))):
        if y_min > y_d:
            y_min = y_d
        if y_max < y_u:
            y_max = y_u
        counter_of_contours_in_line += 1
    else:
        # print(str(y_min) + ' ' + str(y_max) + ' ' + str(counter_of_contours_in_line))
        column = [[y_min], [y_max], [counter_of_contours_in_line + 1]]
        info = np.append(info, column, axis=1)
        y_min = 0
        y_max = image_height
        counter_of_contours_in_line = 0
    print(str(y_min) + ' ' + str(y_max) + ' ' + str(counter_of_contours_in_line))
column = [[y_min],[y_max],[counter_of_contours_in_line]]
info = np.append(info, column, axis=1)
print(info)

pic3 = no_gray
for c in found_better:
    iteratorx = int(info[2][counter_of_lines])
    y_min_ = info[0][counter_of_lines]
    y_max_ = info[1][counter_of_lines]
    print(iteratorx)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(pic3, (x, int(y_min_)), (x + w, int(y_max_)), (77, 22, 174), 2)
    info[2][counter_of_lines] -= 1
    if iteratorx == 0:
        counter_of_lines += 1
cv2.imwrite('All contours with bounding box after normalization.png', pic1)

mser = cv2.MSER_create()
gray_pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
vis = pic2.copy()
regions, _ = mser.detectRegions(gray_pic2)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, (0, 255, 0))

mask = np.zeros((pic2.shape[0], pic2.shape[1], 1), dtype=np.uint8)
mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))
for contour in hulls:
    cv2.drawContours(mask, [contour], -1, (25, 25, 55), -1)
    x, y, w, h = cv2.boundingRect(contour)
    new_img = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    im = pic2[y - height_pad:y + h + height_pad, x - width_pad:x + w + width_pad]
    im = Image.fromarray(im)
    # im.paste(pic2, new_img)
    im.save('try2.png')
    im = cv2.imread('try2.png')
    # cv2.imshow('try2.png', im)
    text_only = cv2.bitwise_and(pic2, pic2, mask=mask)
cv2.imwrite('try1.png', text_only)
regions, _ = mser.detectRegions(gray_pic2)

bounding_boxes = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
# cv2.imwrite('try2.png', bounding_boxes)
# creating vectors for output
pre_output_000 = []    # brisk arial
pre_output_001 = []    # brisk times new roman
pre_output_002 = []    # brisk courier new
pre_output_003 = []    # brisk calibri
pre_output_004 = []    # brisk comic sans
pre_output_005 = []    # brisk alef

pre_output_100 = []    # kaze arial
pre_output_101 = []    # kaze times new roman
pre_output_102 = []    # kaze courier new
pre_output_103 = []    # kaze calibri
pre_output_104 = []    # kaze comic sans
pre_output_105 = []    # kaze alef

pre_output_200 = []   # brisk arial italic
pre_output_201 = []   # brisk times new roman italic
pre_output_202 = []   # brisk courier new italic
pre_output_203 = []   # brisk calibri italic
pre_output_204 = []   # brisk comic sans italic
pre_output_205 = []   # brisk alef italic

pre_output_300 = []   # kaze arial italic
pre_output_301 = []   # kaze times new roman italic
pre_output_302 = []   # kaze courier new italic
pre_output_303 = []   # kaze calibri italic
pre_output_304 = []   # kaze comic sans italic
pre_output_305 = []   # kaze alef italic

pre_output_400 = []   # brisk arial bold
pre_output_401 = []   # brisk times new roman bold
pre_output_402 = []   # brisk courier new bold
pre_output_403 = []   # brisk calibri bold
pre_output_404 = []   # brisk comic sans bold
pre_output_405 = []   # brisk alef bold

pre_output_500 = []   # kaze arial bold
pre_output_501 = []   # kaze times new roman bold
pre_output_502 = []   # kaze courier new bold
pre_output_503 = []   # kaze calibri bold
pre_output_504 = []   # kaze comic sans bold
pre_output_505 = []   # kaze alef bold


program_output = []

found_letter_pad = []
found_letter_v2 = []


def resize_with_pad(image, target_width, target_height, pr):  # function adding pads to found shapes
    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - image.width) / 2), round((target_height - image.height) / 2))
    background.paste(image, offset)
    background.save('found_pad/found_letter_pad_' + str(pr) + '.png')
    imgs = cv2.imread('found_pad/found_letter_pad_' + str(pr) + '.png')
    found_letter_pad.append(imgs)


p = 0
for c in found_better:
    x, y, w, h = cv2.boundingRect(c)
    p = p + 1
    im = gray[y - height_pad:y + h + height_pad, x - width_pad:x + w + width_pad]  # cutting shapes from input image
    cv2.imwrite('found/found_letter_' + str(p) + '.png', im)

    im = Image.fromarray(im)
    # result = Image.fromarray(result)
    im2 = cv2.imread('found/found_letter_' + str(p) + '.png')

    resize_with_pad(im, w + 40, h + 40, p)

def check_if(pre_output_list, method_list, alphabet_list):
    try:
        pre_output_list.append((Alphabet_Letters[method_list.index(min(alphabet_list))]))
    except:
        pass


number_of_found_letters_v2 = len(found_letter_v2)  # number of found shapes
print("OCR 3 have found ", number_of_found_letters_v2, " letters")
# main part of program
for fl in found_letter_pad:

    # x, y, w, h = cv2.boundingRect(fl)
    k = k + 1

    # def resize_with_pad_v2(image, target_width, target_height):  # function adding pads to found shapes

    z = 0  # iterator for naming images containing matches
    distances_kaze_a = []
    distances_kaze_tnr = []
    distances_kaze_cn = []
    distances_kaze_c = []
    distances_kaze_cs = []
    distances_kaze_al = []
    distances_kaze = [distances_kaze_a, distances_kaze_tnr, distances_kaze_cn, distances_kaze_c, distances_kaze_cs,
                      distances_kaze_al]
    distances_kaze_a_i = []
    distances_kaze_tnr_i = []
    distances_kaze_cn_i = []
    distances_kaze_c_i = []
    distances_kaze_cs_i = []
    distances_kaze_al_i = []
    distances_kaze_a_b = []
    distances_kaze_tnr_b = []
    distances_kaze_cn_b = []
    distances_kaze_c_b = []
    distances_kaze_cs_b = []
    distances_kaze_al_b = []
    if flag_i == 1:
        distances_kaze.append(distances_kaze_a_i)
        distances_kaze.append(distances_kaze_tnr_i)
        distances_kaze.append(distances_kaze_cn_i)
        distances_kaze.append(distances_kaze_c_i)
        distances_kaze.append(distances_kaze_cs_i)
        distances_kaze.append(distances_kaze_al_i)
    if flag_b == 1:
        distances_kaze.append(distances_kaze_a_b)
        distances_kaze.append(distances_kaze_tnr_b)
        distances_kaze.append(distances_kaze_cn_b)
        distances_kaze.append(distances_kaze_c_b)
        distances_kaze.append(distances_kaze_cs_b)
        distances_kaze.append(distances_kaze_al_b)
    distances_brisk_a = []
    distances_brisk_tnr = []
    distances_brisk_cn = []
    distances_brisk_c = []
    distances_brisk_cs = []
    distances_brisk_al = []
    distances_brisk = [distances_brisk_a, distances_brisk_tnr, distances_brisk_cn, distances_brisk_c, distances_brisk_cs
                       , distances_brisk_al]
    distances_brisk_a_i = []
    distances_brisk_tnr_i = []
    distances_brisk_cn_i = []
    distances_brisk_c_i = []
    distances_brisk_cs_i = []
    distances_brisk_al_i = []
    distances_brisk_a_b = []
    distances_brisk_tnr_b = []
    distances_brisk_cn_b = []
    distances_brisk_c_b = []
    distances_brisk_cs_b = []
    distances_brisk_al_b = []
    if flag_i == 1:
        distances_brisk.append(distances_brisk_a_i)
        distances_brisk.append(distances_brisk_tnr_i)
        distances_brisk.append(distances_brisk_cn_i)
        distances_brisk.append(distances_brisk_c_i)
        distances_brisk.append(distances_brisk_cs_i)
        distances_brisk.append(distances_brisk_al_i)
    if flag_b == 1:
        distances_brisk.append(distances_brisk_a_b)
        distances_brisk.append(distances_brisk_tnr_b)
        distances_brisk.append(distances_brisk_cn_b)
        distances_brisk.append(distances_brisk_c_b)
        distances_brisk.append(distances_brisk_cs_b)
        distances_brisk.append(distances_brisk_al_b)
    # main comparison loop
    for iterator in range(26):
        alphabets_iterator = 0
        for alphabet_with_pad in Alphabets_pad:
            i = cv2.imread('found_pad/found_letter_pad_' + str(k) + '.png')
            j = alphabet_with_pad[iterator]
            # KAZE
            kaze = cv2.KAZE_create()
            keypoints1, descriptors1 = kaze.detectAndCompute(i, None)
            keypoints2, descriptors2 = kaze.detectAndCompute(j, None)
            if descriptors1 is None or descriptors2 is None:
                pass
            else:
                flann_index_kdtree = 1
                index_params = dict(algorithm=flann_index_kdtree, trees=5)
                search_params = dict(checks=100)
                descriptors1 = np.float32(descriptors1)
                descriptors2 = np.float32(descriptors2)
                flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
                matches = flann.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
                matches2 = []
                for m, n in matches:
                    matches2.append(m)
                if len(matches2) > 0:
                    min_distance = matches2[0]
                    for m in matches2:
                        current_distance = m
                        if min_distance.distance > current_distance.distance:
                            min_distance = current_distance
                    if min_distance.distance <= 0.091:
                        output3 = cv2.drawMatches(img1=i, keypoints1=keypoints1, img2=j,
                                                  keypoints2=keypoints2, matches1to2=matches2, outImg=None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + '.jpg', output3)  #
                        cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + ' i.jpg',
                                    i)  # saving image with drawn matches
                        cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + ' j.jpg', j)  #


                def key_sort(e):
                    return e.distance


                def geo_mean_overflow(iterable):
                    a = np.log(iterable)
                    return np.exp(a.mean())


                matches2.sort(key=key_sort)
                sum_dist = 0
                iter_check = 0
                avg_dist = None
                sum_dist_array = []
                for sum_iter in range(mean_range):
                    try:
                        if flag_mean == 1:
                            sum_dist = sum_dist + matches2[sum_iter].distance  # mean distance calculating
                        if flag_geometric_mean == 1:
                            sum_dist_array.append(matches2[sum_iter].distance)  # geometric mean calculating
                        # print("sum dist= ", sum_dist, " matches2[sum_iter]= ", matches2[sum_iter].distance)
                        iter_check = iter_check + 1
                    except:
                        pass
                if flag_mean == 1:
                    avg_dist = sum_dist / iter_check  # mean distance calculating
                if flag_geometric_mean == 1:
                    avg_dist = geo_mean_overflow(sum_dist_array)  # geometric mean calculating
                if flag_minimum == 1:
                    avg_dist = matches2[0].distance  # minimum distance calculating
                # print("avg_dist= ", avg_dist)
                distances_kaze[alphabets_iterator].append(avg_dist)
            # BRISK
            brisk = cv2.BRISK_create()
            keypoints1, descriptors1 = brisk.detectAndCompute(i, None)
            keypoints2, descriptors2 = brisk.detectAndCompute(j, None)
            if descriptors1 is None or descriptors2 is None:
                pass
            else:
                bfmatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
                matches = bfmatcher.match(queryDescriptors=descriptors1, trainDescriptors=descriptors2)
                matches = sorted(matches, key=lambda x: x.distance)
                flann_index_kdtree = 1
                index_params = dict(algorithm=flann_index_kdtree, trees=5)
                search_params = dict(checks=100)
                descriptors1 = np.float32(descriptors1)
                descriptors2 = np.float32(descriptors2)
                flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
                matches = flann.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
                matches3 = []
                for m, n in matches:
                    matches3.append(m)
                if len(matches3) > 0:
                    min_distance = matches3[0]
                    for m in matches3:
                        current_distance = m
                        if min_distance.distance > current_distance.distance:
                            min_distance = current_distance
                    if min_distance.distance <= 490:
                        output3 = cv2.drawMatches(img1=i, keypoints1=keypoints1, img2=j, keypoints2=keypoints2,
                                                  matches1to2=matches3, outImg=None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                        cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + '.jpg', output3)  #
                        cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + ' i.jpg',
                                    i)  # saving image with drawn matches
                        cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + ' j.jpg', j)  #


                def key_sort(e):
                    return e.distance


                def geo_mean_overflow(iterable):
                    try:
                        a = np.log(iterable)
                    except:
                        a = None
                    return np.exp(a.mean())


                matches3.sort(key=key_sort)
                sum_dist = 0
                sum_dist_array = []
                iter_check = 0
                avg_dist = None
                for sum_iter in range(mean_range):
                    try:
                        if flag_mean == 1:
                            sum_dist = sum_dist + matches3[sum_iter].distance  # mean distance calculating
                        if flag_geometric_mean == 1:
                            sum_dist_array.append(matches3[sum_iter].distance)  # geometric mean calculating
                        iter_check = iter_check + 1
                    except:
                        pass
                if flag_mean == 1:
                    avg_dist = sum_dist / iter_check  # mean distance calculating
                if flag_geometric_mean == 1:
                    avg_dist = geo_mean_overflow(sum_dist_array)  # geometric mean calculating
                if flag_minimum == 1:
                    avg_dist = matches3[0].distance  # minimum distance calculating
                distances_brisk[alphabets_iterator].append(avg_dist)
            alphabets_iterator = alphabets_iterator + 1

        if iterator == 25:
            check_if(pre_output_000, distances_brisk[0], distances_brisk_a)
            check_if(pre_output_001, distances_brisk[1], distances_brisk_tnr)
            check_if(pre_output_002, distances_brisk[2], distances_brisk_cn)
            check_if(pre_output_003, distances_brisk[3], distances_brisk_c)
            check_if(pre_output_004, distances_brisk[4], distances_brisk_cs)
            check_if(pre_output_005, distances_brisk[5], distances_brisk_al)
            check_if(pre_output_100, distances_kaze[0], distances_kaze_a)
            check_if(pre_output_101, distances_kaze[1], distances_kaze_tnr)
            check_if(pre_output_102, distances_kaze[2], distances_kaze_cn)
            check_if(pre_output_103, distances_kaze[3], distances_kaze_c)
            check_if(pre_output_104, distances_kaze[4], distances_kaze_cs)
            check_if(pre_output_105, distances_kaze[5], distances_kaze_al)
            if flag_i == 1:
                check_if(pre_output_200, distances_brisk[6], distances_brisk_a_i)
                check_if(pre_output_201, distances_brisk[7], distances_brisk_tnr_i)
                check_if(pre_output_202, distances_brisk[8], distances_brisk_cn_i)
                check_if(pre_output_203, distances_brisk[9], distances_brisk_c_i)
                check_if(pre_output_204, distances_brisk[10], distances_brisk_cs_i)
                check_if(pre_output_205, distances_brisk[11], distances_brisk_al)
                check_if(pre_output_300, distances_kaze[6], distances_kaze_a_i)
                check_if(pre_output_301, distances_kaze[7], distances_kaze_tnr_i)
                check_if(pre_output_302, distances_kaze[8], distances_kaze_cn_i)
                check_if(pre_output_303, distances_kaze[9], distances_kaze_c_i)
                check_if(pre_output_304, distances_kaze[10], distances_kaze_cs_i)
                check_if(pre_output_305, distances_kaze[11], distances_kaze_al_i)
            if flag_b == 1:
                check_if(pre_output_400, distances_brisk[12], distances_brisk_a_b)
                check_if(pre_output_401, distances_brisk[13], distances_brisk_tnr_b)
                check_if(pre_output_402, distances_brisk[14], distances_brisk_cn_b)
                check_if(pre_output_403, distances_brisk[15], distances_brisk_c_b)
                check_if(pre_output_404, distances_brisk[16], distances_brisk_cs_b)
                check_if(pre_output_405, distances_brisk[17], distances_brisk_al_b)
                check_if(pre_output_500, distances_kaze[12], distances_kaze_a_b)
                check_if(pre_output_501, distances_kaze[13], distances_kaze_tnr_b)
                check_if(pre_output_502, distances_kaze[14], distances_kaze_cn_b)
                check_if(pre_output_503, distances_kaze[15], distances_kaze_c_b)
                check_if(pre_output_504, distances_kaze[16], distances_kaze_cs_b)
                check_if(pre_output_505, distances_kaze[17], distances_kaze_al_b)
        z = z + 1
        # return background.convert('RGB')

    # resize_with_pad_v2(im, w + 40, h + 40)
    print(k, "from", number_of_found_letters)

print()

Pre_outputs = [pre_output_000, pre_output_001, pre_output_002, pre_output_003, pre_output_004, pre_output_005,
               pre_output_100, pre_output_101, pre_output_102, pre_output_103, pre_output_104, pre_output_105]
if flag_i == 1:
    Pre_outputs2 = [pre_output_200, pre_output_201, pre_output_202, pre_output_203, pre_output_204, pre_output_205,
                    pre_output_300, pre_output_301, pre_output_302, pre_output_303, pre_output_304, pre_output_305]
    Pre_outputs = Pre_outputs + Pre_outputs2
if flag_b == 1:
    Pre_outputs3 = [pre_output_400, pre_output_401, pre_output_402, pre_output_403, pre_output_404, pre_output_405,
                    pre_output_500, pre_output_501, pre_output_502, pre_output_503, pre_output_504, pre_output_505]

Semi_final_output = [[] * len(Pre_outputs) for i in range(len(Pre_outputs[0]))]

Final_output = []

for iterator_z in range(len(pre_output_000)):
    for output in Pre_outputs:
        try:
            Semi_final_output[iterator_z].append(output[iterator_z])
        except:
            pass


def most_frequent(frequency_list):  # searching most frequent letter in all lines of complexity
    return max(set(frequency_list), key=frequency_list.count)


for semi in Semi_final_output:
    Final_output.append(most_frequent(semi))
try:
    text_file = open("ocered_text.txt", "w")
except FileExistsError:
    text_file = open("ocered_text.txt", "r+")
    text_file.truncate(0)
    text_file.close()
    text_file = open("ocered_text.txt", "a")
for letter_x in range(len(Final_output)):
    print(Final_output[letter_x], sep=' ', end='', flush=True)
    text_file.write(Final_output[letter_x])
print()
'''
Final_output.reverse()
for letter_x in range(len(Final_output)):
    print(Final_output[letter_x], sep=' ', end='', flush=True)
    text_file.write(Final_output[letter_x])
'''
