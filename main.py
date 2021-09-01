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
    os.mkdir('found_pad')
except FileExistsError:
    for root, dirs, files in os.walk('found_pad'):
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
# flags and changeable elements
flag_i = 0  # italic flag
flag_b = 0  # bold flag
mean_range = 4
flag_geometric_mean = 1
flag_mean = 0
flag_minimum = 0
height_pad = 5
width_pad = 5
# creating vectors for alphabets and importing from "import_letters.py"
Alphabets = [letters_import.Arial_Letters, letters_import.Times_NR_Letters, letters_import.Courier_N_Letters,
             letters_import.Calibri_Letters, letters_import.Comic_S_Letters]
if flag_i == 1:
    Alphabets.append(letters_import.Arial_Letters_italic)
    Alphabets.append(letters_import.Times_NR_Letters_italic)
    Alphabets.append(letters_import.Courier_N_Letters_italic)
    Alphabets.append(letters_import.Calibri_Letters_italic)
    Alphabets.append(letters_import.Comic_S_Letters_italic)
if flag_b == 1:
    Alphabets.append(letters_import.Arial_Letters_bold)
    Alphabets.append(letters_import.Times_NR_Letters_bold)
    Alphabets.append(letters_import.Courier_N_Letters_bold)
    Alphabets.append(letters_import.Calibri_Letters_bold)
    Alphabets.append(letters_import.Comic_S_Letters_bold)
Alphabet_Letters = letters_import.letters
alphabets_names = ["Arial_Letters", "Times_NR_Letters", "Courier_N_Letters", "Calibri_Letters", "Comic_S_Letters"]
if flag_i == 1:
    alphabets_names.append("Arial_Letters_italic")
    alphabets_names.append("Times_NR_Letters_italic")
    alphabets_names.append("Courier_N_Letters_italic")
    alphabets_names.append("Calibri_Letters_italic")
    alphabets_names.append("Comic_S_Letters_italic")
if flag_b == 1:
    alphabets_names.append("Arial_Letters_bold")
    alphabets_names.append("Times_NR_Letters_bold")
    alphabets_names.append("Courier_N_Letters_bold")
    alphabets_names.append("Calibri_Letters_bold")
    alphabets_names.append("Comic_S_Letters_bold")
Alphabets_pad = []
Arial_Letters_no_pad = []
Alphabets_pad.append(Arial_Letters_no_pad)
Times_NR_Letters_no_pad = []
Alphabets_pad.append(Times_NR_Letters_no_pad)
Courier_N_Letters_no_pad = []
Alphabets_pad.append(Courier_N_Letters_no_pad)
Comic_S_Letters_no_pad = []
Alphabets_pad.append(Comic_S_Letters_no_pad)
Calibri_Letters_no_pad = []
Alphabets_pad.append(Calibri_Letters_no_pad)
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
gray = cv2.imread(filename='letters_cn.png', flags=cv2.IMREAD_GRAYSCALE)
gray = np.array(gray)
height, width = gray.shape
white = 0
for h in range(height):
    for w in range(width):
        if gray[h, w] == [255]:
            white = white + 1
if white < (height * width) / 2:
    gray = cv2.cv2.bitwise_not(gray)
ret, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_OTSU)
inverted_binary = ~binary
contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
'''
for shapes in contours:
    shapes = Image.fromarray(shapes)
    height, width = shapes.shape

    for h in range(height):
        for w in range(width):
            if(shapes[h,w] == shapes):
                shapes[h,w] = [255,255,255]
'''
# deleting shapes included in bigger ones
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
        if iter1 == iter2:
            pass
        else:
            x2, y2, w2, h2 = cv2.boundingRect(d)
            if (x >= x2) and (y >= y2) and ((x + w) <= (x2 + w2)) and ((y + h) <= (y2 + h2)):
                found_letter.pop()
number_of_found_letters = len(found_letter)  # number of found shapes
print("OCR have found ", number_of_found_letters, " letters")
# creating vectors for output
pre_output_0 = []
pre_output_1 = []
pre_output_2 = []
pre_output_3 = []
pre_output_4 = []
pre_output_5 = []
pre_output_6 = []
pre_output_7 = []
pre_output_8 = []
pre_output_9 = []
pre_output_10 = []
pre_output_11 = []
pre_output_12 = []
pre_output_13 = []
pre_output_14 = []
pre_output_15 = []
pre_output_16 = []
pre_output_17 = []
pre_output_18 = []
pre_output_19 = []
pre_output_20 = []
pre_output_21 = []
pre_output_22 = []
pre_output_23 = []
pre_output_24 = []
pre_output_25 = []
pre_output_26 = []
pre_output_27 = []
pre_output_28 = []
pre_output_29 = []
program_output = []
# main part of program
for c in found_letter:
    x, y, w, h = cv2.boundingRect(c)
    k = k + 1
    im = gray[y - height_pad:y + h + height_pad, x - width_pad:x + w + width_pad]  # cutting shapes from input image
    cv2.imwrite('found/found_letter_' + str(k) + '.png', im)
    im2 = cv2.imread('found/found_letter_' + str(k) + '.png')
    im = Image.fromarray(im)

    def resize_with_pad(image, target_width, target_height):  # function adding pads to found shapes
        background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
        offset = (round((target_width - image.width) / 2), round((target_height - image.height) / 2))
        background.paste(image, offset)
        background.save('found_pad/found_letter_pad_' + str(k) + '.png')

        z = 0  # iterator for naming images containing matches
        distances_kaze = []
        distances_kaze_a = []
        distances_kaze.append(distances_kaze_a)
        distances_kaze_tnr = []
        distances_kaze.append(distances_kaze_tnr)
        distances_kaze_cn = []
        distances_kaze.append(distances_kaze_cn)
        distances_kaze_c = []
        distances_kaze.append(distances_kaze_c)
        distances_kaze_cs = []
        distances_kaze.append(distances_kaze_cs)
        distances_kaze_a_i = []
        distances_kaze_tnr_i = []
        distances_kaze_cn_i = []
        distances_kaze_c_i = []
        distances_kaze_cs_i = []
        distances_kaze_a_b = []
        distances_kaze_tnr_b = []
        distances_kaze_cn_b = []
        distances_kaze_c_b = []
        distances_kaze_cs_b = []
        if flag_i == 1:
            distances_kaze.append(distances_kaze_a_i)
            distances_kaze.append(distances_kaze_tnr_i)
            distances_kaze.append(distances_kaze_cn_i)
            distances_kaze.append(distances_kaze_c_i)
            distances_kaze.append(distances_kaze_cs_i)
        if flag_b == 1:
            distances_kaze.append(distances_kaze_a_b)
            distances_kaze.append(distances_kaze_tnr_b)
            distances_kaze.append(distances_kaze_cn_b)
            distances_kaze.append(distances_kaze_c_b)
            distances_kaze.append(distances_kaze_cs_b)
        distances_brisk = []
        distances_brisk_a = []
        distances_brisk.append(distances_brisk_a)
        distances_brisk_tnr = []
        distances_brisk.append(distances_brisk_tnr)
        distances_brisk_cn = []
        distances_brisk.append(distances_brisk_cn)
        distances_brisk_c = []
        distances_brisk.append(distances_brisk_c)
        distances_brisk_cs = []
        distances_brisk.append(distances_brisk_cs)
        distances_brisk_a_i = []
        distances_brisk_tnr_i = []
        distances_brisk_cn_i = []
        distances_brisk_c_i = []
        distances_brisk_cs_i = []
        distances_brisk_a_b = []
        distances_brisk_tnr_b = []
        distances_brisk_cn_b = []
        distances_brisk_c_b = []
        distances_brisk_cs_b = []
        if flag_i == 1:
            distances_brisk.append(distances_brisk_a_i)
            distances_brisk.append(distances_brisk_tnr_i)
            distances_brisk.append(distances_brisk_cn_i)
            distances_brisk.append(distances_brisk_c_i)
            distances_brisk.append(distances_brisk_cs_i)
        if flag_b == 1:
            distances_brisk.append(distances_brisk_a_b)
            distances_brisk.append(distances_brisk_tnr_b)
            distances_brisk.append(distances_brisk_cn_b)
            distances_brisk.append(distances_brisk_c_b)
            distances_brisk.append(distances_brisk_cs_b)
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
                        a = np.log(iterable)
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
                try:
                    pre_output_0.append((Alphabet_Letters[distances_brisk[0].index(min(distances_brisk_a))]))
                except:
                    pass
                try:
                    pre_output_1.append((Alphabet_Letters[distances_brisk[1].index(min(distances_brisk_tnr))]))
                except:
                    pass
                try:
                    pre_output_2.append((Alphabet_Letters[distances_brisk[2].index(min(distances_brisk_cn))]))
                except:
                    pass
                try:
                    pre_output_3.append((Alphabet_Letters[distances_brisk[3].index(min(distances_brisk_c))]))
                except:
                    pass
                try:
                    pre_output_4.append((Alphabet_Letters[distances_brisk[4].index(min(distances_brisk_cs))]))
                except:
                    pass
                try:
                    pre_output_5.append((Alphabet_Letters[distances_kaze[0].index(min(distances_kaze_a))]))
                except:
                    pass
                try:
                    pre_output_6.append((Alphabet_Letters[distances_kaze[1].index(min(distances_kaze_tnr))]))
                except:
                    pass
                try:
                    pre_output_7.append((Alphabet_Letters[distances_kaze[2].index(min(distances_kaze_cn))]))
                except:
                    pass
                try:
                    pre_output_8.append((Alphabet_Letters[distances_kaze[3].index(min(distances_kaze_c))]))
                except:
                    pass
                try:
                    pre_output_9.append((Alphabet_Letters[distances_kaze[4].index(min(distances_kaze_cs))]))
                except:
                    pass
                if flag_i == 1:
                    try:
                        pre_output_10.append((Alphabet_Letters[distances_brisk[5].index(min(distances_brisk_a_i))]))
                    except:
                        pass
                    try:
                        pre_output_11.append((Alphabet_Letters[distances_brisk[6].index(min(distances_brisk_tnr_i))]))
                    except:
                        pass
                    try:
                        pre_output_12.append((Alphabet_Letters[distances_brisk[7].index(min(distances_brisk_cn_i))]))
                    except:
                        pass
                    try:
                        pre_output_13.append((Alphabet_Letters[distances_brisk[8].index(min(distances_brisk_c_i))]))
                    except:
                        pass
                    try:
                        pre_output_14.append((Alphabet_Letters[distances_brisk[9].index(min(distances_brisk_cs_i))]))
                    except:
                        pass
                    try:
                        pre_output_15.append((Alphabet_Letters[distances_kaze[5].index(min(distances_kaze_a_i))]))
                    except:
                        pass
                    try:
                        pre_output_16.append((Alphabet_Letters[distances_kaze[6].index(min(distances_kaze_tnr_i))]))
                    except:
                        pass
                    try:
                        pre_output_17.append((Alphabet_Letters[distances_kaze[7].index(min(distances_kaze_cn_i))]))
                    except:
                        pass
                    try:
                        pre_output_18.append((Alphabet_Letters[distances_kaze[8].index(min(distances_kaze_c_i))]))
                    except:
                        pass
                    try:
                        pre_output_19.append((Alphabet_Letters[distances_kaze[9].index(min(distances_kaze_cs_i))]))
                    except:
                        pass
                if flag_b == 1:
                    try:
                        pre_output_10.append((Alphabet_Letters[distances_brisk[10].index(min(distances_brisk_a_b))]))
                    except:
                        pass
                    try:
                        pre_output_11.append((Alphabet_Letters[distances_brisk[11].index(min(distances_brisk_tnr_b))]))
                    except:
                        pass
                    try:
                        pre_output_12.append((Alphabet_Letters[distances_brisk[12].index(min(distances_brisk_cn_b))]))
                    except:
                        pass
                    try:
                        pre_output_13.append((Alphabet_Letters[distances_brisk[13].index(min(distances_brisk_c_b))]))
                    except:
                        pass
                    try:
                        pre_output_14.append((Alphabet_Letters[distances_brisk[14].index(min(distances_brisk_cs_b))]))
                    except:
                        pass
                    try:
                        pre_output_15.append((Alphabet_Letters[distances_kaze[10].index(min(distances_kaze_a_b))]))
                    except:
                        pass
                    try:
                        pre_output_16.append((Alphabet_Letters[distances_kaze[11].index(min(distances_kaze_tnr_b))]))
                    except:
                        pass
                    try:
                        pre_output_17.append((Alphabet_Letters[distances_kaze[12].index(min(distances_kaze_cn_b))]))
                    except:
                        pass
                    try:
                        pre_output_18.append((Alphabet_Letters[distances_kaze[13].index(min(distances_kaze_c_b))]))
                    except:
                        pass
                    try:
                        pre_output_19.append((Alphabet_Letters[distances_kaze[14].index(min(distances_kaze_cs_b))]))
                    except:
                        pass
            z = z + 1
        return background.convert('RGB')


    resize_with_pad(im, w + 40, h + 40)
    print(k, "from", number_of_found_letters)

print()

Pre_outputs = [pre_output_0, pre_output_1, pre_output_2, pre_output_3, pre_output_4, pre_output_5, pre_output_6,
               pre_output_7, pre_output_8, pre_output_9]
if flag_i == 1:
    Pre_outputs.append(pre_output_10)
    Pre_outputs.append(pre_output_11)
    Pre_outputs.append(pre_output_12)
    Pre_outputs.append(pre_output_13)
    Pre_outputs.append(pre_output_14)
    Pre_outputs.append(pre_output_15)
    Pre_outputs.append(pre_output_16)
    Pre_outputs.append(pre_output_17)
    Pre_outputs.append(pre_output_18)
    Pre_outputs.append(pre_output_19)
if flag_b == 1:
    Pre_outputs.append(pre_output_20)
    Pre_outputs.append(pre_output_21)
    Pre_outputs.append(pre_output_22)
    Pre_outputs.append(pre_output_23)
    Pre_outputs.append(pre_output_24)
    Pre_outputs.append(pre_output_25)
    Pre_outputs.append(pre_output_26)
    Pre_outputs.append(pre_output_27)
    Pre_outputs.append(pre_output_28)
    Pre_outputs.append(pre_output_29)

Semi_final_output = [[] * len(Pre_outputs) for i in range(len(Pre_outputs[0]))]

Final_output = []

for iterator_z in range(len(pre_output_0)):
    for output in Pre_outputs:
        try:
            Semi_final_output[iterator_z].append(output[iterator_z])
        except:
            pass


def most_frequent(frequency_list):  # searching most frequent letter in all lines of complexity
    return max(set(frequency_list), key=frequency_list.count)


for semi in Semi_final_output:
    Final_output.append(most_frequent(semi))

text_file = open("ocered_text.txt", "a")
for letter_x in range(len(Final_output)):
    print(Final_output[letter_x], sep=' ', end='', flush=True)
    text_file.write(Final_output[letter_x])
print()
Final_output.reverse()
for letter_x in range(len(Final_output)):
    print(Final_output[letter_x], sep=' ', end='', flush=True)
    text_file.write(Final_output[letter_x])
