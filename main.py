import numpy as np
import cv2
from PIL import Image
import os
import glob
import letters_import
import operator
import time
######################################################## creating or cleaning directories
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
######################################################## flags
falg_i = 0 # italic flag
flag_b = 0 # bold flag
######################################################## creating vectors for alphabetes and importing from "import_letters.py"
Alphabets = []
Alphabets.append(letters_import.Arial_Letters)
Alphabets.append(letters_import.Times_NR_Letters)
Alphabets.append(letters_import.Courier_N_Letters)
Alphabets.append(letters_import.Calibri_Letters)
Alphabets.append(letters_import.Comic_S_Letters)
if(falg_i == 1):
    Alphabets.append(letters_import.Arial_Letters_italic)
    Alphabets.append(letters_import.Times_NR_Letters_italic)
    Alphabets.append(letters_import.Courier_N_Letters_italic)
    Alphabets.append(letters_import.Calibri_Letters_italic)
    Alphabets.append(letters_import.Comic_S_Letters_italic)
Alphabet_Letters = letters_import.letters
alphabets_names = []
alphabets_names.append("Arial_Letters")
alphabets_names.append("Times_NR_Letters")
alphabets_names.append("Courier_N_Letters")
alphabets_names.append("Calibri_Letters")
alphabets_names.append("Comic_S_Letters")
if(falg_i == 1):
    alphabets_names.append("Arial_Letters_italic")
    alphabets_names.append("Times_NR_Letters_italic")
    alphabets_names.append("Courier_N_Letters_italic")
    alphabets_names.append("Calibri_Letters_italic")
    alphabets_names.append("Comic_S_Letters_italic")
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
if(falg_i == 1):
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
######################################################## adding pads to libraries alphabets letters
iterator_empty = 0
for alphabet in Alphabets_pad:
    iterator_full = 0
    for alphabet2 in Alphabets:
        temp = []
        if(iterator_empty == iterator_full):
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
                resize_with_pad(letter_not_array, w + 40, h + 40)
        iterator_full = iterator_full + 1
    iterator_empty = iterator_empty + 1
######################################################## searching all shapes in the input image
gray = cv2.imread(filename = 'test2.png', flags = cv2.IMREAD_GRAYSCALE)
gray = np.array(gray)
height, width = gray.shape
white = 0
for h in range(height):
    for w in range(width):
        if(gray[h,w] == [255]):
            white = white + 1
if(white < (height * width) / 2):
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
######################################################## deleting shapes included in bigger ones
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
        if(iter1 == iter2):
            pass
        else:
            x2, y2, w2, h2 = cv2.boundingRect(d)
            if ((x >= x2) and (y >= y2) and ((x + w) <= (x2 + w2)) and ((y + h) <= (y2 + h2))):
                found_letter.pop()
number_of_found_letters = len(found_letter) # number of found shapes
print("OCR have found ", number_of_found_letters, " letters")
######################################################## creating vectors for output
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
if(falg_i == 1):
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
program_output = []
######################################################## main part of program
for c in found_letter:
    x, y, w, h = cv2.boundingRect(c)
    k=k+1
    im = gray[y-5:y + h+5, x-5:x + w+5]      # cutting shapes from input image with 1 pixel pad on every side
    cv2.imwrite('found/found_letter_' + str(k) + '.png', im)
    im2 = cv2.imread('found/found_letter_' + str(k) + '.png')
    im = Image.fromarray(im)
    def resize_with_pad(im, target_width, target_height): # function adding pads to found shapes
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

        z = 0  # iterator for naming images containing matches
        Distances_kaze = []
        Distances_kaze_a = []
        Distances_kaze.append(Distances_kaze_a)
        Distances_kaze_tnr = []
        Distances_kaze.append(Distances_kaze_tnr)
        Distances_kaze_cn = []
        Distances_kaze.append(Distances_kaze_cn)
        Distances_kaze_c = []
        Distances_kaze.append(Distances_kaze_c)
        Distances_kaze_cs = []
        Distances_kaze.append(Distances_kaze_cs)
        if (falg_i == 1):
            Distances_kaze_a_i = []
            Distances_kaze.append(Distances_kaze_a_i)
            Distances_kaze_tnr_i = []
            Distances_kaze.append(Distances_kaze_tnr_i)
            Distances_kaze_cn_i = []
            Distances_kaze.append(Distances_kaze_cn_i)
            Distances_kaze_c_i = []
            Distances_kaze.append(Distances_kaze_c_i)
            Distances_kaze_cs_i = []
            Distances_kaze.append(Distances_kaze_cs_i)
        Distances_brisk = []
        Distances_brisk_a = []
        Distances_brisk.append(Distances_brisk_a)
        Distances_brisk_tnr = []
        Distances_brisk.append(Distances_brisk_tnr)
        Distances_brisk_cn = []
        Distances_brisk.append(Distances_brisk_cn)
        Distances_brisk_c = []
        Distances_brisk.append(Distances_brisk_c)
        Distances_brisk_cs = []
        Distances_brisk.append(Distances_brisk_cs)
        if (falg_i == 1):
            Distances_brisk_a_i = []
            Distances_brisk.append(Distances_brisk_a_i)
            Distances_brisk_tnr_i = []
            Distances_brisk.append(Distances_brisk_tnr_i)
            Distances_brisk_cn_i = []
            Distances_brisk.append(Distances_brisk_cn_i)
            Distances_brisk_c_i = []
            Distances_brisk.append(Distances_brisk_c_i)
            Distances_brisk_cs_i = []
            Distances_brisk.append(Distances_brisk_cs_i)
        #################################################### main comparison loop
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
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=100)
                    descriptors1 = np.float32(descriptors1)
                    descriptors2 = np.float32(descriptors2)
                    FLANN = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
                    matches = FLANN.knnMatch(queryDescriptors=descriptors1, trainDescriptors=descriptors2, k=2)
                    matches2 = []
                    for m, n in matches:
                        matches2.append(m)
                    if (len(matches2) > 0):
                        min_distance = matches2[0]
                        for m in matches2:
                            current_distance = m
                            if (min_distance.distance > current_distance.distance):
                                min_distance = current_distance
                        if ((min_distance.distance <= 0.091)):
                            output3 = cv2.drawMatches(img1=i, keypoints1=keypoints1, img2=j,
                                                      keypoints2=keypoints2, matches1to2=matches2, outImg=None,
                                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                            cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + '.jpg', output3) #
                            cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + ' i.jpg', i)     # saving image with drawn matches
                            cv2.imwrite('kaze/image_kaze_best_' + str(z) + " " + str(k) + ' j.jpg', j)     #
                    def key_sort(e):
                        return e.distance
                    def geo_mean_overflow(iterable):
                        a = np.log(iterable)
                        return np.exp(a.mean())
                    matches2.sort(key=key_sort)
                    sum_dist = 0
                    iter_check = 0
                    sum_dist_array = []
                    for sum_iter in range(2):
                        try:
                            sum_dist = sum_dist + matches2[sum_iter].distance    # mean distance calculating
                            sum_dist_array.append(matches2[sum_iter].distance)   # geometric mean calculating
                            #print("sum dist= ", sum_dist, " matches2[sum_iter]= ", matches2[sum_iter].distance)
                            iter_check = iter_check + 1
                        except:
                            pass
                    #avg_dist = sum_dist / iter_check                             # mean distance calculating
                    avg_dist = geo_mean_overflow(sum_dist_array)                 # geometric mean calculating
                    #avg_dist = matches2[0].distance                              # minimum distance calculating
                    #print("avg_dist= ", avg_dist)
                    Distances_kaze[alphabets_iterator].append(avg_dist)
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
                    matches3 = []
                    for m, n in matches:
                        matches3.append(m)
                    if (len(matches3) > 0):
                        min_distance = matches3[0]
                        for m in matches3:
                            current_distance = m
                            if (min_distance.distance > current_distance.distance):
                                min_distance = current_distance
                        if ((min_distance.distance <= 490)):
                            output3 = cv2.drawMatches(img1=i, keypoints1=keypoints1, img2=j, keypoints2=keypoints2,
                                                      matches1to2=matches3, outImg=None,
                                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                            cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + '.jpg', output3) #
                            cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + ' i.jpg', i)     # saving image with drawn matches
                            cv2.imwrite('brisk/image_brisk_best_' + str(z) + " " + str(k) + ' j.jpg', j)     #
                    def key_sort(e):
                        return e.distance
                    def geo_mean_overflow(iterable):
                        a = np.log(iterable)
                        return np.exp(a.mean())
                    matches3.sort(key= key_sort)
                    sum_dist = 0
                    sum_dist_array = []
                    iter_check = 0
                    for sum_iter in range(2):
                        try:
                            sum_dist = sum_dist + matches3[sum_iter].distance   # mean distance calculating
                            sum_dist_array.append(matches3[sum_iter].distance)  # geometric mean calculating
                            iter_check = iter_check + 1
                        except: pass
                    #avg_dist = sum_dist / iter_check                            # mean distance calculating
                    avg_dist = geo_mean_overflow(sum_dist_array)                # geometric mean calculating
                    #avg_dist = matches3[0].distance                             # minimum distance calculating
                    Distances_brisk[alphabets_iterator].append(avg_dist)
                alphabets_iterator = alphabets_iterator + 1

            if(iterator == 25):
                try:
                    pre_output_0.append((Alphabet_Letters[Distances_brisk[0].index(min(Distances_brisk_a))]))
                except:
                    pass
                try:
                    pre_output_1.append((Alphabet_Letters[Distances_brisk[1].index(min(Distances_brisk_tnr))]))
                except:
                    pass
                try:
                    pre_output_2.append((Alphabet_Letters[Distances_brisk[2].index(min(Distances_brisk_cn))]))
                except:
                    pass
                try:
                    pre_output_3.append((Alphabet_Letters[Distances_brisk[3].index(min(Distances_brisk_c))]))
                except:
                    pass
                try:
                    pre_output_4.append((Alphabet_Letters[Distances_brisk[4].index(min(Distances_brisk_cs))]))
                except:
                    pass
                try:
                    pre_output_5.append((Alphabet_Letters[Distances_kaze[0].index(min(Distances_kaze_a))]))
                except:
                    pass
                try:
                    pre_output_6.append((Alphabet_Letters[Distances_kaze[1].index(min(Distances_kaze_tnr))]))
                except:
                    pass
                try:
                    pre_output_7.append((Alphabet_Letters[Distances_kaze[2].index(min(Distances_kaze_cn))]))
                except:
                    pass
                try:
                    pre_output_8.append((Alphabet_Letters[Distances_kaze[3].index(min(Distances_kaze_c))]))
                except:
                    pass
                try:
                    pre_output_9.append((Alphabet_Letters[Distances_kaze[4].index(min(Distances_kaze_cs))]))
                except:
                    pass
                if (falg_i == 1):
                    try:
                        pre_output_10.append((Alphabet_Letters[Distances_brisk[5].index(min(Distances_brisk_a_i))]))
                    except:
                        pass
                    try:
                        pre_output_11.append((Alphabet_Letters[Distances_brisk[6].index(min(Distances_brisk_tnr_i))]))
                    except:
                        pass
                    try:
                        pre_output_12.append((Alphabet_Letters[Distances_brisk[7].index(min(Distances_brisk_cn_i))]))
                    except:
                        pass
                    try:
                        pre_output_13.append((Alphabet_Letters[Distances_brisk[8].index(min(Distances_brisk_c_i))]))
                    except:
                        pass
                    try:
                        pre_output_14.append((Alphabet_Letters[Distances_brisk[9].index(min(Distances_brisk_cs_i))]))
                    except:
                        pass
                    try:
                        pre_output_15.append((Alphabet_Letters[Distances_kaze[5].index(min(Distances_kaze_a_i))]))
                    except:
                        pass
                    try:
                        pre_output_16.append((Alphabet_Letters[Distances_kaze[6].index(min(Distances_kaze_tnr_i))]))
                    except:
                        pass
                    try:
                        pre_output_17.append((Alphabet_Letters[Distances_kaze[7].index(min(Distances_kaze_cn_i))]))
                    except:
                        pass
                    try:
                        pre_output_18.append((Alphabet_Letters[Distances_kaze[8].index(min(Distances_kaze_c_i))]))
                    except:
                        pass
                    try:
                        pre_output_19.append((Alphabet_Letters[Distances_kaze[9].index(min(Distances_kaze_cs_i))]))
                    except:
                        pass
            z = z + 1
        return background.convert('RGB')
    resize_with_pad(im, w + 40, h + 40)
    print(k, "from", number_of_found_letters)

print()

Pre_outputs = []
Pre_outputs.append(pre_output_0)
Pre_outputs.append(pre_output_1)
Pre_outputs.append(pre_output_2)
Pre_outputs.append(pre_output_3)
Pre_outputs.append(pre_output_4)
Pre_outputs.append(pre_output_5)
Pre_outputs.append(pre_output_6)
Pre_outputs.append(pre_output_7)
Pre_outputs.append(pre_output_8)
Pre_outputs.append(pre_output_9)
if(falg_i == 1):
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

Semi_final_output = [[] * len(Pre_outputs) for i in range(len(Pre_outputs[0]))]

Final_output = []

for iterator_z in range(len(pre_output_0)):
    for output in Pre_outputs:
        try:
            Semi_final_output[iterator_z].append(output[iterator_z])
        except:
            pass

def most_frequent(List):  # searching most frequent letter in all lines of complexity
    return max(set(List), key = List.count)

for semi in Semi_final_output:
    Final_output.append(most_frequent(semi))

text_file = open("ocered_text.txt","w")
text_file = open("ocered_text.txt","a")
for letter_x in range(len(Final_output)):
    print(Final_output[letter_x], sep=' ', end='', flush=True)
    text_file.write(Final_output[letter_x])
print()
Final_output.reverse()
for letter_x in range(len(Final_output)):
    print(Final_output[letter_x], sep=' ', end='', flush=True)
    text_file.write(Final_output[letter_x])



