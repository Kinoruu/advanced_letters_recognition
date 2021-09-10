# import numpy as np
import cv2
# import os

letters = []
Arial_Letters = []
Arial_Letters_italic = []
Arial_Letters_bold = []
Times_NR_Letters = []
Times_NR_Letters_italic = []
Times_NR_Letters_bold = []
Courier_N_Letters = []
Courier_N_Letters_italic = []
Courier_N_Letters_bold = []
Comic_S_Letters = []
Comic_S_Letters_italic = []
Comic_S_Letters_bold = []
Calibri_Letters = []
Calibri_Letters_italic = []
Calibri_Letters_bold = []
Alef_Letters = []
Alef_Letters_italic = []
Alef_Letters_bold = []
Deja_Vu_Sans_Letters = []
Deja_Vu_Sans_Letters_italic = []
Deja_Vu_Sans_Letters_bold = []
Georgia_Letters = []
Georgia_Letters_italic = []
Georgia_Letters_bold = []
Impact_Letters = []
Impact_Letters_italic = []
Impact_Letters_bold = []
Ink_Free_Letters = []
Ink_Free_Letters_italic = []
Ink_Free_Letters_bold = []

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z']


def import_letters_of_alphabet(folder_name):
    list_name = []
    for letter in letters:
        l_f = cv2.imread(filename=folder_name + '/' + letter + '.png', flags=cv2.IMREAD_GRAYSCALE)
        list_name.append(l_f)

    return list_name


Alef_Letters = import_letters_of_alphabet("alphabet_alef")
Alef_Letters_italic = import_letters_of_alphabet("alphabet_alef_italic")
Alef_Letters_bold = import_letters_of_alphabet("alphabet_alef_bold")

Arial_Letters = import_letters_of_alphabet("alphabet_arial")
Arial_Letters_italic = import_letters_of_alphabet("alphabet_arial_italic")
Arial_Letters_bold = import_letters_of_alphabet("alphabet_arial_bold")

Times_NR_Letters = import_letters_of_alphabet("alphabet_times_new_roman")
Times_NR_Letters_italic = import_letters_of_alphabet("alphabet_times_new_roman_italic")
Times_NR_Letters_bold = import_letters_of_alphabet("alphabet_times_new_roman_bold")

Calibri_Letters = import_letters_of_alphabet("alphabet_calibri")
Calibri_Letters_italic = import_letters_of_alphabet("alphabet_calibri_italic")
Calibri_Letters_bold = import_letters_of_alphabet("alphabet_calibri_bold")

Comic_S_Letters = import_letters_of_alphabet("alphabet_comic_sans")
Comic_S_Letters_italic = import_letters_of_alphabet("alphabet_comic_sans_italic")
Comic_S_Letters_bold = import_letters_of_alphabet("alphabet_comic_sans_bold")

Courier_N_Letters = import_letters_of_alphabet("alphabet_courier_new")
Courier_N_Letters_italic = import_letters_of_alphabet("alphabet_courier_new_italic")
Courier_N_Letters_bold = import_letters_of_alphabet("alphabet_courier_new_bold")

Deja_Vu_Sans_Letters = import_letters_of_alphabet("alphabet_deja_vu_sans")
Deja_Vu_Sans_Letters_italic = import_letters_of_alphabet("alphabet_deja_vu_sans_italic")
Deja_Vu_Sans_Letters_bold = import_letters_of_alphabet("alphabet_deja_vu_sans_bold")

Georgia_Letters = import_letters_of_alphabet("alphabet_georgia")
Georgia_Letters_italic = import_letters_of_alphabet("alphabet_georgia_italic")
Georgia_Letters_bold = import_letters_of_alphabet("alphabet_georgia_bold")

Impact_Letters = import_letters_of_alphabet("alphabet_impact")
Impact_Letters_italic = import_letters_of_alphabet("alphabet_impact_italic")
Impact_Letters_bold = import_letters_of_alphabet("alphabet_impact_bold")

Ink_Free_Letters = import_letters_of_alphabet("alphabet_ink_free")
Ink_Free_Letters_italic = import_letters_of_alphabet("alphabet_ink_free_italic")
Ink_Free_Letters_bold = import_letters_of_alphabet("alphabet_ink_free_bold")
