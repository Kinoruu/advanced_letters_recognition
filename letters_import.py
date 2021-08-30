import numpy as np
import cv2
import os

letters = []
Arial_Letters = []
Arial_Letters_italic = []
Times_NR_Letters = []
Times_NR_Letters_italic = []
Courier_N_Letters = []
Courier_N_Letters_italic = []
Comic_S_Letters = []
Comic_S_Letters_italic = []
Calibri_Letters = []
Calibri_Letters_italic = []

l_a = cv2.imread(filename = 'alphabet_arial/a.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_a)
letters.append("a")
l_b = cv2.imread(filename = 'alphabet_arial/b.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_b)
letters.append("b")
l_c = cv2.imread(filename = 'alphabet_arial/c.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_c)
letters.append("c")
l_d = cv2.imread(filename = 'alphabet_arial/d.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_d)
letters.append("d")
l_e = cv2.imread(filename = 'alphabet_arial/e.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_e)
letters.append("e")
l_f = cv2.imread(filename = 'alphabet_arial/f.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_f)
letters.append("f")
l_g = cv2.imread(filename = 'alphabet_arial/g.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_g)
letters.append("g")
l_h = cv2.imread(filename = 'alphabet_arial/h.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_h)
letters.append("h")
l_i = cv2.imread(filename = 'alphabet_arial/i.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_i)
letters.append("i")
l_j = cv2.imread(filename = 'alphabet_arial/j.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_j)
letters.append("j")
l_k = cv2.imread(filename = 'alphabet_arial/k.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_k)
letters.append("k")
l_l = cv2.imread(filename = 'alphabet_arial/l.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_l)
letters.append("l")
l_m = cv2.imread(filename = 'alphabet_arial/m.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_m)
letters.append("m")
l_n = cv2.imread(filename = 'alphabet_arial/n.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_n)
letters.append("n")
l_o = cv2.imread(filename = 'alphabet_arial/o.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_o)
letters.append("o")
l_p = cv2.imread(filename = 'alphabet_arial/p.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_p)
letters.append("p")
l_q = cv2.imread(filename = 'alphabet_arial/q.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_q)
letters.append("q")
l_r = cv2.imread(filename = 'alphabet_arial/r.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_r)
letters.append("r")
l_s = cv2.imread(filename = 'alphabet_arial/s.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_s)
letters.append("s")
l_t = cv2.imread(filename = 'alphabet_arial/t.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_t)
letters.append("t")
l_u = cv2.imread(filename = 'alphabet_arial/u.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_u)
letters.append("u")
l_v = cv2.imread(filename = 'alphabet_arial/v.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_v)
letters.append("v")
l_w = cv2.imread(filename = 'alphabet_arial/w.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_w)
letters.append("w")
l_x = cv2.imread(filename = 'alphabet_arial/x.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_x)
letters.append("x")
l_y = cv2.imread(filename = 'alphabet_arial/y.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_y)
letters.append("y")
l_z = cv2.imread(filename = 'alphabet_arial/z.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters.append(l_z)
letters.append("z")

l_a = cv2.imread(filename = 'alphabet_arial_italic/a.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_a)
l_b = cv2.imread(filename = 'alphabet_arial_italic/b.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_b)
l_c = cv2.imread(filename = 'alphabet_arial_italic/c.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_c)
l_d = cv2.imread(filename = 'alphabet_arial_italic/d.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_d)
l_e = cv2.imread(filename = 'alphabet_arial_italic/e.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_e)
l_f = cv2.imread(filename = 'alphabet_arial_italic/f.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_f)
l_g = cv2.imread(filename = 'alphabet_arial_italic/g.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_g)
l_h = cv2.imread(filename = 'alphabet_arial_italic/h.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_h)
l_i = cv2.imread(filename = 'alphabet_arial_italic/i.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_i)
l_j = cv2.imread(filename = 'alphabet_arial_italic/j.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_j)
l_k = cv2.imread(filename = 'alphabet_arial_italic/k.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_k)
l_l = cv2.imread(filename = 'alphabet_arial_italic/l.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_l)
l_m = cv2.imread(filename = 'alphabet_arial_italic/m.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_m)
l_n = cv2.imread(filename = 'alphabet_arial_italic/n.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_n)
l_o = cv2.imread(filename = 'alphabet_arial_italic/o.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_o)
l_p = cv2.imread(filename = 'alphabet_arial_italic/p.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_p)
l_q = cv2.imread(filename = 'alphabet_arial_italic/q.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_q)
l_r = cv2.imread(filename = 'alphabet_arial_italic/r.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_r)
l_s = cv2.imread(filename = 'alphabet_arial_italic/s.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_s)
l_t = cv2.imread(filename = 'alphabet_arial_italic/t.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_t)
l_u = cv2.imread(filename = 'alphabet_arial_italic/u.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_u)
l_v = cv2.imread(filename = 'alphabet_arial_italic/v.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_v)
l_w = cv2.imread(filename = 'alphabet_arial_italic/w.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_w)
l_x = cv2.imread(filename = 'alphabet_arial_italic/x.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_x)
l_y = cv2.imread(filename = 'alphabet_arial_italic/y.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_y)
l_z = cv2.imread(filename = 'alphabet_arial_italic/z.png', flags = cv2.IMREAD_GRAYSCALE)
Arial_Letters_italic.append(l_z)

l_a = cv2.imread(filename = 'alphabet_times_new_roman/a.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_a)
l_b = cv2.imread(filename = 'alphabet_times_new_roman/b.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_b)
l_c = cv2.imread(filename = 'alphabet_times_new_roman/c.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_c)
l_d = cv2.imread(filename = 'alphabet_times_new_roman/d.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_d)
l_e = cv2.imread(filename = 'alphabet_times_new_roman/e.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_e)
l_f = cv2.imread(filename = 'alphabet_times_new_roman/f.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_f)
l_g = cv2.imread(filename = 'alphabet_times_new_roman/g.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_g)
l_h = cv2.imread(filename = 'alphabet_times_new_roman/h.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_h)
l_i = cv2.imread(filename = 'alphabet_times_new_roman/i.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_i)
l_j = cv2.imread(filename = 'alphabet_times_new_roman/j.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_j)
l_k = cv2.imread(filename = 'alphabet_times_new_roman/k.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_k)
l_l = cv2.imread(filename = 'alphabet_times_new_roman/l.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_l)
l_m = cv2.imread(filename = 'alphabet_times_new_roman/m.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_m)
l_n = cv2.imread(filename = 'alphabet_times_new_roman/n.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_n)
l_o = cv2.imread(filename = 'alphabet_times_new_roman/o.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_o)
l_p = cv2.imread(filename = 'alphabet_times_new_roman/p.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_p)
l_q = cv2.imread(filename = 'alphabet_times_new_roman/q.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_q)
l_r = cv2.imread(filename = 'alphabet_times_new_roman/r.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_r)
l_s = cv2.imread(filename = 'alphabet_times_new_roman/s.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_s)
l_t = cv2.imread(filename = 'alphabet_times_new_roman/t.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_t)
l_u = cv2.imread(filename = 'alphabet_times_new_roman/u.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_u)
l_v = cv2.imread(filename = 'alphabet_times_new_roman/v.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_v)
l_w = cv2.imread(filename = 'alphabet_times_new_roman/w.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_w)
l_x = cv2.imread(filename = 'alphabet_times_new_roman/x.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_x)
l_y = cv2.imread(filename = 'alphabet_times_new_roman/y.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_y)
l_z = cv2.imread(filename = 'alphabet_times_new_roman/z.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters.append(l_z)

l_a = cv2.imread(filename = 'alphabet_times_new_roman_italic/a.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_a)
l_b = cv2.imread(filename = 'alphabet_times_new_roman_italic/b.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_b)
l_c = cv2.imread(filename = 'alphabet_times_new_roman_italic/c.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_c)
l_d = cv2.imread(filename = 'alphabet_times_new_roman_italic/d.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_d)
l_e = cv2.imread(filename = 'alphabet_times_new_roman_italic/e.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_e)
l_f = cv2.imread(filename = 'alphabet_times_new_roman_italic/f.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_f)
l_g = cv2.imread(filename = 'alphabet_times_new_roman_italic/g.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_g)
l_h = cv2.imread(filename = 'alphabet_times_new_roman_italic/h.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_h)
l_i = cv2.imread(filename = 'alphabet_times_new_roman_italic/i.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_i)
l_j = cv2.imread(filename = 'alphabet_times_new_roman_italic/j.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_j)
l_k = cv2.imread(filename = 'alphabet_times_new_roman_italic/k.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_k)
l_l = cv2.imread(filename = 'alphabet_times_new_roman_italic/l.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_l)
l_m = cv2.imread(filename = 'alphabet_times_new_roman_italic/m.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_m)
l_n = cv2.imread(filename = 'alphabet_times_new_roman_italic/n.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_n)
l_o = cv2.imread(filename = 'alphabet_times_new_roman_italic/o.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_o)
l_p = cv2.imread(filename = 'alphabet_times_new_roman_italic/p.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_p)
l_q = cv2.imread(filename = 'alphabet_times_new_roman_italic/q.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_q)
l_r = cv2.imread(filename = 'alphabet_times_new_roman_italic/r.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_r)
l_s = cv2.imread(filename = 'alphabet_times_new_roman_italic/s.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_s)
l_t = cv2.imread(filename = 'alphabet_times_new_roman_italic/t.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_t)
l_u = cv2.imread(filename = 'alphabet_times_new_roman_italic/u.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_u)
l_v = cv2.imread(filename = 'alphabet_times_new_roman_italic/v.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_v)
l_w = cv2.imread(filename = 'alphabet_times_new_roman_italic/w.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_w)
l_x = cv2.imread(filename = 'alphabet_times_new_roman_italic/x.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_x)
l_y = cv2.imread(filename = 'alphabet_times_new_roman_italic/y.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_y)
l_z = cv2.imread(filename = 'alphabet_times_new_roman_italic/z.png', flags = cv2.IMREAD_GRAYSCALE)
Times_NR_Letters_italic.append(l_z)

l_a = cv2.imread(filename = 'alphabet_courier_new/a.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_a)
l_b = cv2.imread(filename = 'alphabet_courier_new/b.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_b)
l_c = cv2.imread(filename = 'alphabet_courier_new/c.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_c)
l_d = cv2.imread(filename = 'alphabet_courier_new/d.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_d)
l_e = cv2.imread(filename = 'alphabet_courier_new/e.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_e)
l_f = cv2.imread(filename = 'alphabet_courier_new/f.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_f)
l_g = cv2.imread(filename = 'alphabet_courier_new/g.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_g)
l_h = cv2.imread(filename = 'alphabet_courier_new/h.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_h)
l_i = cv2.imread(filename = 'alphabet_courier_new/i.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_i)
l_j = cv2.imread(filename = 'alphabet_courier_new/j.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_j)
l_k = cv2.imread(filename = 'alphabet_courier_new/k.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_k)
l_l = cv2.imread(filename = 'alphabet_courier_new/l.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_l)
l_m = cv2.imread(filename = 'alphabet_courier_new/m.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_m)
l_n = cv2.imread(filename = 'alphabet_courier_new/n.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_n)
l_o = cv2.imread(filename = 'alphabet_courier_new/o.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_o)
l_p = cv2.imread(filename = 'alphabet_courier_new/p.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_p)
l_q = cv2.imread(filename = 'alphabet_courier_new/q.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_q)
l_r = cv2.imread(filename = 'alphabet_courier_new/r.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_r)
l_s = cv2.imread(filename = 'alphabet_courier_new/s.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_s)
l_t = cv2.imread(filename = 'alphabet_courier_new/t.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_t)
l_u = cv2.imread(filename = 'alphabet_courier_new/u.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_u)
l_v = cv2.imread(filename = 'alphabet_courier_new/v.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_v)
l_w = cv2.imread(filename = 'alphabet_courier_new/w.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_w)
l_x = cv2.imread(filename = 'alphabet_courier_new/x.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_x)
l_y = cv2.imread(filename = 'alphabet_courier_new/y.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_y)
l_z = cv2.imread(filename = 'alphabet_courier_new/z.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters.append(l_z)

l_a = cv2.imread(filename = 'alphabet_courier_new_italic/a.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_a)
l_b = cv2.imread(filename = 'alphabet_courier_new_italic/b.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_b)
l_c = cv2.imread(filename = 'alphabet_courier_new_italic/c.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_c)
l_d = cv2.imread(filename = 'alphabet_courier_new_italic/d.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_d)
l_e = cv2.imread(filename = 'alphabet_courier_new_italic/e.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_e)
l_f = cv2.imread(filename = 'alphabet_courier_new_italic/f.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_f)
l_g = cv2.imread(filename = 'alphabet_courier_new_italic/g.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_g)
l_h = cv2.imread(filename = 'alphabet_courier_new_italic/h.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_h)
l_i = cv2.imread(filename = 'alphabet_courier_new_italic/i.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_i)
l_j = cv2.imread(filename = 'alphabet_courier_new_italic/j.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_j)
l_k = cv2.imread(filename = 'alphabet_courier_new_italic/k.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_k)
l_l = cv2.imread(filename = 'alphabet_courier_new_italic/l.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_l)
l_m = cv2.imread(filename = 'alphabet_courier_new_italic/m.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_m)
l_n = cv2.imread(filename = 'alphabet_courier_new_italic/n.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_n)
l_o = cv2.imread(filename = 'alphabet_courier_new_italic/o.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_o)
l_p = cv2.imread(filename = 'alphabet_courier_new_italic/p.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_p)
l_q = cv2.imread(filename = 'alphabet_courier_new_italic/q.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_q)
l_r = cv2.imread(filename = 'alphabet_courier_new_italic/r.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_r)
l_s = cv2.imread(filename = 'alphabet_courier_new_italic/s.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_s)
l_t = cv2.imread(filename = 'alphabet_courier_new_italic/t.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_t)
l_u = cv2.imread(filename = 'alphabet_courier_new_italic/u.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_u)
l_v = cv2.imread(filename = 'alphabet_courier_new_italic/v.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_v)
l_w = cv2.imread(filename = 'alphabet_courier_new_italic/w.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_w)
l_x = cv2.imread(filename = 'alphabet_courier_new_italic/x.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_x)
l_y = cv2.imread(filename = 'alphabet_courier_new_italic/y.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_y)
l_z = cv2.imread(filename = 'alphabet_courier_new_italic/z.png', flags = cv2.IMREAD_GRAYSCALE)
Courier_N_Letters_italic.append(l_z)

l_a = cv2.imread(filename = 'alphabet_calibri/a.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_a)
l_b = cv2.imread(filename = 'alphabet_calibri/b.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_b)
l_c = cv2.imread(filename = 'alphabet_calibri/c.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_c)
l_d = cv2.imread(filename = 'alphabet_calibri/d.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_d)
l_e = cv2.imread(filename = 'alphabet_calibri/e.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_e)
l_f = cv2.imread(filename = 'alphabet_calibri/f.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_f)
l_g = cv2.imread(filename = 'alphabet_calibri/g.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_g)
l_h = cv2.imread(filename = 'alphabet_calibri/h.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_h)
l_i = cv2.imread(filename = 'alphabet_calibri/i.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_i)
l_j = cv2.imread(filename = 'alphabet_calibri/j.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_j)
l_k = cv2.imread(filename = 'alphabet_calibri/k.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_k)
l_l = cv2.imread(filename = 'alphabet_calibri/l.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_l)
l_m = cv2.imread(filename = 'alphabet_calibri/m.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_m)
l_n = cv2.imread(filename = 'alphabet_calibri/n.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_n)
l_o = cv2.imread(filename = 'alphabet_calibri/o.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_o)
l_p = cv2.imread(filename = 'alphabet_calibri/p.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_p)
l_q = cv2.imread(filename = 'alphabet_calibri/q.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_q)
l_r = cv2.imread(filename = 'alphabet_calibri/r.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_r)
l_s = cv2.imread(filename = 'alphabet_calibri/s.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_s)
l_t = cv2.imread(filename = 'alphabet_calibri/t.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_t)
l_u = cv2.imread(filename = 'alphabet_calibri/u.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_u)
l_v = cv2.imread(filename = 'alphabet_calibri/v.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_v)
l_w = cv2.imread(filename = 'alphabet_calibri/w.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_w)
l_x = cv2.imread(filename = 'alphabet_calibri/x.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_x)
l_y = cv2.imread(filename = 'alphabet_calibri/y.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_y)
l_z = cv2.imread(filename = 'alphabet_calibri/z.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters.append(l_z)

l_a = cv2.imread(filename = 'alphabet_calibri_italic/a.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_a)
l_b = cv2.imread(filename = 'alphabet_calibri_italic/b.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_b)
l_c = cv2.imread(filename = 'alphabet_calibri_italic/c.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_c)
l_d = cv2.imread(filename = 'alphabet_calibri_italic/d.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_d)
l_e = cv2.imread(filename = 'alphabet_calibri_italic/e.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_e)
l_f = cv2.imread(filename = 'alphabet_calibri_italic/f.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_f)
l_g = cv2.imread(filename = 'alphabet_calibri_italic/g.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_g)
l_h = cv2.imread(filename = 'alphabet_calibri_italic/h.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_h)
l_i = cv2.imread(filename = 'alphabet_calibri_italic/i.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_i)
l_j = cv2.imread(filename = 'alphabet_calibri_italic/j.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_j)
l_k = cv2.imread(filename = 'alphabet_calibri_italic/k.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_k)
l_l = cv2.imread(filename = 'alphabet_calibri_italic/l.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_l)
l_m = cv2.imread(filename = 'alphabet_calibri_italic/m.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_m)
l_n = cv2.imread(filename = 'alphabet_calibri_italic/n.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_n)
l_o = cv2.imread(filename = 'alphabet_calibri_italic/o.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_o)
l_p = cv2.imread(filename = 'alphabet_calibri_italic/p.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_p)
l_q = cv2.imread(filename = 'alphabet_calibri_italic/q.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_q)
l_r = cv2.imread(filename = 'alphabet_calibri_italic/r.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_r)
l_s = cv2.imread(filename = 'alphabet_calibri_italic/s.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_s)
l_t = cv2.imread(filename = 'alphabet_calibri_italic/t.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_t)
l_u = cv2.imread(filename = 'alphabet_calibri_italic/u.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_u)
l_v = cv2.imread(filename = 'alphabet_calibri_italic/v.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_v)
l_w = cv2.imread(filename = 'alphabet_calibri_italic/w.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_w)
l_x = cv2.imread(filename = 'alphabet_calibri_italic/x.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_x)
l_y = cv2.imread(filename = 'alphabet_calibri_italic/y.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_y)
l_z = cv2.imread(filename = 'alphabet_calibri_italic/z.png', flags = cv2.IMREAD_GRAYSCALE)
Calibri_Letters_italic.append(l_z)

l_a = cv2.imread(filename = 'alphabet_comic_sans/a.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_a)
l_b = cv2.imread(filename = 'alphabet_comic_sans/b.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_b)
l_c = cv2.imread(filename = 'alphabet_comic_sans/c.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_c)
l_d = cv2.imread(filename = 'alphabet_comic_sans/d.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_d)
l_e = cv2.imread(filename = 'alphabet_comic_sans/e.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_e)
l_f = cv2.imread(filename = 'alphabet_comic_sans/f.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_f)
l_g = cv2.imread(filename = 'alphabet_comic_sans/g.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_g)
l_h = cv2.imread(filename = 'alphabet_comic_sans/h.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_h)
l_i = cv2.imread(filename = 'alphabet_comic_sans/i.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_i)
l_j = cv2.imread(filename = 'alphabet_comic_sans/j.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_j)
l_k = cv2.imread(filename = 'alphabet_comic_sans/k.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_k)
l_l = cv2.imread(filename = 'alphabet_comic_sans/l.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_l)
l_m = cv2.imread(filename = 'alphabet_comic_sans/m.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_m)
l_n = cv2.imread(filename = 'alphabet_comic_sans/n.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_n)
l_o = cv2.imread(filename = 'alphabet_comic_sans/o.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_o)
l_p = cv2.imread(filename = 'alphabet_comic_sans/p.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_p)
l_q = cv2.imread(filename = 'alphabet_comic_sans/q.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_q)
l_r = cv2.imread(filename = 'alphabet_comic_sans/r.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_r)
l_s = cv2.imread(filename = 'alphabet_comic_sans/s.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_s)
l_t = cv2.imread(filename = 'alphabet_comic_sans/t.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_t)
l_u = cv2.imread(filename = 'alphabet_comic_sans/u.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_u)
l_v = cv2.imread(filename = 'alphabet_comic_sans/v.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_v)
l_w = cv2.imread(filename = 'alphabet_comic_sans/w.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_w)
l_x = cv2.imread(filename = 'alphabet_comic_sans/x.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_x)
l_y = cv2.imread(filename = 'alphabet_comic_sans/y.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_y)
l_z = cv2.imread(filename = 'alphabet_comic_sans/z.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters.append(l_z)

l_a = cv2.imread(filename = 'alphabet_comic_sans_italic/a.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_a)
l_b = cv2.imread(filename = 'alphabet_comic_sans_italic/b.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_b)
l_c = cv2.imread(filename = 'alphabet_comic_sans_italic/c.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_c)
l_d = cv2.imread(filename = 'alphabet_comic_sans_italic/d.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_d)
l_e = cv2.imread(filename = 'alphabet_comic_sans_italic/e.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_e)
l_f = cv2.imread(filename = 'alphabet_comic_sans_italic/f.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_f)
l_g = cv2.imread(filename = 'alphabet_comic_sans_italic/g.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_g)
l_h = cv2.imread(filename = 'alphabet_comic_sans_italic/h.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_h)
l_i = cv2.imread(filename = 'alphabet_comic_sans_italic/i.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_i)
l_j = cv2.imread(filename = 'alphabet_comic_sans_italic/j.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_j)
l_k = cv2.imread(filename = 'alphabet_comic_sans_italic/k.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_k)
l_l = cv2.imread(filename = 'alphabet_comic_sans_italic/l.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_l)
l_m = cv2.imread(filename = 'alphabet_comic_sans_italic/m.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_m)
l_n = cv2.imread(filename = 'alphabet_comic_sans_italic/n.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_n)
l_o = cv2.imread(filename = 'alphabet_comic_sans_italic/o.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_o)
l_p = cv2.imread(filename = 'alphabet_comic_sans_italic/p.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_p)
l_q = cv2.imread(filename = 'alphabet_comic_sans_italic/q.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_q)
l_r = cv2.imread(filename = 'alphabet_comic_sans_italic/r.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_r)
l_s = cv2.imread(filename = 'alphabet_comic_sans_italic/s.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_s)
l_t = cv2.imread(filename = 'alphabet_comic_sans_italic/t.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_t)
l_u = cv2.imread(filename = 'alphabet_comic_sans_italic/u.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_u)
l_v = cv2.imread(filename = 'alphabet_comic_sans_italic/v.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_v)
l_w = cv2.imread(filename = 'alphabet_comic_sans_italic/w.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_w)
l_x = cv2.imread(filename = 'alphabet_comic_sans_italic/x.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_x)
l_y = cv2.imread(filename = 'alphabet_comic_sans_italic/y.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_y)
l_z = cv2.imread(filename = 'alphabet_comic_sans_italic/z.png', flags = cv2.IMREAD_GRAYSCALE)
Comic_S_Letters_italic.append(l_z)