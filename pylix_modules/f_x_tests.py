# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:25:51 2026

@author: rbean
"""

import numpy as np
import matplotlib.pyplot as plt


def boron(s):
    f = (2.0545*np.exp(-23.2185*s**2) +
         1.3326*np.exp(-1.021*s**2) +
         1.0979*np.exp(-60.3498*s**2) +
         0.7068*np.exp(-0.1403*s**2) +
         -0.1932)
    return f


def carbon(s):
    f = (2.31*np.exp(-20.8439*s**2) +
         1.02*np.exp(-10.2075*s**2) +
         1.5886*np.exp(-0.5687*s**2) +
         0.865*np.exp(-51.6512*s**2) +
         0.2156)
    return f


def ge(s):
    f = (16.0816*np.exp(-2.8509*s**2) +
         6.3747*np.exp(-0.2516*s**2) +
         3.7068*np.exp(-11.4468*s**2) +
         3.683*np.exp(-54.7625*s**2) +
         2.1313)
    return f


g = np.linspace(0.05, 6.0, 1000)
s0 = g/2  # sin theta / lambda
f_x0 = carbon(s0)
mott = 0.02393366096322682
mask = (g != 0)
f_e0 = np.empty_like(g)
f_e0[mask] = mott*(6-f_x0[mask])/(s0[mask]**2)
# plt.plot(g, f_e0)

# from felix
g_magnitude=bloch.uniq_gmag
tol = 1e-6  # tolerance for considering g's equal
g_magnitude = np.round(g_magnitude / tol) * tol
# get unique g's and + mapping back
g_pool_mag, inverse = np.unique(g_magnitude, return_inverse=True)

s = g_pool_mag / (4*np.pi)

# array in real space to calculate integral
# may need more points at small r?
r = np.linspace(1e-6, xtal.r_max, xtal.n_points)

# integrate to get the structure factor
# NB should we use sinc(2qr) or sinc(2qr/pi) ???
qr = 2*np.outer(s, r)
base_core = 4.0 * np.pi * basis.core[i] * np.sinc(qr) * r**2
base_val = 4.0 * np.pi * basis.valence[i] * np.sinc(qr) * r**2
f_x_core = basis.pc[i] * np.trapz(base_core, r, axis=1)
f_x_valence = basis.pv[i] * np.trapz(base_val, r, axis=1)
f_x = f_x_core + f_x_valence

mask = (s != 0)
f_e = np.empty_like(s)
f_e[mask] = mott*(6-f_x[mask])/(s[mask]**2)

fig, ax = plt.subplots(1, 1)
w_f = 10
fig.set_size_inches(w_f, w_f)
plt.plot(s0, f_e0, linestyle='-', label='Tables')
plt.plot(s, f_e, linestyle='-', label='Felix')
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.legend(loc='best', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid()
plt.show()
