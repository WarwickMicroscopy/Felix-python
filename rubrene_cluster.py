# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:06:00 2025

@author: rbean
"""

import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
# Lattice parameters of TIPS pentacene unit cell

ao, bo, co = 26.789, 7.17, 14.211
Vcell = ao * bo * co

# Reading atomic coordinates
xC, yC, zC = np.loadtxt('Rubrene_unit cell_carbon.txt', unpack=True)
nC = len(xC)
xH, yH, zH = np.loadtxt('Rubrene_unit cell_hydrogen.txt', unpack=True)
nH = len(xH)

# Fractional coordinates of atoms
xC, yC, zC = np.mod(xC, ao)/ao, np.mod(yC, bo)/bo, np.mod(zC, co)/co
xH, yH, zH = np.mod(xH, ao)/ao, np.mod(yH, bo)/bo, np.mod(zH, co)/co

# 200 kV parameters
lambda_ = 0.0251
mbymo = 1.3914

# Depth profile
deltaz = 0.4
max_z = 100
z = np.arange(deltaz, max_z + deltaz, deltaz)
nz = len(z)

# Atom scattering factors
Catom_a = np.array([0.212080767, 0.199811865, 0.168254385]) * mbymo
Catom_b = np.array([0.208605417, 0.208610186, 5.57870773])
Catom_c = np.array([0.14204836, 0.363830672, 8.35012044e-4]) * mbymo
Catom_d = np.array([1.33311887, 3.80800263, 4.0398262e-2])

Hatom_a = np.array([4.20298324e-3, 0.0627762505, 0.0300907347]) * mbymo
Hatom_b = np.array([0.225350888, 0.225366950, 0.225331756])
Hatom_c = np.array([0.0677756695, 3.56609237e-3, 0.0276135815]) * mbymo
Hatom_d = np.array([4.38854001, 0.403884823, 1.44490166])

# Mean inner potential correction
fo = nC * np.sum(Catom_c + (Catom_a / Catom_b))
fo += nH * np.sum(Hatom_c + (Hatom_a / Hatom_b))

xi0 = (np.pi * Vcell) / (lambda_ * fo)
K = np.sqrt((1 / lambda_)**2 + (1 / (lambda_ * xi0)))

# Generating clusters
basis1 = np.array([1, 0, 0])
basis2 = np.array([0, 1, 0])
Rinner = 8 / bo
Ninner = 0
Nouter = 0
inner_index = []
outer_index = []
reciprocal = []

for a in range(-32, 33):
    for b in range(-8, 9):
        hkl = a * basis1 + b * basis2
        gmag = np.linalg.norm(hkl / [ao, bo, co])
        if gmag <= Rinner and (abs(a) + abs(b)) != 0:
            inner_index.append([a, b])
            Ninner += 1
        outer_index.append([a, b])
        reciprocal.append(-gmag**2 / (2 * K))
        Nouter += 1

inner_index = np.array(inner_index)
outer_index = np.array(outer_index)
reciprocal = np.array(reciprocal)

# Structure Matrix
Astr = np.zeros(Ninner, dtype=np.complex128)
for idx, (a, b) in enumerate(inner_index):
    hkl = a * basis1 + b * basis2
    h, k, l = hkl / [ao, bo, co]

    # Structure factor
    Fg_C = np.sum(np.exp(-2j * np.pi * (h * xC + k * yC + l * zC)))
    Fg_H = np.sum(np.exp(-2j * np.pi * (h * xH + k * yH + l * zH)))
    s = np.linalg.norm([h, k, l])
    fs_C = np.sum(Catom_a / (s**2 + Catom_b) + Catom_c * np.exp(-Catom_d * s**2))
    fs_H = np.sum(Hatom_a / (s**2 + Hatom_b) + Hatom_c * np.exp(-Hatom_d * s**2))
    Fg = (Fg_C * fs_C) + (Fg_H * fs_H)
    # xi_g = (np.pi * K * Vcell) / Fg
    Astr[idx] = Fg / (2 * np.pi * K * Vcell)
t1 = time.time()-start
print(f"Structure matrix calculated ({t1})")
# Beam 'cluster neighbors' matrix
beamsNN = np.zeros((Nouter, Nouter), dtype=complex)
for a in range(Nouter):
    for b in range(Ninner):
        index_x = outer_index[a, 0] + inner_index[b, 0]
        index_y = outer_index[a, 1] + inner_index[b, 1]
        for c in range(Nouter):
            if (outer_index[c, 0] == index_x) and (outer_index[c, 1] == index_y):
                beamsNN[a, c] = np.conj(Astr[b])
beamsNN += np.diag(reciprocal)
t2 = time.time()-start
print(f"Cluster matrix calculated ({t2})")
# Solve for beam intensities
phi = np.zeros(Nouter, dtype=complex)
phi[0] = 1  # Reference beam (000)
beams = np.zeros((nz, Nouter), dtype=complex)

for depth in range(nz):
    phi += 2j * np.pi * deltaz * beamsNN @ phi
    beams[depth, :] = phi

# Compute beam intensities
Ig000 = np.abs(beams[:, 0])**2
Ig001 = np.abs(beams[:, 1])**2
Ig002 = np.abs(beams[:, 2])**2
Ig003 = np.abs(beams[:, 3])**2

# Plot results
plt.plot(z, Ig000, '-k')
plt.plot(z, Ig001)
plt.plot(z, Ig002)
plt.plot(z, Ig003)
plt.xlabel('Depth (Å)')
plt.ylabel('Intensity')
plt.title('Bragg Beam Intensities in Rubrene')
plt.show()
