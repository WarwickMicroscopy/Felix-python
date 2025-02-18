# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:26:54 2024

@author: Richard

Contains the subroutines needed to produce a LACBED simulation
Each of which call further pylix subroutines
Returns the simulated LACBED patterns

"""

import numpy as np
from scipy.constants import c, h, e, m_e, angstrom
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patheffects import withStroke
from matplotlib.ticker import PercentFormatter
import time
from pylix_modules import pylix as px
from pylix_modules import pylix_dicts as fu


def simulate(v):
    # some setup calculations
    # Electron velocity in metres per second
    electron_velocity = (c * np.sqrt(1.0 - ((m_e * c**2) /
                         (e * v.accelerating_voltage_kv*1000.0 +
                          m_e * c**2))**2))
    # Electron wavelength in Angstroms
    electron_wavelength = h / (
        np.sqrt(2.0 * m_e * e * v.accelerating_voltage_kv*1000.0) *
        np.sqrt(1.0 + (e * v.accelerating_voltage_kv*1000.0) /
                (2.0 * m_e * c**2))) / angstrom
    # Wavevector magnitude k
    electron_wave_vector_magnitude = 2.0 * np.pi / electron_wavelength
    # Relativistic correction
    relativistic_correction = 1.0 / np.sqrt(1.0 - (electron_velocity / c)**2)
    # Conversion from scattering factor to volts
    cell_volume = v.cell_a*v.cell_b*v.cell_c*np.sqrt(1.0-np.cos(v.cell_alpha)**2
                  - np.cos(v.cell_beta)**2 - np.cos(v.cell_gamma)**2
                  +2.0*np.cos(v.cell_alpha)*np.cos(v.cell_beta)*np.cos(v.cell_gamma))
    scatt_fac_to_volts = ((h**2) /
                          (2.0*np.pi * m_e * e * cell_volume * (angstrom**2)))

    # ===============================================
    # fill the unit cell and get mean inner potential
    # when iterating we only do it if necessary?
    # if v.iter_count == 0 or v.current_variable_type < 6:
    atom_position, atom_label, atom_name, B_iso, occupancy = \
        px.unique_atom_positions(
            v.symmetry_matrix, v.symmetry_vector, v.basis_atom_label,
            v.basis_atom_name,
            v.basis_atom_position, v.basis_B_iso, v.basis_occupancy)

    # Generate atomic numbers based on the elemental symbols
    atomic_number = np.array([fu.atomic_number_map[na] for na in atom_name])

    n_atoms = len(atom_label)
    if v.iter_count == 1:
        print("  There are "+str(n_atoms)+" atoms in the unit cell")
    # plot
    if v.iter_count == 0 and v.plot:
        atom_cvals = mcolors.Normalize(vmin=1, vmax=103)
        atom_cmap = plt.cm.viridis
        atom_colours = atom_cmap(atom_cvals(atomic_number))
        border_cvals = mcolors.Normalize(vmin=0, vmax=1)
        border_cmap = plt.cm.plasma
        border_colours = border_cmap(border_cvals(atom_position[:, 2]))
        bb = 5
        fig, ax = plt.subplots(figsize=(bb, bb))
        plt.scatter(atom_position[:, 0], atom_position[:, 1],
                    color=atom_colours, edgecolor=border_colours,
                    linewidth=1, s=100)
        plt.xlim(left=0.0, right=1.0)
        plt.ylim(bottom=0.0, top=1.0)
        ax.set_axis_off
        plt.grid(True)
    if v.debug:
        print("Symmetry operations:")
        for i in range(len(v.symmetry_matrix)):
            print(f"{i+1}: {v.symmetry_matrix[i]}, {v.symmetry_vector[i]}")
        print("atomic coordinates")
        for i in range(n_atoms):
            print(f"{atom_label[i]} {atom_name[i]}: {atom_position[i]}")
    # mean inner potential as the sum of scattering factors at g=0
    # multiplied by h^2/(2pi*m0*e*CellVolume)
    mip = 0.0
    for i in range(n_atoms):  # get the scattering factor
        if v.scatter_factor_method == 0:
            mip += px.f_kirkland(atomic_number[i], 0.0)
        elif v.scatter_factor_method == 1:
            mip += px.f_lobato(atomic_number[i], 0.0)
        elif v.scatter_factor_method == 2:
            mip += px.f_peng(atomic_number[i], 0.0)
        elif v.scatter_factor_method == 3:
            mip += px.f_doyle_turner(atomic_number[i], 0.0)
        else:
            raise ValueError("No scattering factors chosen in felix.inp")
    mip = mip.item()*scatt_fac_to_volts  # NB convert array to float
    if v.iter_count == 1:
        print(f"  Mean inner potential = {mip:.1f} Volts")
    # Wave vector magnitude in crystal
    # high-energy approximation (not HOLZ compatible)
    # K^2=k^2+U0
    big_k_mag = electron_wave_vector_magnitude
    # big_k_mag = np.sqrt(electron_wave_vector_magnitude**2+mip)
    # k-vector for the incident beam (k is along z in the microscope frame)
    big_k = np.array([0.0, 0.0, big_k_mag])

    # ===============================================
    # put the crystal in the micrcoscope reference frame, in Å
    atom_coordinate = (atom_position[:, 0, np.newaxis] * a_vec_m +
                       atom_position[:, 1, np.newaxis] * b_vec_m +
                       atom_position[:, 2, np.newaxis] * c_vec_m)

    # ===============================================


    return


