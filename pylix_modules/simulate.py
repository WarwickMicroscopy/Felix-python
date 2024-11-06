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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from pylix_modules import pylix as px
from pylix_modules import pylix_dicts as fu

def simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
             symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
             cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
             basis_B_iso, basis_occupancy, scatter_factor_method,
             accelerating_voltage_kv, x_direction, incident_beam_direction,
             normal_direction, min_reflection_pool, min_strong_beams, g_limit,
             input_hkls, absorption_method, absorption_per, convergence_angle,
             image_radius, thickness):
    
    # %% some setup calculations
    # Electron velocity in metres per second
    electron_velocity = (c * np.sqrt(1.0 - ((m_e * c**2) /
                         (e * accelerating_voltage_kv*1000.0 + m_e * c**2))**2))
    # Electron wavelength in Angstroms
    electron_wavelength = h / (
        np.sqrt(2.0 * m_e * e * accelerating_voltage_kv*1000.0) *
        np.sqrt(1.0 + (e * accelerating_voltage_kv*1000.0) /
                (2.0 * m_e * c**2))) / angstrom
    # Wavevector magnitude k
    electron_wave_vector_magnitude = 2.0 * np.pi / electron_wavelength
    # Relativistic correction
    relativistic_correction = 1.0 / np.sqrt(1.0 - (electron_velocity / c)**2)
    # Relativistic mass
    relativistic_mass = relativistic_correction * m_e
    # Conversion from scattering factor to volts
    cell_volume = cell_a*cell_b*cell_c*np.sqrt(1.0-np.cos(cell_alpha)**2
                  - np.cos(cell_beta)**2 - np.cos(cell_gamma)**2
                  +2.0*np.cos(cell_alpha)*np.cos(cell_beta)*np.cos(cell_gamma))
    scatt_fac_to_volts = ((h**2) /
                          (2.0*np.pi * m_e * e * cell_volume * (angstrom**2)))
    n_thickness = len(thickness)

    # %% fill the unit cell and get mean inner potential
    atom_position, atom_label, atom_name, B_iso, occupancy = \
        px.unique_atom_positions(
            symmetry_matrix, symmetry_vector, basis_atom_label, basis_atom_name,
            basis_atom_position, basis_B_iso, basis_occupancy)
    
    # Generate atomic numbers based on the elemental symbols
    atomic_number = np.array([fu.atomic_number_map[name] for name in atom_name])
    
    n_atoms = len(atom_label)
    print("There are "+str(n_atoms)+" atoms in the unit cell")
    # plot
    if plot:
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
    if debug:
        print("Symmetry operations:")
        for i in range(len(symmetry_matrix)):
            print(f"{i+1}: {symmetry_matrix[i]}, {symmetry_vector[i]}")
        print("atomic coordinates")
        for i in range(n_atoms):
            print(f"{atom_label[i]} {atom_name[i]}: {atom_position[i]}")
    # mean inner potential as the sum of scattering factors at g=0 multiplied by
    # h^2/(2pi*m0*e*CellVolume)
    mip = 0.0
    for i in range(n_atoms):  # get the scattering factor
        if scatter_factor_method == 0:
            mip += px.f_kirkland(atomic_number[i], 0.0)
        elif scatter_factor_method == 1:
            mip += px.f_lobato(atomic_number[i], 0.0)
        elif scatter_factor_method == 2:
            mip += px.f_peng(atomic_number[i], 0.0)
        elif scatter_factor_method == 3:
            mip += px.f_doyle_turner(atomic_number[i], 0.0)
        else:
            error_flag = True
            raise ValueError("No scattering factors chosen in felix.inp")
    mip = mip.item()*scatt_fac_to_volts  # comes back as an array, convert to float
    print(f"Mean inner potential = {mip:.1f} Volts")
    # Wave vector magnitude in crystal
    # high-energy approximation (not HOLZ compatible)
    # K^2=k^2+U0
    big_k_mag = np.sqrt(electron_wave_vector_magnitude**2+mip)
    # k-vector for the incident beam (k is along z in the microscope frame)
    big_k = np.array([0.0, 0.0, big_k_mag])
    
    
    # %% set up reference frames
    a_vec_m, b_vec_m, c_vec_m, ar_vec_m, br_vec_m, cr_vec_m, norm_dir_m = \
        px.reference_frames(cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                            cell_gamma, space_group, x_direction,
                            incident_beam_direction, normal_direction)
    
    # put the crystal in the micrcoscope reference frame, in Å
    atom_coordinate = (atom_position[:, 0, np.newaxis] * a_vec_m +
                       atom_position[:, 1, np.newaxis] * b_vec_m +
                       atom_position[:, 2, np.newaxis] * c_vec_m)
    
    # %% set up beam pool
    strt = time.time()
    # NB g_pool are in reciprocal Angstroms in the microscope reference frame
    hkl, g_pool, g_pool_mag, g_output = px.hkl_make(ar_vec_m, br_vec_m, cr_vec_m,
                                                    big_k, lattice_type,
                                                    min_reflection_pool,
                                                    min_strong_beams, g_limit,
                                                    input_hkls,
                                                    electron_wave_vector_magnitude)
    n_hkl = len(g_pool)
    n_out = len(g_output)  # redefined to match things we can actually output
    # NEEDS SOME MORE WORK TO MATCH SIM/EXPT PATTERNS if this happens
    
    # outputs
    print(f"Beam pool: {n_hkl} reflexions ({min_strong_beams} strong beams)")
    # we will have larger g-vectors in g_matrix since this has differences g - h
    # but the maximum of the g pool is probably a more useful thing to know
    print(f"Maximum |g| = {np.max(g_pool_mag)/(2*np.pi):.3f} 1/Å")
    
    # plot
    if plot:
        xm = np.ceil(np.max(g_pool_mag/(2*np.pi)))
        fig, ax = plt.subplots(1, 1)
        w_f = 10
        fig.set_size_inches(w_f, w_f)
        ax.set_facecolor('black')
        # colour according to Laue zone
        lz_cvals = mcolors.Normalize(vmin=np.min(g_pool[:, 2]),
                                     vmax=np.max(g_pool[:, 2]))
        lz_cmap = plt.cm.brg
        lz_colours = lz_cmap(lz_cvals(g_pool[:, 2]))
        # plots the g-vectors in the pool, colours for different Laue zones
        plt.scatter(g_pool[:, 0]/(2*np.pi), g_pool[:, 1]/(2*np.pi),
                    s=20, color=lz_colours)
        # title
        plt.annotate("Beam pool", xy=(5, 5), color='white',
                     xycoords='axes pixels', size=24)
        # major grid at 1 1/Å
        plt.grid(True,  which='major', color='lightgrey',
                 linestyle='-', linewidth=1.0)
        plt.gca().set_xticks(np.arange(-xm, xm, 1))
        plt.gca().set_yticks(np.arange(-xm, xm, 1))
        plt.grid(True, which='minor', color='grey', linestyle='--',
                 linewidth=0.5)
        # minor grid at 0.2 1/Å
        plt.gca().set_xticks(np.arange(-xm, xm, 0.2), minor=True)
        plt.gca().set_yticks(np.arange(-xm, xm, 0.2), minor=True)
        # remove axis labels
        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        left=False, right=False,
                        labelbottom=False, labelleft=False)
        plt.show()
    
    # g-vector matrix
    g_matrix = np.zeros((n_hkl, n_hkl, 3))
    g_matrix = g_pool[:, np.newaxis, :] - g_pool[np.newaxis, :, :]
    # g-vector magnitudes
    g_magnitude = np.sqrt(np.sum(g_matrix**2, axis=2))
    
    # Conversion factor from F_g to U_g
    Fg_to_Ug = relativistic_correction / (np.pi * cell_volume)
    
    # now make the Ug matrix, i.e. calculate the structure factor Fg for all
    # g-vectors in g_matrix and convert using the above factor
    ug_matrix = Fg_to_Ug * px.Fg_matrix(n_hkl, scatter_factor_method, n_atoms,
                                        atom_coordinate, atomic_number, occupancy,
                                        B_iso, g_matrix, g_magnitude,
                                        absorption_method, absorption_per,
                                        electron_velocity)
    # ug_matrix = 10 ug_matrix
    # matrix of dot products with the surface normal
    g_dot_norm = np.dot(g_pool, norm_dir_m)
    
    print("Ug matrix constructed")
    if debug:
        np.set_printoptions(precision=3, suppress=True)
        print(100*ug_matrix[:5, :5])
    
    
    # %% deviation parameter for each pixel and g-vector
    # s_g [n_hkl, image diameter, image diameter]
    # and k vector for each pixel, tilted_k [image diameter, image diameter, 3]
    s_g, tilted_k = px.deviation_parameter(convergence_angle, image_radius,
                                           big_k_mag, g_pool, g_pool_mag)
    
    # %% Bloch wave calculation
    mid = time.time()
    # Dot product of k with surface normal, size [image diameter, image diameter]
    k_dot_n = np.tensordot(tilted_k, norm_dir_m, axes=([2], [0]))
    
    lacbed_sim = np.zeros([n_thickness, 2*image_radius, 2*image_radius, len(g_output)],
                          dtype=float)
    
    print("Bloch wave calculation...", end=' ')
    if debug:
        print("")
        print("output indices")
        print(g_output[:15])
    # pixel by pixel calculations from here
    for pix_x in range(2*image_radius):
        # progess
        print(f"\rBloch wave calculation... {50*pix_x/image_radius:.0f}%", end="")
    
        for pix_y in range(2*image_radius):
            s_g_pix = np.squeeze(s_g[pix_x, pix_y, :])
            k_dot_n_pix = k_dot_n[pix_x, pix_y]
    
            # works for multiple thicknesses
            wave_functions = px.wave_functions(
                g_output, s_g_pix, ug_matrix, min_strong_beams, n_hkl, big_k_mag,
                g_dot_norm, k_dot_n_pix, thickness, debug)
    
            intensity = np.abs(wave_functions)**2
    
            # Map diffracted intensity to required output g vectors
            # note x and y swapped!
            lacbed_sim[:, -pix_y, pix_x, :] = intensity[:, :len(g_output)]
    
    print("\rBloch wave calculation... done    ")

    # timings
    setup = mid-strt
    bwc = time.time()-mid

    # %% return
    return hkl, g_output, lacbed_sim, setup, bwc