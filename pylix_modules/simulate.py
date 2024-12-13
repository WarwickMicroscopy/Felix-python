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
    # set up reference frames
    a_vec_m, b_vec_m, c_vec_m, ar_vec_m, br_vec_m, cr_vec_m, norm_dir_m = \
        px.reference_frames(v.cell_a, v.cell_b, v.cell_c, v.cell_alpha,
                            v.cell_beta,
                            v.cell_gamma, v.space_group, v.x_direction,
                            v.incident_beam_direction, v.normal_direction)

    # put the crystal in the micrcoscope reference frame, in Å
    atom_coordinate = (atom_position[:, 0, np.newaxis] * a_vec_m +
                       atom_position[:, 1, np.newaxis] * b_vec_m +
                       atom_position[:, 2, np.newaxis] * c_vec_m)

    # ===============================================
    # set up beam pool
    # currently we do this every iteration, but could be restricted to cases
    # where we need to do it in iterations
    strt = time.time()
    # NB g_pool are in reciprocal Angstroms in the microscope reference frame
    v.hkl, g_pool, g_pool_mag, v.g_output = \
        px.hkl_make(ar_vec_m, br_vec_m, cr_vec_m,
                    big_k, v.lattice_type, v.min_reflection_pool,
                    v.min_strong_beams, v.g_limit, v.input_hkls, big_k_mag)
    n_hkl = len(g_pool)
    n_out = len(v.g_output)  # redefined to match what we can actually output
    # NEEDS SOME MORE WORK TO MATCH SIM/EXPT PATTERNS if this happens

    # outputs
    if v.iter_count == 1:
        print(f"  Beam pool: {n_hkl} reflexions ({v.min_strong_beams} strong beams)")
        # we will have larger g-vectors in g_matrix since this has
        # differences g - h
        # but the maximum of the g pool is probably a more useful thing to know
        print(f"  Maximum |g| = {np.max(g_pool_mag)/(2*np.pi):.3f} 1/Å")

    # plot
    if v.iter_count == 1 and v.plot:
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
    ug_matrix = Fg_to_Ug * px.Fg_matrix(n_hkl, v.scatter_factor_method,
                                        n_atoms, atom_coordinate,
                                        atomic_number, occupancy,
                                        B_iso, g_matrix, g_magnitude,
                                        v.absorption_method, v.absorption_per,
                                        electron_velocity)
    # ug_matrix = 10 ug_matrix
    # matrix of dot products with the surface normal
    g_dot_norm = np.dot(g_pool, norm_dir_m)
    if v.iter_count == 1:
        print("    Ug matrix constructed")
    if v.debug:
        np.set_printoptions(precision=3, suppress=True)
        print(100*ug_matrix[:5, :5])

    # ===============================================
    # deviation parameter for each pixel and g-vector
    # s_g [n_hkl, image diameter, image diameter]
    # and k vector for each pixel, tilted_k [image diameter, image diameter, 3]
    s_g, tilted_k = px.deviation_parameter(v.convergence_angle, v.image_radius,
                                           big_k_mag, g_pool, g_pool_mag)

    # ===============================================
    # Bloch wave calculation
    mid = time.time()
    # Dot product of k with surface normal, [image diameter, image diameter]
    k_dot_n = np.tensordot(tilted_k, norm_dir_m, axes=([2], [0]))

    v.lacbed_sim = np.zeros([v.n_thickness, 2*v.image_radius, 2*v.image_radius,
                             len(v.g_output)], dtype=float)

    print("Bloch wave calculation...", end=' ')
    if v.debug:
        print("")
        print("output indices")
        print(v.g_output[:15])
    # pixel by pixel calculations from here
    for pix_x in range(2*v.image_radius):
        # progess
        print(f"\rBloch wave calculation... {50*pix_x/v.image_radius:.0f}%", end="")

        for pix_y in range(2*v.image_radius):
            s_g_pix = np.squeeze(s_g[pix_x, pix_y, :])
            k_dot_n_pix = k_dot_n[pix_x, pix_y]

            # works for multiple thicknesses
            wave_functions = px.wave_functions(
                v.g_output, s_g_pix, ug_matrix, v.min_strong_beams, n_hkl,
                big_k_mag, g_dot_norm, k_dot_n_pix, v.thickness, v.debug)

            intensity = np.abs(wave_functions)**2

            # Map diffracted intensity to required output g vectors
            # note x and y swapped!
            v.lacbed_sim[:, -pix_y, pix_x, :] = intensity[:, :len(v.g_output)]

    print("\rBloch wave calculation... done    ")

    # timings
    setup = mid-strt
    bwc = time.time()-mid

    # increment iteration counter
    v.iter_count += 1

    return setup, bwc


def zncc(img1, img2):
    """ input: img1, img2 sets of n images, both of size [pix_x, pix_y, n]
    output is a numpy array of length n, giving zncc for each pair of images
    zncc is -1 = perfect anticorrelation, +1 = perfect correlation
    """
    n_pix = img1.shape[0]**2
    img1_normalised = (img1 - np.mean(img1, axis=(0, 1), keepdims=True)) / (
        np.std(img1, axis=(0, 1), keepdims=True))
    img2_normalised = (img2 - np.mean(img2, axis=(0, 1), keepdims=True)) / (
        np.std(img2, axis=(0, 1), keepdims=True))

    # zero-mean normalised 2D cross-correlation
    cc = np.sum(img1_normalised * img2_normalised, axis=(0, 1))/n_pix
    return cc


def figure_of_merit(v):
    """ needs fleshing out with image processing & correlation options
    takes as an input v.lacbed_sim, shape [v.n_thickness, pix_x, pix_y, n_out]
    applies image processing if required
    image processing = 0 -> no Gaussian blur (applied with radius 0)
    image processing = 1 -> Gaussian blur radius defined in felix.inp
    image processing = 2 -> find the best blur radius
    uses zncc, which works on sets of images both of size [pix_x, pix_y, n]
    currently returns a single figure of merit fom that is mean of zncc's for
    the best thickness.  Could give a more sophisticated analysis..
    """
    # figure of merit - might need a NaN check? size [n_thick, n_out]
    n_out = v.lacbed_expt.shape[2]  # ***is this in v? get rid if so
    fom_array = np.ones([v.n_thickness, n_out])
    # set up plot for blur optimisation
    if v.plot and v.image_processing == 2:
        fig, ax = plt.subplots(1, 1)
        w_f = 10
        fig.set_size_inches(w_f, w_f)
        ax.set_xlabel('Blur radius', size=24)
        ax.set_ylabel('Figure of merit', size=24)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
    # loop over thicknesses
    for i in range(v.n_thickness):
        # image processing = 2 -> find the best blur
        if v.image_processing == 2:
            radii = np.arange(0.2, 2.1, 0.1)  # range of blurs to try
            b_fom = ([])  # mean fom for each blur
            for r in radii:
                blacbed = np.copy(v.lacbed_sim[i, :, :, :])
                for j in range(n_out):
                    blacbed[:, :, j] = gaussian_filter(blacbed[:, :, j],
                                                       sigma=r)
                b_fom.append(np.mean(1.0 - zncc(v.lacbed_expt, blacbed)))
            if v.plot:
                plt.plot(radii, b_fom)
            v.blur_radius = radii[np.argmin(b_fom)]
        if v.image_processing != 0:
            for j in range(n_out):
                v.lacbed_sim[i, :, :, j] = gaussian_filter(v.lacbed_sim[i, :, :, j],
                                                           sigma=v.blur_radius)
        fom_array[i, :] = 1.0 - zncc(v.lacbed_expt, v.lacbed_sim[i, :, :, :])
    if v.plot and v.image_processing == 2:
        plt.show()
    # print best values
    if v.image_processing != 0:
        print(f"  Best blur={v.blur_radius:.1f}")
    if v.n_thickness > 1:
        v.best_t = np.argmin(np.mean(fom_array, axis=1))
        print(f"  Best thickness {0.1*v.thickness[v.best_t]:.1f} nm")
        # mean figure of merit
        fom = np.mean(fom_array[v.best_t])
    else:
        v.best_t = 0
        fom = np.mean(fom_array[0])

    # plot
    if v.plot and v.n_thickness > 1:
        fig, ax = plt.subplots(1, 1)
        w_f = 10
        fig.set_size_inches(w_f, w_f)
        plt.plot(v.thickness/10, np.mean(fom_array, axis=1), 'ro', linewidth=2)
        for i in range(n_out):
            annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
            plt.plot(v.thickness/10, fom_array[:, i], label=annotation)
        ax.set_xlabel('Thickness (nm)', size=24)
        ax.set_ylabel('Figure of merit', size=24)
        ax.legend()
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.show()
    return fom


def update_variables(v):
    """
    Updates the different refinement variables
    current_var is an array of variable values
    refined_variable_type is a matching array saying what type they are
    atom_refine_flag is a matching array giving the index of the atom
    in the basis that is being refined. (-1 = not an atomic refinement)
    basis_atom_position is the position of an atom (in A, microscope frame???)
    basis_atom_delta is the uncertainty in position of an atom, forgotten
    how this works
    """

    # will tackle this when doing atomic position refinement
    # basis_atom_delta.fill(0)  # Reset atom coordinate uncertainties to zero

    for i in range(v.n_variables):
        # Check the type of variable, last digit of v.refined_variable_type
        variable_type = v.refined_variable_type[i] % 10

        if variable_type == 0:
            # Structure factor refinement (handled elsewhere)
            variable_check = 1

        # elif variable_type == 2:  # NEEDS WORK
            # Atomic coordinates
            # atom_id = atom_refine_flag[j]

            # # Update position: r' = r - v*(r.v) + v*current_var
            # dot_product = np.dot(basis_atom_position[atom_id, :], vector[j - 1, :])
            # basis_atom_position[atom_id, :] = np.mod(
            #     basis_atom_position[atom_id, :] - vector[j - 1, :] * dot_product + 
            #     vector[jnd - 1, :] * current_var[i], 1
            # )

            # # Update uncertainty if iependent_delta is non-zero
            # if abs(independent_delta[i]) > 1e-10:  # Tiny threshold
            #     basis_atom_delta[atom_id, :] += vector[j - 1, :] * independent_delta[i]
            # j += 1

        elif variable_type == 3:
            # Occupancy
            v.basis_occupancy[v.atom_refine_flag[i]] = v.refined_variable[i]*1.0

        elif variable_type == 4:
            # Iso Debye-Waller factor
            v.basis_B_iso[v.atom_refine_flag[i]] = v.refined_variable[i]*1.0

        elif variable_type == 5:
            # Aniso Debye-Waller factor (not implemented)
            raise NotImplementedError("Anisotropic DWF not implemented")

        elif variable_type == 6:
            # Lattice parameters a, b, c
            if v.refined_variable_type[i] == 6:
                v.cell_a = v.cell_b = v.cell_c = v.refined_variable[i]*1.0
            elif v.refined_variable_type[i] == 16:
                v.cell_b = v.refined_variable[i]*1.0
            elif v.refined_variable_type[i] == 26:
                v.cell_c = v.refined_variable[i]*1.0

        # elif variable_type == 7:
        #     # Lattice angles alpha, beta, gamma
        #     variable_check[6] = 1
        #     if j == 1:
        #         v.cell_alpha = current_var[i]
        #     elif j == 2:
        #         v.cell_beta = current_var[i]
        #     elif j == 3:
        #         v.cell_gamma = current_var[i]

        elif variable_type == 8:
            # Convergence angle
            v.convergence_angle = v.refined_variable[i]*1.0

        elif variable_type == 9:
            # Accelerating voltage
            v.accelerating_voltage_kv = v.refined_variable[i]*1.0

    return


def print_LACBED(v):
    w = int(np.ceil(np.sqrt(v.n_out)))
    h = int(np.ceil(v.n_out/w))
    # only print all thicknesses for the first simulation
    if v.iter_count == 0:
        for j in range(v.n_thickness):
            fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
            text_effect = withStroke(linewidth=3, foreground='black')
            axes = axes.flatten()
            for i in range(v.n_out):
                axes[i].imshow(v.lacbed_sim[j, :, :, i], cmap='pink')
                axes[i].axis('off')
                annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
                axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                                 size=30, color='w', path_effects=[text_effect])
            for i in range(v.n_out, len(axes)):
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()
    else:
        j = v.best_t
        fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
        text_effect = withStroke(linewidth=3, foreground='black')
        axes = axes.flatten()
        for i in range(v.n_out):
            axes[i].imshow(v.lacbed_sim[j, :, :, i], cmap='pink')
            axes[i].axis('off')
            annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
            axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                             size=30, color='w', path_effects=[text_effect])
        for i in range(v.n_out, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()


def print_current_var(v, var):
    # prints the variable being refined
    if v.current_variable_type % 10 == 1:
        print(f"Current Ug {var:.3f}")
    elif v.current_variable_type % 10 == 2:
        print(f"Current atomic coordinate {var:.4f}")
    elif v.current_variable_type % 10 == 3:
        print(f"Current occupancy {var:.2f}")
    elif v.current_variable_type % 10 == 4:
        print(f"Current isotropic Debye-Waller factor {var:.2f}")
    elif v.current_variable_type % 10 == 5:
        print(f"Current anisotropic Debye-Waller factor {var:.2f}")
    elif v.current_variable_type % 10 == 6:
        print(f"Current lattice parameter {var:.4f}")
    elif v.current_variable_type % 10 == 8:
        print(f"Current convergence angle {var:.3f} Å^-1")
    elif v.current_variable_type % 10 == 9:
        print(f"Current accelerating voltage {var:.1f} kV")
