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
from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
from scipy.ndimage import shift
from scipy.fft import fftn, ifftn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patheffects import withStroke
from matplotlib.ticker import PercentFormatter
import time
import os
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
    # Cell volume
    cell_volume = (v.cell_a*v.cell_b*v.cell_c
                   * np.sqrt(1.-np.cos(v.cell_alpha)**2
                             - np.cos(v.cell_beta)**2
                             - np.cos(v.cell_gamma)**2
                             + 2.0*np.cos(v.cell_alpha)
                             * np.cos(v.cell_beta) * np.cos(v.cell_gamma)))
    # Conversion from scattering factor to volts
    scatt_fac_to_volts = ((h**2) /
                          (2.0*np.pi * m_e * e * cell_volume * (angstrom**2)))

    # ===============================================
    # added unique APD tensors u_ij
    # fill the unit cell and get mean inner potential
    # when iterating we only do it if necessary?
    # if v.iter_count == 0 or v.current_variable_type < 6:
    # print (v.atom_site_type_symbol)

    atom_position, atom_label, atom_type, atom_name, u_ij, occupancy, \
        pv, kappas = \
        px.unique_atom_positions(
            v.symmetry_matrix, v.symmetry_vector, v.basis_atom_label,
            v.atom_site_type_symbol, v.basis_atom_name, v.basis_atom_position,
            v.basis_u_ij, v.basis_occupancy, v.basis_pv,
            v.basis_kappa, v.debug)

    # Generate atomic numbers based on the elemental symbols
    atomic_number = np.array([fu.atomic_number_map[na] for na in atom_name])
    atomic_number_basis = np.array([fu.atomic_number_map[s]
                                    for s in v.basis_atom_name])

    if v.scatter_factor_method == 4:
        print("Precomputing atom core and valence densities")
        for i in range(len(atomic_number_basis)):
            px.precompute_densities(atomic_number_basis[i],
                                    v.basis_kappa[i], v.basis_pv[i])

        print(v.basis_kappa)
        print(v.basis_pv)

    n_atoms = len(atom_label)
    if v.iter_count == 0:
        print("  There are "+str(n_atoms)+" atoms in the unit cell")

    # output for debugging
    if v.debug:
        print("Symmetry operations:")
        for i in range(len(v.symmetry_matrix)):
            print(f"{i+1}: {v.symmetry_matrix[i]}, {v.symmetry_vector[i]}")
        np.set_printoptions(precision=5, suppress=True)
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
        elif v.scatter_factor_method == 4:
            mip += px.f_kirkland(atomic_number[i], 0.0)    # because we use kirkland for S<0.5 we can just set it here as well
        else:
            raise ValueError("No scattering factors chosen in felix.inp")
    mip = mip.item()*scatt_fac_to_volts  # NB convert array to float
    if v.iter_count == 0:
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
    a_vec_m, b_vec_m, c_vec_m, ar_vec_m, br_vec_m, cr_vec_m, norm_dir_m, t_mat_o2m, t_mat_c2o = \
        px.reference_frames(v.cell_a, v.cell_b, v.cell_c, v.cell_alpha, 
                            v.cell_beta, v.cell_gamma, v.space_group,
                            v.x_direction, v.incident_beam_direction,
                            v.normal_direction, v.debug)
    # put the crystal in the micrcoscope reference frame, in Å
    atom_coordinate = (atom_position[:, 0, np.newaxis] * a_vec_m +
                       atom_position[:, 1, np.newaxis] * b_vec_m +
                       atom_position[:, 2, np.newaxis] * c_vec_m)

    # plot unit cell and save .xyz file
    if v.iter_count == 0 and v.plot >= 1:
        atom_cvals = mcolors.Normalize(vmin=1, vmax=103)
        atom_cmap = plt.cm.prism
        atom_colours = atom_cmap(atom_cvals(atomic_number))
        border_cvals = mcolors.Normalize(vmin=0, vmax=1)
        border_cmap = plt.cm.plasma
        border_colours = border_cmap(border_cvals(atom_position[:, 2]))
        bb = 5
        fig, ax = plt.subplots(figsize=(bb, bb))
        plt.scatter(atom_coordinate[:, 0], atom_coordinate[:, 1],
                    color=atom_colours, edgecolor=border_colours,
                    linewidth=1, s=100)
        # plt.xlim(left=0.0, right=1.0)
        # plt.ylim(bottom=0.0, top=1.0)
        ax.set_axis_off
        ax.set_aspect('equal')
        plt.grid(True)

        # # xyz file
        # text = "\n"
        # for i in range(n_atoms):
        #     sas = str(atom_coordinate[i])
        #     xyz = sas[1:len(sas)-1]
        #     text = text + atom_name[i] + "  " + xyz + "\n"
        # fnam = v.chemical_formula_sum+".xyz"
        # f = open(fnam, "x")
        # f.write(str(n_atoms)+"\n")
        # f.write(text)
        # f.close()

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
    # n_out = len(v.g_output)  # redefined to match what we can actually output
    # NEEDS SOME MORE WORK TO MATCH SIM/EXPT PATTERNS if this happens

    # outputs
    if v.iter_count == 0:
        print(f"  Beam pool: {n_hkl} reflexions ({v.min_strong_beams} strong beams)")
        # we will have larger g-vectors in g_matrix since this has
        # differences g - h
        # but the maximum of the g pool is probably a more useful thing to know
        print(f"  Maximum |g| = {np.max(g_pool_mag)/(2*np.pi):.3f} 1/Å")
        # for i in range(n_hkl):
        #     print(f"{i},  {v.hkl[i]}")

    # plot beam pool
    if v.iter_count == 0 and v.plot >= 1:
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

    # g-vector matrix, array [n_hkl, n_hkl, 3]
    g_matrix = np.zeros((n_hkl, n_hkl, 3))
    g_matrix = g_pool[:, np.newaxis, :] - g_pool[np.newaxis, :, :]

    # Conversion factor from F_g to U_g
    Fg_to_Ug = relativistic_correction / (np.pi * cell_volume)

    # now make the Ug matrix, i.e. calculate the structure factor Fg for all
    # g-vectors in g_matrix and convert using the above factor
    ug_matrix = Fg_to_Ug * \
        px.Fg_matrix(n_hkl, v.scatter_factor_method, v.basis_atom_label,
                     atom_label, atom_coordinate, atomic_number, occupancy,
                     u_ij, g_matrix, v.absorption_method, v.absorption_per,
                     electron_velocity, kappas, pv, v.Debye_model,
                     v.model_flag, v.debug)
    # matrix of dot products with the surface normal
    g_dot_norm = np.dot(g_pool, norm_dir_m)
    if v.iter_count == 0:
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

    # timings
    setup = mid-strt
    bwc = time.time()-mid
    print(f"\rBloch wave calculation... done in {bwc:.1f}s")  # " (beam pool setup {setup:.1f} s)")
    if v.iter_count == 0: 
        print(f"    {1000*(bwc)/(4*v.image_radius**2):.2f} ms/pixel")

    # increment iteration counter
    v.iter_count += 1

    return


def zncc(img1, img2):
    """ input: img1, img2 sets of n images, both of size [pix_x, pix_x, n]
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


def phase_d_xy(A, B):
    """
    Sub-pixel shift to align B to A using Fourier phase correlation.

    Returns dy, dx : float
        Shift to apply to B so it aligns with A
    """

    A = A.astype(np.float64)
    B = B.astype(np.float64)
    FA = np.fft.fft2(A)
    FB = np.fft.fft2(B)
    # Cross power spectrum
    R = FA * np.conj(FB)
    R /= np.abs(R) + 1e-12   # normalize, avoid divide-by-zero
    # Correlation surface
    r = np.fft.ifft2(R)
    r = np.abs(r)
    # Integer peak
    y0, x0 = np.unravel_index(np.argmax(r), r.shape)
    # Wrap peak to signed coordinates
    ny, nx = r.shape
    if y0 > ny // 2:
        y0 -= ny
    if x0 > nx // 2:
        x0 -= nx

    # --- Sub-pixel refinement (parabolic fit) ---
    def subpix(fm1, f0, fp1):
        d = 2*f0 - fm1 - fp1
        if d == 0:
            return 0.0
        return 0.5 * (fp1 - fm1) / d

    # indices with wrap
    ym1, yp1 = (y0 - 1) % ny, (y0 + 1) % ny
    xm1, xp1 = (x0 - 1) % nx, (x0 + 1) % nx
    dy = y0 + subpix(r[ym1, x0], r[y0, x0], r[yp1, x0])
    dx = x0 + subpix(r[y0, xm1], r[y0, x0], r[y0, xp1])

    return dy, dx


def pcc(stack1, stack2):
    """ input: stack1, stack2 sets of n images, both of size [pix_x, pix_y, n]
    We correct for sub-pixel shits using fourier shifting
    output is a numpy array of length n, giving pcc for each pair of images
    pcc is -1 = perfect anticorrelation, +1 = perfect correlation
    """
    if stack1.ndim == 2:
        stack1 = stack1[:, :, np.newaxis]
        stack2 = stack2[:, :, np.newaxis]
    n = stack1.shape[2]
    pcc = np.zeros(n)
    shifts = np.zeros((n, 2))

    for i in range(n):
        img1 = stack1[:, :, i]
        img2 = stack2[:, :, i]

        # Estimate sub-pixel shift
        up = 10  # shifts are accurate to 1/upsample_factor
        shift, error, diffphase = phase_cross_correlation(img1, img2,
                                                          upsample_factor=up)
        shifts[i] = shift

        # Shift img2 in Fourier space
        img2_shifted = np.real(ifftn(fourier_shift(fftn(img2), shift)))

        # Compute correlation coefficient
        pcc[i] = np.corrcoef(img1.ravel(), img2_shifted.ravel())[0, 1]

    return pcc  # , shifts


def optimise_pool(v):
    """
    runs simulations with decreasing numbers of strong beams
    gives a plot of intensity change to inform best pool size for a refinement
    """
    # baseline simulation = highest fidelity: pool 600 strong 250
    poo = 600
    stro = 250
    v.min_reflection_pool = poo
    v.min_strong_beams = stro
    print(f"Baseline simulation: beam pool {poo}, {stro} strong beams")
    simulate(v)
    print_LACBED(v, 0)
    baseline = np.copy(v.lacbed_sim)
    # subtract mean and divide by SD
    for i in range(v.n_thickness):
        for j in range(v.n_out):
            a0 = baseline[i, :, :, j]
            a = (a0 - np.mean(a0))/np.std(a0)
            baseline[i, :, :, j] = a
    # now do decreasing beam pool size and compare against baseline
    strong = np.array([200, 150, 100, 75, 50, 25])
    n_strong = len(strong)
    diff_max = np.zeros([n_strong, v.n_thickness, v.n_out])  # max difference
    diff_mean = np.zeros([n_strong, v.n_thickness, v.n_out])  # mean difference
    for k in range(n_strong):
        v.min_strong_beams = strong[i]
        print("-------------------------------")
        print(f"Simulation: beam pool {poo}, {strong[i]} strong beams")
        simulate(v)
        for i in range(v.n_thickness):
            for j in range(v.n_out):
                a0 = v.lacbed_sim[i, :, :, j]
                a = (a0 - np.mean(a0))/np.std(a0)
                b = baseline[i, :, :, j]
                pcc = b-a
                v.diff_image[:, :, j] = pcc
                diff_max[k, i, j] = np.max(abs(pcc))
                diff_mean[k, i, j] = np.mean(abs(pcc))
            print_LACBED(v, 2)
    # make some plots
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    for i in range(v.n_thickness):
        max_ = np.sum(diff_max, axis=2)  # max[strong, thickness]
        plt.scatter(strong, max_[:, i])
    ax.set_xlabel('Strong beams', size=24)
    ax.set_ylabel('Max difference', size=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    for i in range(v.n_thickness):
        mean_ = np.sum(diff_mean, axis=2)  # max[strong, thickness]
        plt.scatter(strong, max_[:, i])
    ax.set_xlabel('Strong beams', size=24)
    ax.set_ylabel('Max difference', size=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()

    return diff_max, diff_mean


def figure_of_merit(v):
    """
    takes as an input v.lacbed_sim, shape [v.n_thickness, pix_x, pix_y, n_out]
    applies image processing if required
    image processing = 0 -> no Gaussian blur (applied with radius 0)
    image processing = 1 -> Gaussian blur radius defined in felix.inp
    image processing = 2 -> find the best blur radius
    uses zncc/pcc, which work on sets of images both of size [pix_x, pix_y, n]
    currently returns a single figure of merit fom that is mean of zncc's for
    the best thickness.  Could give a more sophisticated analysis..
    """
    # figure of merit - might need a NaN check? size [n_thick, n_out]
    fom_array = np.ones([v.n_thickness, v.n_out])
    # difference images
    v.diff_image = np.copy(v.lacbed_expt)
    # set up plot for blur optimisation
    if v.plot >= 2 and v.image_processing == 2:
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
                for j in range(v.n_out):
                    blacbed[:, :, j] = gaussian_filter(blacbed[:, :, j],
                                                       sigma=r)
                # b_fom.append(np.mean(1.0 - zncc(v.lacbed_expt, blacbed)))
                b_fom.append(np.mean(1.0 - pcc(v.lacbed_expt, blacbed)))
            # if v.plot >= 2:
            #     plt.plot(radii, b_fom)
            v.blur_radius = radii[np.argmin(b_fom)]
        if v.image_processing != 0:
            for j in range(v.n_out):
                v.lacbed_sim[i, :, :, j] = gaussian_filter(v.lacbed_sim[i, :, :, j],
                                                           sigma=v.blur_radius)
        v.lacbed_expt = np.copy(v.lacbed_expt_raw)
        # sub-pixel shift for correlation if required
        if v.correlation_type == 2:
            for j in range(v.n_out):
                # we only do this for zncc images that exist
                if np.sum(v.lacbed_expt_raw[j]) != 0:
                    a0 = v.lacbed_sim[i, :, :, j]
                    b0 = v.lacbed_expt_raw[:, :, j]
                    # zero mean normalise the images
                    a = (a0 - np.mean(a0))/np.std(a0)
                    b = (b0 - np.mean(b0))/np.std(b0)
                    # the shift
                    dy, dx = phase_d_xy(a, b)
                    c = shift(b, shift=(dy, dx), order=3,
                              mode="constant", cval=0)
                    # replace empty experimental pixels with simulation
                    # (which )prevents them from contribution to the zncc)
                    c[c == 0] = a[c == 0]
                    v.lacbed_expt[:, :, j] = c

        # figure of merit
        if v.correlation_type == 0 or 2:
            fom_array[i, :] = 1.0 - zncc(v.lacbed_expt,
                                         v.lacbed_sim[i, :, :, :])
        elif v.correlation_type == 1:
            fom_array[i, :] = 1.0 - pcc(v.lacbed_expt,
                                        v.lacbed_sim[i, :, :, :])
        else:
            raise ValueError("Invalid correlation_type !(0 or 1) in felix.inp")

        # difference images
        if v.plot ==3:
            for j in range(v.n_out):
                a0 = v.lacbed_sim[i, :, :, j]
                b0 = v.lacbed_expt[:, :, j]
    
                # zero mean normalise the images
                a = (a0 - np.mean(a0))/np.std(a0)
                b = (b0 - np.mean(b0))/np.std(b0)
                v.diff_image[:, :, j] = a-b
            print_LACBED(v, 2)

    # plot of blur fit when v.image_processing == 2
    if v.plot >= 2 and v.image_processing == 2:
        plt.show()
    # print best values
    if v.image_processing == 2:
        print(f"  Best blur={v.blur_radius:.1f}")
    if v.n_thickness > 1:
        v.best_t = np.argmin(np.mean(fom_array, axis=1))
        print(f"  Best thickness {0.1*v.thickness[v.best_t]:.1f} nm")
        # mean figure of merit
        fom = np.mean(fom_array[v.best_t])
    else:
        v.best_t = 0
        fom = np.mean(fom_array[0])

    # plot FoM vs thickness for all LACBED patterns
    if v.plot >= 2 and v.n_thickness > 1:
        fig, ax = plt.subplots(1, 1)
        w_f = 10
        fig.set_size_inches(1.5*w_f, w_f)
        plt.plot(v.thickness/10, np.mean(fom_array, axis=1), 'ro', linewidth=2)
        colours = plt.cm.gnuplot(np.linspace(0, 1, v.n_out))
        # I have 99 styles but . ain't one
        styles = ['-', '-.', '--', ':', '-', '-.', '--', ':', '-', '-.', '--',
                  ':', '-', '-.', '--', ':', '-', '-.', '--', ':', '-', '-.',
                  '--', ':', '-', '-.', '--', ':', '-', '-.', '--', ':', '-',
                  '-.', '--', ':', '-', '-.', '--', ':', '-', '-.', '--', ':',
                  '-', '-.', '--', ':', '-', '-.', '--', ':', '-', '-.', '--',
                  ':', '-', '-.', '--', ':', '-', '-.', '--', ':', '-', '-.',
                  '--', ':', '-', '-.', '--', ':', '-', '-.', '--', ':', '-',
                  '-.', '--', ':', '-', '-.', '--', ':', '-', '-.', '--', ':',
                  '-', '-.', '--', ':', '-', '-.', '--', ':', '-', '-.', '--']
        for i in range(v.n_out):
            annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
            plt.plot(v.thickness/10, fom_array[:, i],
                     color=colours[i],
                     linestyle=styles[i],
                     label=annotation)
        ax.set_xlabel('Thickness (nm)', size=24)
        ax.set_ylabel('Figure of merit', size=24)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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
    how this works!

    All updated variables are in the global class v so no specific return
    
    v.refined_variable_type:
    10 A1 = Ug amplitude *** NOT YET IMPLEMENTED ***
    11 A2 = Ug phase *** NOT YET IMPLEMENTED ***
    20 B = atom coordinate *** PARTIALLY IMPLEMENTED *** not all space groups
    21 C = occupancy
    22 D = isotropic atomic displacement parameter (ADP)
    23,24,25,26,27,28 E = anisotropic atomic displacement parameters (ADPs)
    30,31,32 F = lattice parameters ***PARTIALLY IMPLEMENTED*** not rhombohedral
    33,34,35 G = unit cell angles *** NOT YET IMPLEMENTED ***
    40 H = convergence angle
    41 I = accelerating_voltage_kv *** NOT YET IMPLEMENTED ***
    50 J = Kappa
    51 K = valence electrons
    """

    # will tackle this when doing atomic position refinement
    # basis_atom_delta.fill(0)  # Reset atom coordinate uncertainties to zero

    typ = v.refined_variable_type // 10  # variable type
    sub = v.refined_variable_type % 10  # variable subtype

    for i in range(v.n_variables):
        j = v.atom_refine_flag[i]  # neat
        var = np.copy(v.refined_variable[i])

        if typ[i] == 0:
            # Structure factor refinement (handled elsewhere)
            variable_check = 1

        elif typ[i] == 2:
            if sub[i] == 0:  # Atomic coordinates
                atom_id = j
                # Update position: r' = r - v*(r.v) + v*current_var
                r_dot_v = np.dot(v.basis_atom_position[atom_id],
                                 v.atom_refine_vec[i])
                v.basis_atom_position[atom_id, :] = np.mod(
                    v.basis_atom_position[atom_id, :] + v.atom_refine_vec[i] *
                    (var - r_dot_v), 1)

                # error estimate - needs work
                # Update uncertainty if independent_delta is non-zero
                # if abs(independent_delta[i]) > 1e-10:  # Tiny threshold
                #     basis_atom_delta[atom_id, :] += vector[j - 1, :]
                #                                   * independent_delta[i]

            elif sub[i] == 1:  # Occupancy
                v.basis_occupancy[j] = var
                # shared occupancy is held in v.basis_mult_occ
                if v.basis_mult_occ[j] != 0:
                    # get the indices of atoms on the same site
                    mask = v.basis_mult_occ == v.basis_mult_occ[j]
                    mask[j] = False
                    # scale their occupancies in propotion to existing
                    v.basis_occupancy[mask] *= (1 - var) \
                        / v.basis_occupancy[mask].sum()

            elif sub[i] == 2:   # Iso ADPs
                if 0 < var:  # must lie in range
                    v.basis_u_ij[j, 0, 0] = var / (8 * np.pi**2)
                    v.basis_u_ij[j, 1, 1] = var / (8 * np.pi**2)
                    v.basis_u_ij[j, 2, 2] = var / (8 * np.pi**2)
                else:
                    v.basis_u_ij[j, :, :] = 0.0
            elif sub[i] == 3:  # u[1,1]
                v.basis_u_ij[j, 0, 0] = var
            elif sub[i] == 4:  # u[1,1]
                v.basis_u_ij[j, 1, 1] = var
            elif sub[i] == 5:  # u[2,2]
                v.basis_u_ij[j, 2, 2] = var
            elif sub[i] == 6:  # u[1,2]
                v.basis_u_ij[j, 0, 1] = var
                v.basis_u_ij[j, 1, 0] = var
            elif sub[i] == 7:  # u[1,3]
                v.basis_u_ij[j, 0, 2] = var
                v.basis_u_ij[j, 2, 0] = var
            elif sub[i] == 8:  # u[2,3]
                v.basis_u_ij[j, 1, 2] = var
                v.basis_u_ij[j, 2, 1] = var

        elif typ[i] == 3:
            # Lattice parameters a, b, c
            if sub[i] == 0:
                v.cell_a = v.cell_b = v.cell_c = var
            elif sub[i] == 1:
                v.cell_b = var
            elif sub[i] == 2:
                v.cell_c = var
            elif sub[i] == 3:
                v.cell_alpha = var
            elif sub[i] == 4:
                v.cell_beta = var
            elif sub[i] == 5:
                v.cell_gamma = var

        elif typ[i] == 4:
            if sub[i] == 0:  # Convergence angle
                v.convergence_angle = var
            elif sub[i] == 1:  # Accelerating voltage
                v.accelerating_voltage_kv = var

        elif typ[i] == 5:
            if sub[i] == 0:  # kappa
                if 0.7 < var < 1.3:  # must lie in a reasonable range
                    v.basis_kappa[j] = var*1.0
                else:
                    v.basis_kappa[j] = np.clip(var, 0.7, 1.3)
            elif sub[i] == 1:  # valence electrons
                if 0.5 < var < 1.5:  # must lie in a reasonable range
                    v.basis_pv[j] = var*1.0
                else:
                    v.basis_pv[j] = 0.0
                    # v.basis_pv[j] = np.clip(var, ,1.5 )
    return


def print_montage(v, images, lut, j=0):
    '''
    images[wid, wid, n] = array of n images each of size [wid, wid]
    '''
    n = images.shape[2]
    w = int(np.ceil(np.sqrt(n)))
    h = int(np.ceil(n/w))
    fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
    text_effect = withStroke(linewidth=3, foreground='black')
    axes = axes.flatten()
    for i in range(n):
        img = images[:, :, i]
        # difference or LACBED pattern
        if lut == "diff":
            cmap = LinearSegmentedColormap.from_list(
                "two_color_black_center",
                [(0.0, "c"), (0.5, "k"), (1.0, "orange")])
            norm = TwoSlopeNorm(vmin=img.min(), vcenter=0.0, vmax=img.max())
            axes[i].imshow(img, cmap=cmap, norm=norm)
        else:
            axes[i].imshow(img, cmap=lut)

        axes[i].axis('off')
        annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
        axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                         size=30, color='w', path_effects=[text_effect])
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    annotation = f"{v.thickness[j]/10:.0f} nm"
    plt.annotate(annotation, xy=(0.105, 0.96), xycoords='figure fraction',
                 size=30, color='c', path_effects=[text_effect])
    plt.show()


def print_LACBED(v, image_type):
    '''
    Plots all LACBED patterns in a montage
    image_type options 0=sim, 1=expt, 2=difference
    '''
    if image_type == 0:  # simulation output
        # only print all thicknesses for the first simulation
        if v.iter_count == 1:
            for j in range(v.n_thickness):
                out_image = v.lacbed_sim[j, :, :, :]
                print_montage(v, out_image, 'pink', j)
        else:
            out_image = v.lacbed_sim[v.best_t, :, :, :]
            print_montage(v, out_image, 'pink')
    elif image_type == 1:  # experiment output
        out_image = v.lacbed_expt
        print_montage(v, out_image, 'grey')
    elif v.plot >= 3:
        out_image = v.diff_image
        print_montage(v, out_image, 'diff')


def print_LACBED_pattern(i, j, v):
    # Prints an individual LACBED pattern
    # i = 0  # index of the pattern to plot
    # j = 0  # index of the thickness to plot
    fig, ax = plt.subplots(1, 1)
    text_effect = withStroke(linewidth=3, foreground='black')
    ax.imshow(v.lacbed_sim[j, :, :, i], cmap='pink')
    ax.axis('off')
    annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
    ax.annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                     size=30, color='w', path_effects=[text_effect])


def save_LACBED(v):
    '''
    Saves all LACBED patterns in .npy and .png format
    '''
    j = 0
    # j = v.best_t
    print(os.getcwd())
    if not os.path.isdir(v.chemical_formula_sum):
        os.mkdir(v.chemical_formula_sum)
    os.chdir(v.chemical_formula_sum)
    for i in range(v.lacbed_sim.shape[3]):
        signed_str = "".join(f"{x:+d}" for x in v.hkl[v.g_output[i], :])
        fname = f"{v.chemical_formula_sum}_{signed_str}.bin"
        v.lacbed_sim[j, :, :, i].tofile(fname)
        fname = f"{v.chemical_formula_sum}_{signed_str}.png"
        plt.imsave(fname, v.lacbed_sim[j, :, :, i], cmap='gray')
    os.chdir("..")


def print_current_var(v, i):
    # prints the variable being refined
    typ = v.refined_variable_type[i]  # variable type & subtype
    atom_id = v.atom_refine_flag[i]
    label = v.basis_atom_label[atom_id]

    # dictionary of format strings
    formats = {
        10: (f"Current Ug", "{:.3f}"),
        11: (f"Current Ug", "{:.3f}"),
        21: (f" Atom {atom_id}: {label} Current occupancy", "{:.2f}"),
        22: (f" Atom {atom_id}: {label} Current B_iso", "{:.2f}"),
        23: (f" Atom {atom_id}: {label} Current U[1,1]", "{:.5f}"),
        24: (f" Atom {atom_id}: {label} Current U[2,2]", "{:.5f}"),
        25: (f" Atom {atom_id}: {label} Current U[3,3]", "{:.5f}"),
        26: (f" Atom {atom_id}: {label} Current U[1,2]", "{:.5f}"),
        27: (f" Atom {atom_id}: {label} Current U[1,3]", "{:.5f}"),
        28: (f" Atom {atom_id}: {label} Current U[2,3]", "{:.5f}"),
        30: (f"Current lattice parameter a", "{:.4f}"),
        31: (f"Current lattice parameter b", "{:.4f}"),
        32: (f"Current lattice parameter c", "{:.4f}"),
        33: (f"Current lattice alpha", "{:.4f}"),
        34: (f"Current lattice beta", "{:.4f}"),
        35: (f"Current lattice gamma", "{:.4f}"),
        40: (f"Current convergence angle", "{:.3f} Å^-1"),
        41: (f"Current accelerating voltage", "{:.1f} kV"),
        50: (f" Atom {atom_id}: Current Kappa", "{:.3f}"),
        51: (f" Atom {atom_id}: Current proportion of valence electrons", "{:.4f}")
            }

    if typ == 20:  # atomic coords
        with np.printoptions(formatter={'float': lambda x: f"{x:.4f}"}):
            print(f"  Atom {atom_id}: {label}  {v.basis_atom_position[atom_id, :]}")
    elif typ in formats:
        label, fmt = formats[typ]
        print(f"  {label} {fmt.format(v.refined_variable[i])}")


def variable_message(vtype):
    """Map variable type → message string."""
    msg = {
        10: "Changing Ug amplitude",
        11: "Changing Ug phase",
        20: "Changing atom coordinates",
        21: "Changing occupancy",
        22: "Changing B_iso",
        23: "Changing U[1,1]",
        24: "Changing U[2,2]",
        25: "Changing U[3,3]",
        26: "Changing U[1,2]",
        27: "Changing U[1,3]",
        28: "Changing U[2,3]",
        30: "Changing lattice parameter a", 
        31: "Changing lattice parameter b", 
        32: "Changing lattice parameter c", 
        33: "Changing lattice alpha", 
        34: "Changing lattice beta", 
        35: "Changing lattice gamma", 
        40: "Changing convergence angle",
        41: "Changing accelerating voltage",
        50: "Changing Kappa",
        51: "Changing proportion of valence electrons",
    }
    return msg[vtype]


def sim_fom(v, i):
    '''
    wraps multiple subroutine calls into a single line: update variables
    simulate and figure of merit
    input i = index of variable to be refined, for single variables
    i = -1 for multiple variables
    '''
    update_variables(v)
    print_current_var(v, i)
    simulate(v)
    # figure of merit
    fom = figure_of_merit(v)
    v.fit_log.append(fom)
    print(f"  Figure of merit {100*fom:.3f}% (previous best {100*v.best_fit:.3f}%)")

    return fom


def refine_single_variable(v, i):
    '''
    Does 3-point refinement for a single variable and if no minimum is found
    retuns the step size for a subsequent multidimensional refinement
    i: integer, index of variable being refined
    Uses the whole variable space v as it is passed on to the simulation:
    v.best_fit: float, best figure of merit
    v.best_var: float array, the refined variables with best FoM
    v.refined_variable: float array of variables being refined
    v.refined_variable_type: integer array, type of variable being refined
    v.refinement_scale: float, step to change the variable (from felix.inp)
    v.next_var: float array of predicted variable values
    ----------------
    Updates:
        v.best_fit, v.best_var if necessary
        v.next_var as an output
    Returns:
        dydx_i: the gradient of this variable
    '''
    r3_var = np.zeros(3)  # for parabolic minimum
    r3_fom = np.zeros(3)
    # Check if ADP is negative, skip if so NB u12,u13,u23 can be -ve
    if 21 < v.refined_variable_type[i] < 26 and v.refined_variable[i] < 1e-10:
        dydx_i = 0.0
    else:
        # middle point is the previous best simulation
        r3_var[1] = v.best_var[i]*1.0
        r3_fom[1] = v.best_fit*1.0
        print(f"Finding gradient, variable {i+1} of {v.n_variables}")
        print(variable_message(v.refined_variable_type[i]))
        # print_current_var(v, i)

        # delta is a small change in the current variable
        # which is either refinement_scale for atomic coordinates and
        # refinement_scale*variable for everything else
        delta = abs(v.refinement_scale * v.refined_variable[i])
        if v.refined_variable_type[i] == 20:
            delta = abs(v.refinement_scale)
        # Three-point gradient measurement, starting with plus
        v.refined_variable[i] += delta
        # simulate and get figure of merit
        fom = sim_fom(v, i)
        r3_var[2] = v.refined_variable[i]*1.0
        r3_fom[2] = fom*1.0
        print("-1-----------------------------")  # " {r3_var},{r3_fom}")
        # update best fit
        if (fom < v.best_fit):
            v.best_fit = fom*1.0
            v.best_var = np.copy(v.refined_variable)

        # keep going or turn round?
        if r3_fom[2] < r3_fom[1]:  # keep going
            v.refined_variable[i] += np.exp(0.5) * delta
        else:  # turn round
            delta = - delta
            v.refined_variable[i] += np.exp(0.75) * delta
        # simulate and get figure of merit
        fom = sim_fom(v, i)
        r3_var[0] = v.refined_variable[i]*1.0
        r3_fom[0] = fom*1.0
        print("-2-----------------------------")  # " {r3_var},{r3_fom}")
        # update best fit
        if (fom < v.best_fit):
            v.best_fit = fom*1.0
            v.best_var = np.copy(v.refined_variable)

        # test to see if the variable has an effect
        if np.max(r3_fom)-np.min(r3_fom) < 1e-5:  # no effect on FoM
            exclude = True
            print(f"  Low effect, fixed at {v.best_var[i]}")
        else:
            v.next_var[i], exclude = px.convex(r3_var, r3_fom)
        # predict the next point as a minimum or a step onwards
        # v.next_var gets passed on to the multidimensional refinement
        # as a global variable
        if exclude:
            dydx_i = 0.0  # this variable excluded from vector downhill
        else:
            # we weight the variable by -df/delta
            dydx_i = -(r3_fom[2] - r3_fom[0]) / (2 * delta)
        # error estimate goes here
        # independent_delta[i] = delta_x(r3_var, r3_fom, precision, err)
        # uncert_brak(var_min, independent_delta[i])

    return dydx_i


def variable_check(x, t):
    '''
    x: float, variable being refined
    t: integer, type of variable being refined
    '''
    continue_ = True
    #Atomic displacement parameters
    if int(t/10) == 2 and np.mod(t, 10) > 0:
        if x < 0:
            x = 0.0
            continue_ = False
            print("  ADP set to zero")
    return x, continue_


def refine_multi_variable(v, dydx, single=True):
    '''
    multidimensional refinement
    dydx: float array of gradients, generated in refine_single_variable
    Uses the whole variable space v, only:
    v.refined_variable: array of variables to refine size [n_var]
    v.refined_variable_type: what kind of variable (see felixrefine)
    v.best_fit: best figure of merit so far
    v.best_var: array of variables that gives best fit
    v.refinement_scale: size of change in variable to obtain gradient

    dydx = array of gradients, size [n_var]
    '''
    # starting point is the current best set of variables
    last_fit = 1.0*v.best_fit

    n_var = np.count_nonzero(dydx)
    if n_var > 1:
        print(f"Multidimensional refinement, {n_var} variables")
        with np.printoptions(formatter={'float': lambda x: f"{x:.3f}"}):
            print(f"    Refinement vector {dydx}")
        p_mag = np.linalg.norm(dydx)
        if np.isinf(p_mag) or np.isnan(p_mag):
            raise ValueError("Infinite or NaN gradient!")
        dydx = dydx / p_mag   # Normalized direction of max gradient
    elif n_var == 1:
        print("Single variable refinement")
    else:
        raise ValueError("No refinement variables defined!")
    # index of principal variable
    j = np.argmax(abs(dydx))
    t = v.refined_variable_type[j]
    print(f"  Principal variable: {variable_message(t)}")

    # Check the gradient vector magnitude and initialize vector descent
    if not single:
        print(f"    Extrapolation, should be better than {100*v.best_fit:.2f}%")
        # initial trial uses the predicted best set of variables
        v.refined_variable = 1.0*v.next_var
        # simulate and get figure of merit
        fom = sim_fom(v, j)
        # is it actually any better
        if fom < last_fit:
            v.best_fit = fom*1.0
            v.best_var = np.copy(v.refined_variable)
            print("Point 1 of 3: extrapolated")  # yes, use it
        else:
            print("Point 1 of 3: previous best")  # no, use the best
        v.refined_variable = np.copy(v.best_var)
        print_LACBED(v, 0)
        print("-a-----------------------------")  # "{r3_var},{r3_fom}")

    # First point: incoming best simulation
    r3_var = np.zeros(3)
    r3_fom = np.zeros(3)
    r3_var[0] = 1.0*v.best_var[j]  # using principal variable
    r3_fom[0] = 1.0*v.best_fit

    # set the refinement scale
    # if v.refined_variable_type[j] == 20:  # atom coordinates, absolute value
    delta = dydx * v.refinement_scale
    # delta = v.best_var[j] * v.refinement_scale

    # Second point
    print("Refining, point 2 of 3")
    # Change the array of variables by a small amount
    v.refined_variable += delta  # point 2
    # check for validity: ADPs must be >=0
    v.refined_variable[j], cont = variable_check(v.refined_variable[j], t)
    fom = sim_fom(v, j)  # simulate and get figure of merit
    # check for no effect
    if fom == v.best_fit:
        raise ValueError(f"{variable_message(t)} has no effect!")
    r3_var[1] = 1.0*v.refined_variable[j]
    r3_fom[1] = 1.0*fom
    print(f"-b-----------------------------{r3_var},{r3_fom}")
    if fom < v.best_fit:
        v.best_fit = fom*1.0
        v.best_var = np.copy(v.refined_variable)
    if not cont:
        dydx[j] = 0.0
        return dydx

    # Third point
    print("Refining, point 3 of 3")
    if r3_fom[1] > r3_fom[0]:  # if second point is worse
        # Go in the opposite direction
        v.refined_variable -= np.exp(0.8)*delta
    else:  # keep going
        v.refined_variable += np.exp(0.4)*delta
    v.refined_variable[j], cont = variable_check(v.refined_variable[j], t)
    fom = sim_fom(v, j)
    r3_var[2] = 1.0*v.refined_variable[j]
    r3_fom[2] = 1.0*fom
    print(f"-c----------------------------- {r3_var},{r3_fom}")
    if fom < v.best_fit:
        v.best_fit = fom*1.0
        v.best_var = np.copy(v.refined_variable)
    if not cont:
        dydx[j] = 0.0
        return dydx

    # We continue downhill until we get a predicted minnymum
    minny = False
    while minny is False:
        last_x = 1.0*v.refined_variable[j]
        # predict the next point as a minimum or a step on
        next_x, minny = px.convex(r3_var, r3_fom)
        v.refined_variable *= next_x/last_x
        # print(f"**..** next x = {v.refined_variable[j]}")
        v.refined_variable[j], cont = variable_check(v.refined_variable[j], t)
        fom = sim_fom(v, j)
        if (fom < v.best_fit):  # it's better, keep going
            v.best_fit = fom*1.0
            v.best_var = np.copy(v.refined_variable)
        if not cont:  # variable has gone out of the valid range
            dydx[j] = 0.0
            return dydx
        # replace worst point with this one
        i = np.argmax(r3_fom)
        r3_var[i] = 1.0*v.refined_variable[j]
        r3_fom[i] = 1.0*fom
        with np.printoptions(formatter={'float': lambda x: f"{x:.4f}"}):
            print(f"-.-----------------------------{r3_var}: {r3_fom}")
        # if (fom < v.best_fit):  # it's better, keep going
        #     v.best_fit = fom*1.0
        #     v.best_var = np.copy(v.refined_variable)
        # else:  # we must have a minimum
        #     last_x = 1.0*v.refined_variable[j]
        #     # check it out
        #     next_x, minny = px.convex(r3_var, r3_fom)
        #     v.refined_variable += dydx * (next_x-last_x)/next_x
        #     fom = sim_fom(v, j)
        #     if (fom < v.best_fit):  # it's better, keep going
        #         v.best_fit = fom*1.0
        #         v.best_var = np.copy(v.refined_variable)
        #     if not cont:
        #         dydx[j] = 0.0
        #         return dydx
    # we have taken the principal variable to a minimum
    dydx[j] = 0.0
    print(f"    ====Refined variable {j}====")
    print_LACBED(v, 0)

    return dydx
