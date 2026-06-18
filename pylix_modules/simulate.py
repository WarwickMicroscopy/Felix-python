# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:26:54 2024

@author: Richard

Contains the subroutines needed to produce a LACBED simulation
Each of which call further pylix subroutines
Returns the simulated LACBED patterns

"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.constants import c, h, e, m_e, angstrom, epsilon_0
from scipy.ndimage import gaussian_filter
# from skimage import transform, registration
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
from skimage.filters import sobel
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


def simulate(xtal, basis, cell, hkl, bloch, cbed, rc):

    typ = rc.refined_variable_type // 10  # array of variable types

    # some setup calculations
    # Electron velocity in metres per second
    bloch.electron_velocity = (c * np.sqrt(1.0 - ((m_e * c**2) /
                               (e * rc.accelerating_voltage_kv*1000.0 +
                                m_e * c**2))**2))
    # Electron wavelength in Angstroms
    electron_wavelength = h / (
        np.sqrt(2.0 * m_e * e * rc.accelerating_voltage_kv*1000.0) *
        np.sqrt(1.0 + (e * rc.accelerating_voltage_kv*1000.0) /
                (2.0 * m_e * c**2))) / angstrom
    # Wavevector magnitude k
    electron_wave_vector_magnitude = 2.0 * np.pi / electron_wavelength
    # Relativistic correction
    bloch.relativistic_correction = (1.0 / np.sqrt(1.0 -
                                     (bloch.electron_velocity / c)**2))
    # Cell volume
    xtal.cell_volume = (xtal.cell_a*xtal.cell_b*xtal.cell_c
                        * np.sqrt(1.-np.cos(xtal.cell_alpha)**2 -
                                  np.cos(xtal.cell_beta)**2 -
                                  np.cos(xtal.cell_gamma)**2 +
                                  2.0*np.cos(xtal.cell_alpha) *
                                  np.cos(xtal.cell_beta) *
                                  np.cos(xtal.cell_gamma)))
    # Conversion from scattering factor to volts
    xtal.mott = m_e * e**2 * angstrom / (8.0 * np.pi * epsilon_0 * h**2)
    scatt_fac_to_volts = (h**2 /
                          (2.0*np.pi * m_e * e * xtal.cell_volume *
                           angstrom**2))

    # ===============================================
    # fill the unit cell and get mean inner potential
    # only if necessary, i.e. for B, C, D, E
    if np.any(typ == 2) or rc.iter_count == 0:
        px.unique_atom_positions(xtal, basis, cell, rc)
        # mean inner potential as the sum of scattering factors at g=0
        # multiplied by h^2/(2pi*m0*e*CellVolume)
        mip = 0.0
        for i in range(cell.n_atoms):  # get the scattering factor
            if rc.scatter_factor_method == 0:
                mip += px.f_kirkland(cell.atomic_number[i], 0.0)
            elif rc.scatter_factor_method == 1:
                mip += px.f_lobato(cell.atomic_number[i], 0.0)
            elif rc.scatter_factor_method == 2:
                mip += px.f_peng(cell.atomic_number[i], 0.0)
            elif rc.scatter_factor_method == 3:
                mip += px.f_doyle_turner(cell.atomic_number[i], 0.0)
            elif rc.scatter_factor_method > 3:
                mip += px.f_kirkland(cell.atomic_number[i], 0.0)
            else:
                raise ValueError("No scattering factors chosen in felix.inp")
        mip = mip.item()*scatt_fac_to_volts  # NB convert array to float

        # Wave vector magnitude in crystal
        # high-energy approximation (not HOLZ compatible)
        # K^2=k^2+U0
        # big_k_mag = electron_wave_vector_magnitude  # version without mip
        bloch.big_k_mag = np.sqrt(electron_wave_vector_magnitude**2+mip)
        # k-vector for the incident beam (k is along z in the microscope frame)
        bloch.big_k = np.array([0.0, 0.0, bloch.big_k_mag])

    if rc.iter_count == 0:
        print(f"  There are {cell.n_atoms} atoms in the unit cell")
        print(f"  Mean inner potential = {mip:.1f} Volts")

    # output for debugging
    if rc.debug > 1:
        print("Symmetry operations:")
        for i in range(len(xtal.symmetry_matrix)):
            print(f"{i+1}: {xtal.symmetry_matrix[i]}, {xtal.symmetry_vector[i]}")
    if rc.debug > 0:
        np.set_printoptions(precision=5, suppress=True)
        print("atomic coordinates")
        for i in range(cell.n_atoms):
            print(f"{cell.atom_label[i]} {cell.atom_name[i]}: {cell.atom_position[i]}")

    # ===============================================
    # set up reference frames
    # only if necessary, i.e. for cell dimensions F, G
    if np.any(typ == 3) or rc.iter_count == 0:
        px.reference_frames(xtal, cell, rc)
    # apply the reference frame to the unit cell
    if np.any(typ == 2) or rc.iter_count == 0:
        px.update_coordinates(xtal, cell)
        if rc.debug > 0:
            np.set_printoptions(precision=5, suppress=True)
            print("u_aniso in microscope frame")
            for i in range(cell.n_atoms):
                print(f"{cell.atom_label[i]}: {cell.u_aniso_m[i]}")
    # plot unit cell and save .xyz file
    if rc.iter_count == 0 and rc.plot >= 1:
        atom_cvals = mcolors.Normalize(vmin=1, vmax=103)
        atom_cmap = plt.cm.prism
        atom_colours = atom_cmap(atom_cvals(cell.atomic_number))
        border_cvals = mcolors.Normalize(vmin=0, vmax=1)
        border_cmap = plt.cm.plasma
        border_colours = border_cmap(border_cvals(cell.atom_position[:, 2]))
        bb = 5
        fig, ax = plt.subplots(figsize=(bb, bb))
        plt.scatter(cell.atom_coordinate[:, 0], cell.atom_coordinate[:, 1],
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
        #     sas = str(cell.atom_coordinate[i])
        #     xyz = sas[1:len(sas)-1]
        #     text = text + atom_name[i] + "  " + xyz + "\n"
        # fnam = v.chemical_formula_sum+".xyz"
        # f = open(fnam, "x")
        # f.write(str(n_atoms)+"\n")
        # f.write(text)
        # f.close()

    # ===============================================
    # set up beam pool
    # NB g_pool are in reciprocal Angstroms in the microscope reference frame
    # only if necessary, i.e. for cell dimensions F, G and kV I
    strt = time.time()
    if rc.iter_count == 0 or np.any(typ == 3) or np.any(typ == 4):
        px.hkl_make(xtal, hkl, bloch, rc)
        # output redefined to match what we can actually do
        bloch.n_out = len(bloch.hkl_output)
        # NEEDS SOME MORE WORK TO MATCH SIM/EXPT PATTERNS if this happens

    # outputs
    if rc.iter_count == 0:
        print(f"  Beam pool: {bloch.n_hkl} reflexions ({rc.min_strong_beams} strong beams)")
        # we will have larger g-vectors in g_matrix since this has
        # differences g - h
        # but the maximum of the g pool is probably a more useful thing to know
        print(f"  Maximum |g| = {np.max(bloch.g_pool_mag)/(2*np.pi):.3f} 1/Å")
        # for i in range(n_hkl):
        #     print(f"{i},  {bloch.hkl_indices[i]}")

    # plot beam pool
    if rc.iter_count == 0 and rc.plot > 0:
        xm = np.ceil(np.max(bloch.g_pool_mag/(2*np.pi)))
        fig, ax = plt.subplots(1, 1)
        w_f = 10
        fig.set_size_inches(w_f, w_f)
        ax.set_facecolor('black')
        # colour according to Laue zone
        lz_cvals = mcolors.Normalize(vmin=np.min(bloch.g_pool[:, 2]),
                                     vmax=np.max(bloch.g_pool[:, 2]))
        lz_cmap = plt.cm.brg
        lz_colours = lz_cmap(lz_cvals(bloch.g_pool[:, 2]))
        # plots the g-vectors in the pool, colours for different Laue zones
        plt.scatter(bloch.g_pool[:, 0]/(2*np.pi), bloch.g_pool[:, 1]/(2*np.pi),
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

    # ===============================================
    # now make the Ug matrix, i.e. calculate the structure factor Fg for all
    # g-vectors in g_matrix and convert to Ug
    # any change results in recalculation
    px.Fg_matrix(xtal, basis, cell, bloch, rc)
    # plot_f_g(xtal, basis, bloch, 0)

    if rc.iter_count == 0:
        print("    Ug matrix constructed")
    if rc.debug > 0:
        np.set_printoptions(precision=5, suppress=True)
        print(100*bloch.ug_matrix[:5, :5])

    # ===============================================
    # deviation parameter for each pixel and g-vector
    px.deviation_parameter(bloch, rc)

    # ===============================================
    # Bloch wave calculation
    mid = time.time()
    # Dot product of k with surface normal, [image diameter, image diameter]
    bloch.k_dot_n = np.tensordot(bloch.tilted_k, xtal.norm_dir_m,
                                 axes=([2], [0]))
    # # reset output container
    # cbed.lacbed_sim = np.zeros([rc.n_thickness, 2*rc.image_radius,
    #                            2*rc.image_radius, len(bloch.hkl_output)],
    #                            dtype=float)
    print("Bloch wave calculation...", end=' ')
    if rc.debug > 0:
        print("")
        print("output indices")
        print(bloch.hkl_output[:15])

    # = = = = = = = = = = = = = = = = = = = = = = = =
    # pixel by pixel calculations from here
    for pix_x in range(2*rc.image_radius):
        # progess
        print(f"\rBloch wave calculation... {50*pix_x/rc.image_radius:.0f}%", end="")

        for pix_y in range(2*rc.image_radius):
            bloch.s_g_pix = np.squeeze(bloch.s_g[pix_x, pix_y, :])
            bloch.k_dot_n_pix = bloch.k_dot_n[pix_x, pix_y]

            # works for multiple thicknesses
            px.wave_functions(bloch, rc)

            intensity = np.abs(bloch.wave_function)**2

            # Map diffracted intensity to required output g vectors
            # note x and y swapped!
            cbed.lacbed_sim[:, -pix_y, pix_x, :] = intensity[:, :len(bloch.hkl_output)]
    # = = = = = = = = = = = = = = = = = = = = = = = =

    # timings
    setup = mid-strt
    bwc = time.time()-mid
    print(f"\rBloch wave calculation... done in {bwc:.1f}s")  # " (beam pool setup {setup:.1f} s)")
    if rc.iter_count == 0: 
        print(f"    {1000*(bwc)/(4*rc.image_radius**2):.2f} ms/pixel")

    # increment iteration counter
    rc.iter_count += 1

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


def phase_d_xy(img1, img2):
    """
    Sub-pixel shift to align img2 to img1 using Fourier phase correlation.

    Returns dy, dx : float
        Shift to apply to img2 so it aligns with img1
    """

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    FA = np.fft.fft2(img1)
    FB = np.fft.fft2(img2)
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
    shift_ = (dx, dy)

    return shift_


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
        up = 50  # shifts are accurate to 1/upsample_factor
        shift_, error, diffphase = phase_cross_correlation(img1, img2,
                                                           upsample_factor=up)

        # only accept shifts smaller than 5 pixels
        if np.linalg.norm(shift_) < 5.0:
            # Shift img2 in Fourier space
            img2_shifted = np.real(ifftn(fourier_shift(fftn(img2), shift_)))
            # correlation coefficient
            pcc[i] = np.corrcoef(img1.ravel(), img2_shifted.ravel())[0, 1]
            # shifts[i] = shift_

    return pcc  # , shifts


def stretch(d, w):
    s = np.array([[1+d[0], 0, -0.5*d[0]*w],
                  [0, 1+d[1], -0.5*d[1]*w],
                  [0, 0, 1]])
    return AffineTransform(s)


def shif(d, w):
    s = np.array([[1, 0, d[0]*w],
                  [0, 1, d[1]*w],
                  [0, 0, 1]])
    return AffineTransform(s)


def cc_d_xy(img1, img2):
    """
    Sub-pixel shift to align img2 to img1 using sobel-filtered
    Pearson correlation.
    Correlation is done on 1D vectors
    We take the central sub-region of the simulation and compare with
    a set of sub-regions of experiment

    Returns dy, dx : float
        Shift to apply to img2 so it aligns with img1
    """
    s0 = sobel(img1)  # simulation
    e0 = sobel(img2)  # experimental
    
    # normalise
    s0 = (s0 - s0.mean()) / s0.std()
    e0 = (e0 - e0.mean()) / e0.std()
    nx, ny = e0.shape

    # maximum pixel shift to search for best correlation
    d_max = 5
    nw = 2*d_max + 1
    # vectorised central region of simulation
    s1 = s0[d_max:nx-d_max, d_max:ny-d_max]
    s_flat = s1.ravel()  # size [nx*ny]

    # extract regions from experiment and vectorise
    wx = nx - 2*d_max
    wy = ny - 2*d_max
    windows = sliding_window_view(e0, (wx, wy))
    e_flat = windows.reshape(nw*nw, -1)  # size [nw*nw, nx*ny]

    # zncc for nw*nw images
    corr = (e_flat @ s_flat) / s_flat.size
    # corr = np.mean(e_flat * s_flat, axis=1)
    corr2d = corr.reshape(nw, nw)

    # best correlation
    ix, iy = np.unravel_index(np.argmax(corr2d), corr2d.shape)
    xcorr = corr2d[ix-1:ix+2, iy]
    ycorr = corr2d[ix, iy-1:iy+2]
    dx = ix - d_max
    dy = iy - d_max

    if (ix == 0 or ix == 2*d_max or iy == 0 or iy == 2*d_max):
        # maximum lies on boundary
        sx, sy = 0.0, 0.0
    else:    
        #sub-pixel
        sx, _, _ = px.parabo3(np.array([dx-1, dx, dx+1]), xcorr)
        sy, _, _ = px.parabo3(np.array([dy-1, dy, dy+1]), ycorr)

    return np.array([sx, sy])


def cc_s_xy(img1, img2):
    """
    Sub-pixel stretch to align img2 to img1 using sobel-filtered
    Pearson correlation.

    Returns dy, dx : float
        Stretch to apply to img2 so it aligns with img1
    """
    e0 = sobel(img2)  # experimental
    s0 = sobel(img1)  # simulation
    w = e0.shape[0]
    xy_range = np.arange(-0.06, 0.06, 0.001)  # -10% to 10% steps of 1%

    # best y-stretch
    best_fit = -np.inf
    fit_p = []
    for dy in xy_range:
        e1 = warp(e0, inverse_map=stretch(([0, dy]), w).inverse)
        fit = pcc(s0, e1)
        # fit = np.corrcoef(s0.ravel(), e1.ravel())[0, 1]
        fit_p.append(fit)
        if fit > best_fit:
            best_fit = fit
            s_y = dy
    # plt.plot(fit_p)
    # plt.show()

    best_fit = -np.inf
    fit_p = []
    for dx in xy_range:
        e1 = warp(e0, inverse_map=stretch(([dx, 0]), w).inverse)
        fit = pcc(s0, e1)
        # fit = np.corrcoef(s0.ravel(), e1.ravel())[0, 1]
        fit_p.append(fit)
        if fit > best_fit:
            best_fit = fit
            s_x = dx
    # plt.plot(fit_p)
    # plt.show()

    return s_x, s_y


def affine(cbed, rc):
    """
    Determines the x-y stretch to fit the 000 experimental LACBED
    pattern to the best fit simulation using sobel-filtered versions
    Then applies it to all experimental LACBED patterns
    """
    expt000 = cbed.lacbed_expt_raw[:, :, 0]
    sim000 = cbed.lacbed_sim[rc.best_t, :, :, 0]
    w = expt000.shape[0]
    # xy_range = np.arange(-0.1, 0.1, 0.001)  # -10% to 10% steps of 1%
    # plt.imshow(expt000)
    # plt.show()

    # best y-stretch
    s_ii = cc_s_xy(sim000, expt000)
    expt000 = warp(expt000, inverse_map=stretch(s_ii, w).inverse)
    # plt.imshow(expt000)
    # plt.show()

    # translation
    t_ii = cc_d_xy(sim000, expt000)
    # expt000 = warp(expt000, inverse_map=shif(t_ii, w).inverse)
    expt000 = shift(expt000, shift=t_ii, order=3,
                    mode="constant", cval=0)
    # plt.imshow(expt000)
    # plt.show()

    # outputs
    if rc.iter_count != 0 and rc.plot > 1:
        print(f"    Image stretch x={100*s_ii[0]:.1f}%,  y={100*s_ii[1]:.1f}%")

        text_effect = withStroke(linewidth=3, foreground='black')
        fig, ax = plt.subplots(1, 1)
        ax.imshow(sobel(sim000))
        ax.axis('off')
        annotation = "Simulation"
        ax.annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                    size=30, color='w', path_effects=[text_effect])
        plt.show()

        fig, ax = plt.subplots(1, 1)
        ax.imshow(sobel(expt000))
        ax.axis('off')
        annotation = "Experiment"
        ax.annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                    size=30, color='w', path_effects=[text_effect])
        plt.show()

    # apply best transformation
    s = np.array([[1+s_ii[0], 0, (t_ii[0]-0.5*s_ii[0])*w],
                  [0, 1+s_ii[1], (t_ii[1]-0.5*s_ii[1])*w],
                  [0, 0, 1]])
    tform = AffineTransform(s)
    for i in range(cbed.lacbed_expt.shape[2]):
        cbed.lacbed_expt[:, :, i] = warp(cbed.lacbed_expt_raw[:, :, i],
                                         inverse_map=tform.inverse)


def optimise_pool(xtal, basis, cell, hkl, bloch, cbed, rc):
    """
    runs simulations with decreasing numbers of strong beams
    gives a plot of intensity change to inform best pool size for a refinement
    """
    # baseline simulation = highest fidelity: pool 600 strong 250
    poo = 400
    strong = np.array([300, 250, 200, 150, 100, 80, 50, 30])
    n_strong = len(strong)
    times = []
    rc.min_reflection_pool = poo
    rc.min_strong_beams = strong[0]
    print(f"Baseline simulation: beam pool {poo}, {strong[0]} strong beams")
    t0 = time.time()
    simulate(xtal, basis, cell, hkl, bloch, cbed, rc)
    times.append(time.time()-t0)
    print_LACBED(bloch, cbed, rc, 0)
    baseline = np.copy(cbed.lacbed_sim)
    # difference between baseline & simulations with different strong beams
    # cbed.lacbed_diff = np.copy(cbed.lacbed_sim)
    # # subtract mean and divide by SD
    # for i in range(rc.n_thickness):
    #     for j in range(rc.n_out):
    #         a0 = baseline[i, :, :, j]
    #         a = (a0 - np.mean(a0))/np.std(a0)
    #         baseline[i, :, :, j] = a
    # now do decreasing beam pool size and compare against baseline
    diff_max = np.zeros([n_strong, rc.n_thickness, rc.n_out])  # max difference
    diff_mean = np.zeros([n_strong, rc.n_thickness, rc.n_out])
    for k in range(1, n_strong):
        rc.min_strong_beams = strong[k]
        print("-------------------------------")
        print(f"Simulation: beam pool {poo}, {strong[k]} strong beams")
        t0 = time.time()
        simulate(xtal, basis, cell, hkl, bloch, cbed, rc)
        times.append(time.time()-t0)
        for i in range(rc.n_thickness):
            for j in range(rc.n_out):
                a = cbed.lacbed_sim[i, :, :, j]
                # a = (a0 - np.mean(a0))/np.std(a0)
                b = baseline[i, :, :, j]
                pcc = b-a
                cbed.lacbed_diff[i, :, :, j] = pcc
                diff_max[k, i, j] = np.max(abs(pcc))
                diff_mean[k, i, j] = np.mean(abs(pcc))
            print_LACBED(bloch, cbed, rc, 2)

    # make some plots
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    for i in range(rc.n_thickness):
        max_ = np.sum(diff_max, axis=2)  # max[strong, thickness]
        ax.semilogy(strong, max_[:, i])
    ax.set_xlabel('Strong beams', size=24)
    ax.set_ylabel('Max difference', size=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    for i in range(rc.n_thickness):
        mean_ = np.sum(diff_mean, axis=2)  # mean[strong, thickness]
        ax.semilogy(strong, mean_[:, i])
    ax.set_xlabel('Strong beams', size=24)
    ax.set_ylabel('Mean difference', size=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    for i in range(rc.n_thickness):
        plt.scatter(strong, times)
    ax.set_xlabel('Strong beams', size=24)
    ax.set_ylabel('Time (s)', size=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()

    return diff_max, diff_mean, times


def correlations(xtal, basis, cell, hkl, bloch, cbed, rc):
    """
    Runs simulations with changes in each variables and gives the correlation
    in the changes in each LACBED pattern
    We already have a baseline simulation
    """
    # first check we have enough variables to correlate
    if rc.n_variables < 2:
        raise ValueError(" ## Too few variables to correlate! ##")
        return

    # normalise each LACBED pattern
    # simulation has size [n_thickness, imgX, imgY, n_out]
    mean = cbed.lacbed_sim.mean(axis=(1, 2), keepdims=True)
    std = cbed.lacbed_sim.std(axis=(1, 2), keepdims=True)
    cbed.lacbed_ref = (cbed.lacbed_sim - mean) / std

    detla = {
        20: 0.0005,  # atom coordinate
        21: 0.05,  # occupancy
        22: 0.1,  # isotropic atomic displacement parameters (ADPs)
        23: 0.001,  # anisotropic ADP U11
        24: 0.001,  # anisotropic ADP U22
        25: 0.001,  # anisotropic ADP U33
        26: 0.001,  # anisotropic ADP U12
        27: 0.001,  # anisotropic ADP U23
        28: 0.001,  # anisotropic ADP U13
        30: 0.01,  # lattice parmeter
        31: 0.01,  # lattice parmeter
        32: 0.01,  # lattice parmeter
        33: 0.01,  # lattice parmeter
        34: 0.01,  # lattice parmeter
        35: 0.01,  # lattice parmeter
        40: 0.05,  # convergence angle
        50: 0.02,  # kappa
        51: 0.05,  # valence electrons
    }

    # now go through each variable and make signature images
    for i in range(rc.n_variables):
        t = rc.refined_variable_type[i]
        # set the refinement scale
        delta = detla[rc.refined_variable_type[i]]
        print(f"  Changing {variable_message(t)} by {delta}")
        rc.refined_variable[i] += delta
        # check for validity: Occupancy and ADPs must be >=0
        var, cont = variable_check(rc.refined_variable[i], t)
        if not cont:
            # try going the other way
            rc.refined_variable[i] -= 2*delta
            delta = -delta
            var, cont = variable_check(rc.refined_variable[i], t)
            if not cont:
                raise ValueError(" ## Variable out of range ##")
                return
        update_variables(xtal, basis, rc)
        print_current_var(xtal, basis, rc, i)
        # new simulation
        simulate(xtal, basis, cell, hkl, bloch, cbed, rc)
        print_LACBED(bloch, cbed, rc, 0)
        mean = cbed.lacbed_sim.mean(axis=(1, 2), keepdims=True)
        std = cbed.lacbed_sim.std(axis=(1, 2), keepdims=True)
        # signature images, size [n_variables, n_thickness, imgX, imgY, n_out]
        cbed.lacbed_sig[i] = (cbed.lacbed_sim - mean) / std - cbed.lacbed_ref
        rc.refined_variable[i] -= delta
        update_variables(xtal, basis, rc)


def figure_of_merit(bloch, cbed, rc):
    """
    takes as an input cbed.lacbed_sim, shape [n_thickness, pix_x, pix_y, n_out]
    applies image processing if required
    image processing = 0 -> no Gaussian blur (applied with radius 0)
    image processing = 1 -> Gaussian blur radius defined in felix.inp
    image processing = 2 -> find the best blur radius
    uses zncc/pcc, which work on sets of images both of size [pix_x, pix_y, n]
    currently returns a single figure of merit fom that is mean of zncc's for
    the best thickness.  Could give a more sophisticated analysis..
    """
    # figure of merit - might need a NaN check? size [n_thick, n_out]
    fom_array = np.ones([rc.n_thickness, rc.n_out])
    rc.lacbed_fit_sigma = np.zeros(rc.n_thickness)
    # difference images
    # cbed.lacbed_diff = np.copy(cbed.lacbed_expt)
    # set up plot for blur optimisation
    if rc.plot >= 2 and rc.image_processing == 2:
        fig, ax = plt.subplots(1, 1)
        w_f = 10
        fig.set_size_inches(w_f, w_f)
        ax.set_xlabel('Blur radius', size=24)
        ax.set_ylabel('Figure of merit', size=24)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
    # affine transformation option, once we have a best thickness
    if rc.correlation_type > 2 and rc.iter_count > 1:
        affine(cbed, rc)
    # loop over thicknesses
    lacbed_sobel = np.empty_like(cbed.lacbed_sim)
    for i in range(rc.n_thickness):
        # image processing = 2 -> find the best blur
        if rc.image_processing == 2:
            radii = np.arange(0.2, 2.1, 0.1)  # range of blurs to try
            b_fom = ([])  # mean fom for each blur
            for r in radii:
                blacbed = np.copy(cbed.lacbed_sim[i, :, :, :])
                for j in range(rc.n_out):
                    blacbed[:, :, j] = gaussian_filter(blacbed[:, :, j],
                                                       sigma=r)
                # b_fom.append(np.mean(1.0 - zncc(cbed.lacbed_expt, blacbed)))
                b_fom.append(np.mean(1.0 - pcc(cbed.lacbed_expt, blacbed)))
            if rc.plot > 1:
                plt.plot(radii, b_fom)
            rc.blur_radius = radii[np.argmin(b_fom)]

        # apply the blur
        if rc.image_processing != 0:
            for j in range(rc.n_out):
                cbed.lacbed_sim[i, :, :, j] = gaussian_filter(cbed.lacbed_sim[i, :, :, j],
                                                           sigma=rc.blur_radius)

        # sub-pixel shift for correlation if required
        c_time = time.time()
        if rc.correlation_type> 1:
            for j in range(rc.n_out):
                # we only do this for zncc images that exist
                if np.sum(cbed.lacbed_expt_raw[j]) != 0:
                    a0 = cbed.lacbed_sim[i, :, :, j]
                    b0 = cbed.lacbed_expt_raw[:, :, j]
                    # zero mean normalise the images
                    a = (a0 - np.mean(a0))/np.std(a0)
                    b = (b0 - np.mean(b0))/np.std(b0)
                    # the shift
                    # shift_ = phase_d_xy(a, b)
                    # plt.imshow(a)
                    # plt.show()
                    # plt.imshow(b)
                    # plt.show()
                    shift_ = cc_d_xy(b, a)
                    if rc.write_flag > 0:
                        np.set_printoptions(precision=1, suppress=True)
                        g_string = px.hkl_string(bloch.hkl_indices[bloch.hkl_output[j]])
                        print(f"Pattern {g_string} translated by {shift_} pixels")
                    c = shift(b, shift=shift_, order=3,
                              mode="constant", cval=0)
                    # replace empty experimental pixels with simulation
                    # (which )prevents them from contribution to the zncc)
                    c[c == 0] = a[c == 0]
                    # plt.imshow(c)
                    # plt.show()
                    cbed.lacbed_expt[:, :, j] = c
        d_time = time.time()
        if rc.debug > 0:
            print(f"    Correlation took {d_time-c_time} s")

        # normalised experimental images
        mean = cbed.lacbed_expt.mean(axis=(0, 1), keepdims=True)
        std  = cbed.lacbed_expt.std(axis=(0, 1), keepdims=True)
        cbed.lacbed_expt_norm = (cbed.lacbed_expt - mean) / std

        # affine transformation option without a best thickness
        if rc.correlation_type == 3 and rc.iter_count == 0:
            affine(cbed, rc)

        # figure of merit
        if rc.correlation_type == 0:
            fom_array[i, :] = 1.0 - pcc(cbed.lacbed_expt,
                                        cbed.lacbed_sim[i, :, :, :])
        elif rc.correlation_type < 4:
            fom_array[i, :] = 1.0 - zncc(cbed.lacbed_expt,
                                         cbed.lacbed_sim[i, :, :, :])
        else:  # correlation_type = 4
            for j in range(rc.n_out):
                lacbed_sobel[i, :, :, j] = sobel(cbed.lacbed_sim[i, :, :, j])
            fom_array[i, :] = 1.0 - zncc(sobel(cbed.lacbed_expt),
                                         lacbed_sobel[i, :, :, :])
        # standard deviation of fits for this thickness
        rc.lacbed_fit_sigma[i] = np.std(fom_array[i,:])

        # difference images
        sim = cbed.lacbed_sim[i]
        mean = sim.mean(axis=(0, 1), keepdims=True)
        std  = sim.std(axis=(0, 1), keepdims=True)
        sim_norm = (sim - mean) / std
        cbed.lacbed_diff[i] =  cbed.lacbed_expt_norm - sim_norm
        # for j in range(rc.n_out):
        #     a0 = cbed.lacbed_sim[i, :, :, j]
        #     b0 = cbed.lacbed_expt[:, :, j]
        #     # zero mean normalise the images
        #     a = (a0 - np.mean(a0))/np.std(a0)
        #     b = (b0 - np.mean(b0))/np.std(b0)
        #     cbed.lacbed_diff[i, :, :, j] = a-b

    # print best values
    if rc.image_processing == 2:
        print(f"  Best blur={rc.blur_radius:.1f}")
    if rc.n_thickness > 1:
        # mean figure of merit
        rc.best_t = np.argmin(np.mean(fom_array, axis=1))
        t_nm = 0.1*rc.thickness[rc.best_t]
        # take error in thickness as std dev of individual best t
        best_fom = np.argmin(fom_array, axis=0)  # locations of best FoMs
        t_sigma = np.std(rc.thickness[best_fom])

        print(f"  Best thickness {t_nm:.1f} +/- {0.1*t_sigma:.1f} nm")
        # mean figure of merit
        fom = np.mean(fom_array[rc.best_t])
    else:
        rc.best_t = 0
        fom = np.mean(fom_array[0])

    # plot of blur fit when rc.image_processing == 2
    if rc.plot >= 2 and rc.image_processing == 2:
        plt.show()

    # plot FoM vs thickness for all LACBED patterns
    if rc.plot >= 2 and rc.n_thickness > 1:
        fig, ax = plt.subplots(1, 1)
        w_f = 10
        fig.set_size_inches(1.5*w_f, w_f)
        plt.plot(rc.thickness/10, np.mean(fom_array, axis=1), 'ro', linewidth=2)
        colours = plt.cm.gnuplot(np.linspace(0, 1, rc.n_out))
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
        for i in range(rc.n_out):
            annotation = f"{bloch.hkl_indices[bloch.hkl_output[i], 0]}{bloch.hkl_indices[bloch.hkl_output[i], 1]}{bloch.hkl_indices[bloch.hkl_output[i], 2]}"
            plt.plot(rc.thickness/10, fom_array[:, i],
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


def variable_message(vtype):
    """Map variable type → message string."""
    msg = {
        10: "Ug amplitude",
        11: "Ug phase",
        20: "Atom coordinates",
        21: "Occupancy",
        22: "B_iso",
        23: "U[1,1]",
        24: "U[2,2]",
        25: "U[3,3]",
        26: "U[1,2]",
        27: "U[1,3]",
        28: "U[2,3]",
        30: "Lattice parameter a",
        31: "Lattice parameter b",
        32: "Lattice parameter c",
        33: "Lattice alpha",
        34: "Lattice beta",
        35: "Lattice gamma",
        40: "Convergence angle",
        41: "Accelerating voltage",
        50: "Kappa",
        51: "Pv",
    }
    return msg[vtype]


def variable_check(x, t):
    '''
    x: float, variable being refined
    t: integer, type of variable being refined
    '''
    continue_ = True

    # error check - NaN, inf
    if not np.all(np.isfinite(x)):
        continue_ = False
        x = 0.0
        print("  ***NaN or inf: refined variable set to zero***")
        
    # Occupancy and Atomic displacement parameters
    if int(t/10) == 2 and 0 < np.mod(t, 10) < 6:
        if x < 0:
            x = 0.0
            continue_ = False
            print("  ***Refined variable set to zero***")

    # Occupancy upper limit
    if t == 21:
        if x > 1.0:
            x = 1.0
            continue_ = False
            print("  ***Occupancy set to 1.0***")
        
    return x, continue_


def update_variables(xtal, basis, rc):
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
    51 J = Pv
    50 K = Kappa
    """

    # will tackle this when doing atomic position refinement
    # basis_atom_delta.fill(0)  # Reset atom coordinate uncertainties to zero

    typ = rc.refined_variable_type // 10  # variable type
    sub = rc.refined_variable_type % 10  # variable subtype

    for i in range(rc.n_variables):
        j = rc.atom_refine_flag[i]  # neat
        var = np.copy(rc.refined_variable[i])

        if typ[i] == 0:
            # Structure factor refinement (handled elsewhere)
            variable_check = 1

        elif typ[i] == 2:
            if sub[i] == 0:  # Atomic coordinates
                atom_id = j
                # Update position: r' = r - v*(r.v) + v*current_var
                r_dot_v = np.dot(basis.atom_position[atom_id],
                                 rc.atom_refine_vec[i])
                basis.atom_position[atom_id, :] = np.mod(
                    basis.atom_position[atom_id, :] + rc.atom_refine_vec[i] *
                    (var - r_dot_v), 1)

                # error estimate - needs work
                # Update uncertainty if independent_delta is non-zero
                # if abs(independent_delta[i]) > 1e-10:  # Tiny threshold
                #     basis_atom_delta[atom_id, :] += vector[j - 1, :]
                #                                   * independent_delta[i]

            elif sub[i] == 1:  # Occupancy
                basis.occupancy[j] = var
                # shared occupancy is held in basis.mult_occ
                if basis.mult_occ[j] != 0:
                    # get the indices of atoms on the same site
                    mask = basis.mult_occ == basis.mult_occ[j]
                    mask[j] = False
                    # sum of occupancies not being refined
                    other_occ_sum = basis.occupancy[mask].sum()
                    # scale their occupancies in proportion to existing
                    # minimum allowed occupancy is 1%
                    if other_occ_sum > 0.01:
                        basis.occupancy[mask] *= (1 - var) \
                            / basis.occupancy[mask].sum()
                    else:
                        basis.occupancy[mask] = 0.0

            elif sub[i] == 2:   # Iso ADPs
                if 0 < var:  # must lie in range
                    basis.u_aniso[j, 0, 0] = var / (8 * np.pi**2)
                    basis.u_aniso[j, 1, 1] = var / (8 * np.pi**2)
                    basis.u_aniso[j, 2, 2] = var / (8 * np.pi**2)
                else:
                    basis.u_aniso[j, :, :] = 0.0
            elif sub[i] == 3:  # u[1,1]
                basis.u_aniso[j, 0, 0] = var
            elif sub[i] == 4:  # u[1,1]
                basis.u_aniso[j, 1, 1] = var
                # basis.u_aniso[j, 2, 2] = var
            elif sub[i] == 5:  # u[2,2]
                # basis.u_aniso[j, 1, 1] = var
                basis.u_aniso[j, 2, 2] = var
            elif sub[i] == 6:  # u[1,2]
                basis.u_aniso[j, 0, 1] = var
                basis.u_aniso[j, 1, 0] = var
            elif sub[i] == 7:  # u[1,3]
                basis.u_aniso[j, 0, 2] = var
                basis.u_aniso[j, 2, 0] = var
            elif sub[i] == 8:  # u[2,3]
                basis.u_aniso[j, 1, 2] = var
                basis.u_aniso[j, 2, 1] = var

        elif typ[i] == 3:
            # Lattice parameters a, b, c
            if sub[i] == 0:
                xtal.cell_a = xtal.cell_b = xtal.cell_c = var
            elif sub[i] == 1:
                xtal.cell_b = var
            elif sub[i] == 2:
                xtal.cell_c = var
            elif sub[i] == 3:
                xtal.cell_alpha = var
            elif sub[i] == 4:
                xtal.cell_beta = var
            elif sub[i] == 5:
                xtal.cell_gamma = var

        elif typ[i] == 4:
            if sub[i] == 0:  # Convergence angle
                rc.convergence_angle = var
            elif sub[i] == 1:  # Accelerating voltage
                rc.accelerating_voltage_kv = var

        elif typ[i] == 5:
            if sub[i] == 0:  # kappa
                if 0.7 < var < 1.3:  # must lie in a reasonable range
                    basis.kappa[j] = var*1.0
                else:
                    basis.kappa[j] = np.clip(var, 0.7, 1.3)
            elif sub[i] == 1:  # Pv
                basis.pv[j] = var*1.0


def print_montage(bloch, cbed, rc, images, image_type, j):
    '''
    images[wid, wid, n] = array of n images each of size [wid, wid]
    j = thickness index in simulated pattern array
    j = variable index in signature array
    '''
    if image_type == 0:
        lut = 'pink'
    elif image_type == 1:
        lut = 'grey'
    n = images.shape[2]
    w = int(np.ceil(np.sqrt(n)))
    h = int(np.ceil(n/w))
    fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
    text_effect = withStroke(linewidth=3, foreground='black')
    axes = axes.flatten()
    for i in range(n):
        img = images[:, :, i]
        # difference or LACBED pattern
        if image_type >= 2 and img.min() < 0:
            cmap = LinearSegmentedColormap.from_list(
                "two_color_black_center",
                [(0.0, "c"), (0.5, "k"), (1.0, "orange")])
            # two colour look up table
            norm = TwoSlopeNorm(vmin=img.min(), vcenter=0.0, vmax=img.max())
            axes[i].imshow(img, cmap=cmap, norm=norm)
        else:
            axes[i].imshow(img, cmap=lut)

        axes[i].axis('off')
        annotation = f"{bloch.hkl_indices[bloch.hkl_output[i], 0]}{bloch.hkl_indices[bloch.hkl_output[i], 1]}{bloch.hkl_indices[bloch.hkl_output[i], 2]}"
        axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                         size=30, color='w', path_effects=[text_effect])
    for i in range(n, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    if image_type !=1:  # don't put a thickness on experimental images
        annotation = f"{rc.thickness[j]/10:.0f} nm"
        plt.annotate(annotation, xy=(0.105, 0.96), xycoords='figure fraction',
                     size=30, color='c', path_effects=[text_effect])
    if image_type ==3:  # signature image
        annotation = variable_message(rc.refined_variable_type[j])
        plt.annotate(annotation, xy=(0.105, 0.05), xycoords='figure fraction',
                     size=30, color='w', path_effects=[text_effect])
    plt.show()


def print_LACBED(bloch, cbed, rc, image_type):
    '''
    Plots all LACBED patterns in a montage
    image_type options 0=sim, 1=expt, 2=difference, 3=signature
    '''
    if image_type == 0:  # simulation output
        # only print all thicknesses for the first simulation
        if rc.iter_count == 1:
            for j in range(rc.n_thickness):
                out_image = cbed.lacbed_sim[j, :, :, :]
                print_montage(bloch, cbed, rc, out_image, image_type, j)
        else:  # just best thickness
            out_image = cbed.lacbed_sim[rc.best_t, :, :, :]
            print_montage(bloch, cbed, rc, out_image, image_type, rc.best_t)
    elif image_type == 3:  # signature output
        for j in range(rc.n_variables):
            out_image = cbed.lacbed_sig[j, :, :, :]
            print_montage(bloch, cbed, rc, out_image, image_type, j)
    else:
        # All other image types
        image_map = {
            1: (cbed.lacbed_expt, 0),
            2: (cbed.lacbed_diff[rc.best_t], rc.best_t),
        }
    
        out_image, thickness = image_map[image_type]
        print_montage(bloch, cbed, rc, out_image, image_type, thickness)


def print_LACBED_pattern(i, j, cbed, bloch):
    # Prints an individual LACBED pattern
    # i = 0  # index of the pattern to plot
    # j = 0  # index of the thickness to plot
    fig, ax = plt.subplots(1, 1)
    text_effect = withStroke(linewidth=3, foreground='black')
    ax.imshow(cbed.lacbed_sim[j, :, :, i], cmap='pink')
    ax.axis('off')
    annotation = f"{bloch.hkl_indices[bloch.hkl_output[i], 0]}{bloch.hkl_indices[bloch.hkl_output[i], 1]}{bloch.hkl_indices[bloch.hkl_output[i], 2]}"
    ax.annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                     size=30, color='w', path_effects=[text_effect])


def save_LACBED(xtal, bloch, cbed, rc):
    '''
    Saves all LACBED patterns in .npy and .png format
    '''
    print(os.getcwd())
    if not os.path.isdir(xtal.chemical_formula):
        os.mkdir(xtal.chemical_formula)
    os.chdir(xtal.chemical_formula)
    j = 0
    # j = v.best_t
    for j in range(rc.n_thickness):
        t = int(rc.thickness[j]/10)
        t_ = f"{t}nm"
        if not os.path.isdir(t_):
            os.mkdir(t_)
        os.chdir(t_)
        for i in range(cbed.lacbed_sim.shape[3]):
            signed_str = "".join(f"{x:+d}" for x in bloch.hkl_indices[bloch.hkl_output[i], :])
            fname = f"{xtal.chemical_formula}_{signed_str}_{t}nm.bin"
            cbed.lacbed_sim[j, :, :, i].tofile(fname)
            # out_image[:, :, i].tofile(fname)
            fname = f"{xtal.chemical_formula}_{signed_str}_{t}nm.png"
            plt.imsave(fname, cbed.lacbed_sim[j, :, :, i], cmap='gray')
        os.chdir("..")
    os.chdir("..")


def print_current_var(xtal, basis, rc, i):
    # prints the variable being refined
    typ = rc.refined_variable_type[i]  # variable type & subtype
    var = rc.refined_variable[i]  # variable
    if rc.refined_variable_sigma is not None:
        sigma = rc.refined_variable_sigma[i]  # error estimate
    else:
        sigma = 0.0
    atom_id = rc.atom_refine_flag[i]
    label = basis.atom_label[atom_id]
    # Z = basis.atomic_number[atom_id]

    if sigma > 1e-06:
        # dictionary of format strings
        formats = {
            10: ("Current Ug amplitude", "{:.3f}"),
            11: ("Current Ug phase", "{:.3f}"),
            21: (f" Atom {label}: Current occupancy", "{:.3f} +/-", "{:.3f}"),
            22: (f" Atom {label}: Current B_iso", "{:.3f} +/-", "{:.3f}"),
            23: (f" Atom {label}: Current U[1,1]", "{:.5f} +/-", "{:.5f}"),
            24: (f" Atom {label}: Current U[2,2]", "{:.5f} +/-", "{:.5f}"),
            25: (f" Atom {label}: Current U[3,3]", "{:.5f} +/-", "{:.5f}"),
            26: (f" Atom {label}: Current U[1,2]", "{:.5f} +/-", "{:.5f}"),
            27: (f" Atom {label}: Current U[1,3]", "{:.5f} +/-", "{:.5f}"),
            28: (f" Atom {label}: Current U[2,3]", "{:.5f} +/-", "{:.5f}"),
            30: ("Current lattice parameter a", "{:.4f} +/-", "{:.4f}"),
            31: ("Current lattice parameter b", "{:.4f} +/-", "{:.4f}"),
            32: ("Current lattice parameter c", "{:.4f} +/-", "{:.4f}"),
            33: ("Current lattice alpha", "{:.4f} +/-", "{:.4f}"),
            34: ("Current lattice beta", "{:.4f} +/-", "{:.4f}"),
            35: ("Current lattice gamma", "{:.4f} +/-", "{:.4f}"),
            40: ("Current convergence angle", "{:.3f} +/-", "{:.3f} Å^-1"),
            41: ("Current accelerating voltage", "{:.1f} +/-", "{:.1f} kV"),
            50: (f" Atom {label}: Kappa", "{:.3f} +/-", "{:.3f}"),
            51: (f" Atom {label}: Pv", "{:.4f} +/-", "{:.4f}")
                }
        if typ == 20:  # atomic coords
            with np.printoptions(formatter={'float': lambda x: f"{x:.5f}"}):
                print(f"  Atom {atom_id}: {label}  {basis.atom_position[atom_id, :]} +/- {sigma:.5f}")
        elif typ in formats:
            label, fmt, fmt_s = formats[typ]
            print(f"  {label} {fmt.format(var)} {fmt_s.format(sigma)}")
    else:
        formats = {
            10: ("Current Ug amplitude", "{:.3f}"),
            11: ("Current Ug phase", "{:.3f}"),
            21: (f" Atom {label}: Current occupancy", "{:.3f}"),
            22: (f" Atom {label}: Current B_iso", "{:.3f}"),
            23: (f" Atom {label}: Current U[1,1]", "{:.5f}"),
            24: (f" Atom {label}: Current U[2,2]", "{:.5f}"),
            25: (f" Atom {label}: Current U[3,3]", "{:.5f}"),
            26: (f" Atom {label}: Current U[1,2]", "{:.5f}"),
            27: (f" Atom {label}: Current U[1,3]", "{:.5f}"),
            28: (f" Atom {label}: Current U[2,3]", "{:.5f}"),
            30: ("Current lattice parameter a", "{:.4f}"),
            31: ("Current lattice parameter b", "{:.4f}+"),
            32: ("Current lattice parameter c", "{:.4f}"),
            33: ("Current lattice alpha", "{:.4f}"),
            34: ("Current lattice beta", "{:.4f}"),
            35: ("Current lattice gamma", "{:.4f}"),
            40: ("Current convergence angle", "{:.3f} Å^-1"),
            41: ("Current accelerating voltage", "{:.1f} kV"),
            50: (f" Atom {label}: Kappa", "{:.3f}"),
            51: (f" Atom {label}: Pv", "{:.4f}")
                }
        if typ == 20:  # atomic coords
            with np.printoptions(formatter={'float': lambda x: f"{x:.5f}"}):
                print(f"  Atom {atom_id}: {label}  {basis.atom_position[atom_id, :]}")
        elif typ in formats:
            label, fmt = formats[typ]
            print(f"  {label} {fmt.format(var)}")


def sim_fom(xtal, basis, cell, hkl, bloch, cbed, rc, i):
    '''
    wraps multiple subroutine calls into a single line: update variables
    simulate and figure of merit
    input i = index of variable to be refined, for single variables
    i = -1 for multiple variables

    The 'signature' of variable i is the change it produces in each LACBED
    pattern.  We give the zero-mean normalised difference here
    '''
    update_variables(xtal, basis, rc)
    print_current_var(xtal, basis, rc, i)

    # update reference images
    cbed.lacbed_ref = np.copy(cbed.lacbed_sim[rc.best_t, :, :, :])
    # normalise each one
    mean = cbed.lacbed_ref.mean(axis=(0, 1), keepdims=True)
    std = cbed.lacbed_ref.std(axis=(0, 1), keepdims=True)
    cbed.lacbed_ref = (cbed.lacbed_ref - mean) / std

    # simulate
    simulate(xtal, basis, cell, hkl, bloch, cbed, rc)

    # calculate signature images
    sim = np.copy(cbed.lacbed_sim[rc.best_t, :, :, :])
    mean = sim.mean(axis=(0, 1), keepdims=True)
    std = sim.std(axis=(0, 1), keepdims=True)
    sim_norm = (sim - mean) / std
    if i >= 0:  # get a signature image if it's a single variable refinement
        cbed.lacbed_sig[i, rc.best_t] = sim_norm - cbed.lacbed_ref
        print_LACBED(bloch, cbed, rc, 3)

        # *** output for development ***
        img = cbed.lacbed_sig[0, rc.best_t, :, :, 0]
        cmap = LinearSegmentedColormap.from_list(
            "two_color_black_center",
            [(0.0, "c"), (0.5, "k"), (1.0, "orange")])
        # two colour look up table
        norm = TwoSlopeNorm(vmin=img.min(), vcenter=0.0, vmax=img.max())
        plt.imshow(img, cmap=cmap, norm=norm)
        plt.axis('off')
        plt.show()
        
    # figure of merit
    fom = figure_of_merit(bloch, cbed, rc)
    rc.fit_log.append(fom)
    rc.param_log.append(np.copy(rc.refined_variable))
    print(f"  Figure of merit {100*fom:.3f}% (previous best {100*rc.best_fit:.3f}%)")

    return fom


def refine_single_variable(xtal, basis, cell, hkl, bloch, cbed, rc, i):
    '''
    Does 3-point refinement for a single variable and if no minimum is found
    retuns the step size for a subsequent multidimensional refinement
    i: integer, index of variable being refined
    Uses the whole variable space as it is passed on to the simulation:
    rc.best_fit: float, best figure of merit
    rc.best_var: float array, the refined variables with best FoM
    rc.refined_variable: float array of variables being refined
    rc.refined_variable_type: integer array, type of variable being refined
    rc.refinement_scale: float, step to change the variable (from felix.inp)
    rc.next_var: float array of predicted variable values
    ----------------
    Updates:
        rc.best_fit, rc.best_var if necessary
        rc.next_var as an output
    Returns:
        dydx_i: the gradient of this variable
    '''
    # uncertainty in figure of merit  ***hack at the moment to fixed value***
    dy = 0.003
    r3_var = np.zeros(3)  # for parabolic minimum
    r3_fom = np.zeros(3)
    # Check if ADP is negative, skip if so.
    # NB u12,u13,u23 can be -ve
    if (21 < rc.refined_variable_type[i] < 26 and
            rc.refined_variable[i] < 1e-10):
        dydx_i = 0.0
    else:
        # middle point is the previous best simulation
        r3_var[1] = rc.best_var[i]*1.0
        r3_fom[1] = rc.best_fit*1.0
        print(f"Finding gradient, variable {i+1} of {rc.n_variables}")
        print(f"Changing {variable_message(rc.refined_variable_type[i])}")
        # print_current_var(rc, i)

        # delta is a small change in the current variable
        # which is either refinement_scale for atomic coordinates and
        # refinement_scale*variable for everything else
        delta = abs(rc.refinement_scale * rc.refined_variable[i])
        if rc.refined_variable_type[i] == 20:
            delta = abs(rc.refinement_scale)
        # Three-point gradient measurement, starting with plus
        rc.refined_variable[i] += delta
        # simulate and get figure of merit
        fom = sim_fom(xtal, basis, cell, hkl, bloch, cbed, rc, i)
        r3_var[2] = rc.refined_variable[i]*1.0
        r3_fom[2] = fom*1.0
        print("-1-----------------------------")  # " {r3_var},{r3_fom}")
        # update best fit
        if (fom < rc.best_fit):
            rc.best_fit = fom*1.0
            rc.best_var = np.copy(rc.refined_variable)

        # keep going or turn round?
        if r3_fom[2] < r3_fom[1]:  # keep going
            rc.refined_variable[i] += np.exp(0.5) * delta
        else:  # turn round
            delta = - delta
            rc.refined_variable[i] += np.exp(0.75) * delta
        # simulate and get figure of merit
        fom = sim_fom(xtal, basis, cell, hkl, bloch, cbed, rc, i)
        r3_var[0] = rc.refined_variable[i]*1.0
        r3_fom[0] = fom*1.0
        print("-2-----------------------------")  # " {r3_var},{r3_fom}")
        # update best fit
        if (fom < rc.best_fit):
            rc.best_fit = fom*1.0
            rc.best_var = np.copy(rc.refined_variable)

        # test to see if the variable has an effect
        if np.max(r3_fom)-np.min(r3_fom) < 1e-5:  # no effect on FoM
            exclude = True
            print(f"  Low effect, fixed at {rc.best_var[i]}")
        else:
            rc.next_var[i], exclude, dx = px.convex(r3_var, r3_fom, dy)
        # predict the next point as a minimum or a step onwards
        # rc.next_var gets passed on to the multidimensional refinement
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


def refine(xtal, basis, cell, hkl, bloch, cbed, rc):
    # figure of merit - includes determining/applying shifts
    cbed.lacbed_expt = np.copy(cbed.lacbed_expt_raw)  # with shifts
    fom = figure_of_merit(bloch, cbed, rc)
    print(f"  Figure of merit {100*fom:.2f}%")

    # Initialise variables for refinement
    rc.best_fit = fom*1.0
    rc.last_fit = fom*1.0

    # for a plot
    rc.fit_log = ([rc.last_fit*1.0])
    rc.param_log = ([np.copy(rc.refined_variable)])

    # Refinement loop
    df = 1.0
    rc.refinement_scale *= 2.0
    while df >= rc.precision and rc.refinement_scale >= abs(rc.precision):
        # rc.refined_variable is the working array of variables
        # best_var is the best array of variables during this refinement cycle
        rc.best_var = np.copy(rc.refined_variable)
        # next_var is the predicted next (best) point
        rc.next_var = np.copy(rc.refined_variable)
        # reduce refinement scale for next round
        # rc.refinement_scale *= (1 - 1 / (1 + rc.n_variables))
        rc.refinement_scale *= 0.5
        print(f"Step size {rc.refinement_scale:g}")

        if rc.refine_method == 0:
            print("Gradient descent, one parameter at a time")
            # dydx is a vector along the gradient in n-dimensional space
            dydx = np.zeros(rc.n_variables)
            # random order for 3 or more variables
            indices = np.arange(rc.n_variables)
            if rc.n_variables > 2 or rc.iter_count == 0:
                np.random.shuffle(indices)
            for i in indices:
                dydx[i] = 1.0
                dydx = refine_multi_variable(xtal, basis, cell, hkl,
                                                 bloch, cbed, rc, dydx)
        elif rc.refine_method == 1:
            print("Multiparameter refinement, finding parameter gradients")
            # =========== step 1: individual variable minimisation
            # if all variables have been refined, reset
            if np.sum(np.abs(dydx)) < 1e-10:
                dydx = np.ones(rc.n_variables)
            # Go through the variables looking at three points in the hope
            # of capturing a minimum - if there is one, we take it and remove
            # that variable from multidimensional refinement, dydx[i] = 0.
            # Otherwise dydx[i] is the gradient for that variable.
            # We also get a predicted best starting point
            # for gradient descent, rc.next_var
            for i in range(rc.n_variables):
                # Skip variables already optimized
                if abs(dydx[i]) < 1e-10:
                    dydx[i] = 0.0
                    continue
                dydx[i] = refine_single_variable(xtal, basis, cell, hkl,
                                                     bloch, cbed, rc, i)

            # all variables have updated/predicted so do a final simulation
            # if it's better, update rc.best_fit and rc.best_var accordingly
            if np.count_nonzero(dydx) == 0:
                print("Closing simulation for this cycle")
                rc.refined_variable = np.copy(rc.next_var)
                fom = sim_fom(xtal, basis, hkl, bloch, cbed, rc, i)
                if (fom < rc.best_fit):
                    rc.best_fit = fom*1.0
                    rc.best_var = np.copy(rc.refined_variable)
            print("Vector gradient descent")
            # ===========step 2: vector descent
            # Downhill minimisation until we eliminate all variables
            while np.sum(np.abs(dydx)) > 1e-10:
                # the returned dydx will have an extra zero!
                dydx = refine_multi_variable(xtal, basis, cell, hkl,
                                                 bloch, cbed, rc, dydx, False)
        else:
            raise ValueError("No valid refine method (0,1) in felix.inp")
        if rc.plot > 0:
            plot_progress(rc)
            print_LACBED(bloch, cbed, rc, 0)

        # Update for next iteration
        df = rc.last_fit - rc.best_fit

        rc.last_fit = np.copy(rc.best_fit)
        rc.refined_variable = np.copy(rc.best_var)
        if rc.precision > 0:
            print(f"Improvement in fit {100*df:.2f}%, will stop at {100*rc.precision:.2f}%")
        else:
            print(f"Improvement in fit {100*df:.2f}%, will stop after step size < {abs(rc.precision)}")
        print("-------------------------------")
    print(f"Refinement complete after {rc.iter_count} simulations.")
    for i in range(rc.n_variables):
        print_current_var(xtal, basis, rc, i)


def refine_multi_variable(xtal, basis, cell, hkl, bloch, cbed,
                          rc, dydx, single=True):
    '''
    multidimensional refinement
    dydx: float array of gradients, generated in refine_single_variable
    Uses the whole variable space rc, only:
    rc.refined_variable: array of variables to refine size [n_var]
    rc.refined_variable_type: what kind of variable (see felixrefine)
    rc.best_fit: best figure of merit so far
    rc.best_var: array of variables that gives best fit
    rc.refinement_scale: size of change in variable to obtain gradient

    dydx = array of gradients, size [n_var]
    '''
    # uncertainty in figure of merit  ***hack at the moment to fixed value***
    dy = 0.00003  # 0.003%

    # starting point is the current best set of variables
    rc.last_fit = 1.0*rc.best_fit

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
    t = rc.refined_variable_type[j]
    print(f"  Principal variable: changing {variable_message(t)}")

    # Check the gradient vector magnitude and initialize vector descent
    if not single:
        print(f"    Extrapolation, should be better than {100*rc.best_fit:.2f}%")
        # initial trial uses the predicted best set of variables
        rc.refined_variable = 1.0*rc.next_var
        # simulate and get figure of merit
        # check for validity: Occupancy and ADPs must be >=0
        rc.refined_variable[j], cont = variable_check(rc.refined_variable[j],
                                                      t)
        fom = sim_fom(xtal, basis, cell, hkl, bloch, cbed, rc, j)
        if rc.plot > 1:
            print_LACBED(bloch, cbed, rc, 0)
        # is it actually any better
        if fom < rc.last_fit:
            rc.best_fit = fom*1.0
            rc.best_var = np.copy(rc.refined_variable)
            print("Point 1 of 3: extrapolated")  # yes, use it
        else:
            print("Point 1 of 3: previous best")  # no, use the best
        rc.refined_variable = np.copy(rc.best_var)
        print_LACBED(bloch, cbed, rc, 0)
        if rc.plot == 3:  # also do difference image
            print_LACBED(bloch, cbed, rc, 2)

        print(f"-a-----------------------------{rc.iter_count}")  # "{r3_var},{r3_fom}")

    # First point: incoming best simulation
    r3_var = np.zeros(3)
    r3_fom = np.zeros(3)
    r3_var[0] = 1.0*rc.best_var[j]  # using principal variable
    r3_fom[0] = 1.0*rc.best_fit
    # variable and fit arrays for error estimate
    # V = np.zeros(3)
    # F = np.zeros(3)

    # set the refinement scale
    delta = np.random.choice([-1, 1]) * dydx * rc.refinement_scale
    if rc.refined_variable_type[j] == 20:  # atom coordinates
        # reduce scale by an order of magnitude to allow it to be refined
        # alongside other parameters like Biso
        # *** should probably do the same for Uij refinement ***
        delta *= 0.1

    # Second point
    print("Refining, point 2 of 3")
    # Change the array of variables by a small amount
    rc.refined_variable += delta  # point 2
    # check for validity: Occupancy and ADPs must be >=0
    rc.refined_variable[j], cont = variable_check(rc.refined_variable[j], t)
    # simulate and get figure of merit
    fom = sim_fom(xtal, basis, cell, hkl, bloch, cbed, rc, j)
    improvement = rc.best_fit - fom
    if fom < rc.best_fit:
        rc.best_fit = fom*1.0
        rc.best_var = np.copy(rc.refined_variable)
    if rc.plot > 1:
        print_LACBED(bloch, cbed, rc, 0)
    # check for no effect or parameter out of range
    if abs(improvement) < 0.1*abs(rc.precision):
        # we leave best_var unchanged and go on to the next
        print(f"    Improvement {abs(improvement)} is smaller than {0.1*abs(rc.precision)}")
        print(f"-next=========================={rc.iter_count}")
        dydx[j] = 0.0
        return dydx
    elif cont is False:
        print("    Parameter out of range")
        print(f"-next=========================={rc.iter_count}")
        dydx[j] = 0.0
        return dydx
    else:
        r3_var[1] = 1.0*rc.refined_variable[j]
        r3_fom[1] = 1.0*fom
        # with np.printoptions(formatter={'float': lambda x: f"{x:.4f}"}):
        print(f"-b-----------------------------{rc.iter_count}")  # {r3_var},{r3_fom}")

    # Third point
    print("Refining, point 3 of 3")
    if r3_fom[1] > r3_fom[0]:  # if second point is worse
        # Go in the opposite direction
        rc.refined_variable -= np.exp(0.8)*delta
    else:  # keep going
        rc.refined_variable += np.exp(0.4)*delta
    rc.refined_variable[j], cont = variable_check(rc.refined_variable[j], t)
    fom = sim_fom(xtal, basis, cell, hkl, bloch, cbed, rc, j)
    if fom < rc.best_fit:
        rc.best_fit = fom*1.0
        rc.best_var = np.copy(rc.refined_variable)
    if rc.plot > 1:
        print_LACBED(bloch, cbed, rc, 0)
    r3_var[2] = 1.0*rc.refined_variable[j]
    r3_fom[2] = 1.0*fom
    # if cont is False:
    #     print(f"-next=========================={rc.iter_count}")
    #     dydx[j] = 0.0
    #     return dydx
    # with np.printoptions(formatter={'float': lambda x: f"{x:.4f}"}):
    print(f"-c-----------------------------{rc.iter_count}")  # {r3_var},{r3_fom}")


    # We continue downhill until we get a predicted minnymum
    minny = False
    dx = 0
    while minny is False:
        # predict the next point as a minimum or a step on
        next_x, minny, dx = px.convex(r3_var, r3_fom, dy)
        # version that scales the whole vector
        # last_x = 1.0*rc.refined_variable[j]
        # rc.refined_variable[j] *= next_x/last_x
        # version that just changes one variable 
        rc.refined_variable[j] = next_x
        rc.refined_variable[j], cont = variable_check(rc.refined_variable[j], t)
        fom = sim_fom(xtal, basis, cell, hkl, bloch, cbed, rc, j)
        improvement = rc.best_fit - fom
        if fom < rc.best_fit:
            rc.best_fit = fom*1.0
            rc.best_var = np.copy(rc.refined_variable)
        if rc.plot > 1:
            print_LACBED(bloch, cbed, rc, 0)
        # with np.printoptions(formatter={'float': lambda x: f"{x:.4f}"}):
        print(f"-.-----------------------------{rc.iter_count}")  # {r3_var}: {r3_fom}")
        if (improvement > 0.1*abs(rc.precision)) and cont is True:  # it's better, keep going
            # replace worst point with this one
            i = np.argmax(r3_fom)
            r3_var[i] = 1.0*rc.refined_variable[j]
            r3_fom[i] = 1.0*fom
        else:  # we're done
            minny = True
    # we have taken the principal variable to a minimum
    rc.refined_variable = np.copy(rc.best_var)
    dydx[j] = 0.0
    # Error estimate
    rc.refined_variable_sigma[j] = dx
    print(f"    ====Refined variable {j}====")
    print_LACBED(bloch, cbed, rc, 0)
    if rc.plot == 3:  # also do difference image
        print_LACBED(bloch, cbed, rc, 2)

    return dydx


def plot_progress(rc):
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(1.5*w_f, w_f)
    plt.plot(rc.fit_log)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax.set_xlabel('Iteration', size=24)
    ax.set_ylabel('Figure of merit', size=24)
    plt.show()


def plot_charge_density(xtal, basis, rc, i):
    # plots radial charge density of atom i in the basis
    r = np.linspace(1e-6, xtal.r_max, xtal.n_points)
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    # charge densities
    cd_core = basis.core[i, :] * r**2
    cd_valence = basis.valence[i, :] * r**2
    cd_total = cd_core + cd_valence
    plt.plot(r, cd_core, label='core')
    plt.plot(r, cd_valence, label='valence')
    plt.plot(r, cd_total, label='total')
    ax.set_xlim(left=1e-02)
    ax.set_ylim(bottom=1e-02)
    ax.set_xlabel(r'$r$, Å', size=24)
    ax.set_ylabel(r'Charge density, electrons/Å$^3$', size=24)
    ax.legend(loc='best', fontsize=22)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    if rc.scatter_factor_method == 4:
        tit = f"Radial charge density for {basis.atom_label[i]} (Coppens)"
    else:
        tit = f"Radial charge density for {basis.atom_label[i]} (Bunge)"
    plt.title(tit, fontsize=24)
    annotation = f"Kappa = {basis.kappa[i]:.2f}"
    plt.annotate(annotation, xy=(0.17, 0.5), xycoords='figure fraction',
                 size=22)
    # annotation = f"Pv = {basis.pv[i]:.2f}"
    # plt.annotate(annotation, xy=(0.17, 0.45), xycoords='figure fraction',
    #              size=22)
    plt.show()


def plot_f_e(basis, rc, s, f_kappa, f_k, i):
    # plot the scattering factor only when it's being refined
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    smax = 300
    plt.plot(s[1:smax], f_kappa[1:smax], label='$f_e$')
    # plt.plot(s[:smax], f_x[:smax], label='$f_X$')
    plt.plot(s[1:smax], f_k[1:smax], linestyle='-.', label='$f_e(0)$')
    # plt.plot(s[:smax], f_xx[:smax], linestyle='-.', label='$f_X(e)$')
    # plt.yscale('log')
    ax.set_ylim(bottom=0)
    # ax.set_xlim(left=0)
    # ax.set_ylim(top=10)
    ax.set_xlabel(r'$s$ (Å$^{-1}$)', size=24)
    ax.set_ylabel(r'$f_e$', size=24)
    ax.legend(loc='best', fontsize=22)
    # plt.xscale('log')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True, which='both')
    tit = f"Electron scattering factor for atom {basis.atom_label[i]}"
    plt.title(tit, fontsize=24)
    annotation = f"Kappa = {basis.kappa[i]:.2f}"
    plt.annotate(annotation, xy=(0.15, 0.2), xycoords='figure fraction',
                 size=22)
    annotation = f"Pv = {basis.pv[i]:.2f}"
    plt.annotate(annotation, xy=(0.15, 0.15), xycoords='figure fraction',
                 size=22)
    plt.show()
    return


def plot_f_x(s, f_x_v, f_x_c, kappa, pv, name):
    """
    Utility subroutine to plot x-ray scattering factors
    """
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    plt.plot(s, f_x_v, label='valence')
    if f_x_c is not None:
        plt.plot(s, f_x_c, linestyle='-.', label='core')
        plt.plot(s, f_x_v+f_x_c, linestyle=':', label='total')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True, which='both')
    tit = f"X-ray scattering factor for atom {name}"
    plt.title(tit, fontsize=24)
    ax.set_xlabel(r'$s$ (Å$^{-1}$)', size=24)
    ax.set_ylabel('$f_x$', size=24)
    ax.set_xlim(left=0)
    ax.legend(loc='best', fontsize=22)
    # plt.xscale('log')
    annotation = f"Kappa = {kappa:.2f}"
    plt.annotate(annotation, xy=(0.6, 0.8), xycoords='figure fraction',
                 size=22)
    annotation = f"Pv = {pv:.2f}"
    plt.annotate(annotation, xy=(0.6, 0.75), xycoords='figure fraction',
                 size=22)
    plt.show()


def plot_f_g(xtal, basis, rc, bloch, j=0):
    """
    Utility subroutine to plot electron scattering factors
    """
    Z = basis.atomic_number[j]

    g = np.linspace(1, 10, 200)
    # g = bloch.uniq_gmag
    f_g = px.f_kappa(xtal, basis, rc, g, j).ravel()
    f_g_k = px.f_kirkland(Z, g).ravel()
    f0_k = f_g_k[0]
    f0 = f_g[0]

    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)

    plt.plot(g, f_g, linestyle='-', label='Kappa')
    plt.plot(g, f_g_k, linestyle='-', label='Kirkland')
    # f_g = px.f_lobato(Z, g).ravel()
    # plt.plot(g, f_g, linestyle='-.', label='Lobato')
    # f_g = px.f_peng(Z, g).ravel()
    # plt.plot(g, f_g, linestyle='--', label='Peng')
    # f_g = px.f_doyle_turner(Z, g).ravel()
    # plt.plot(g, f_g, linestyle=':', label='Doyle & Turner')

    ax.set_xlabel('$g$, A$^{-1}$', size=24)
    ax.set_ylabel('$f_g$', size=24)
    ax.legend(loc='best', fontsize=12)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.show()

    diff = (f_g - f_g_k)  #/f_g_k
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)

    plt.plot(g, diff, linestyle='-', label='difference')
    ax.set_xlabel('$g$, A$^{-1}$', size=24)
    ax.set_ylabel('$f_g$', size=24)
    ax.legend(loc='best', fontsize=12)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid()
    plt.show()

    print(f"kappa/kirkland = {f0/f0_k}")

    return


def plot_parameter(rc, i):
    # plot of parameter vs fit
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    x = np.array(rc.param_log).ravel()
    y = np.array(rc.fit_log).ravel()
    idx = np.argsort(x)
    xs = x[idx]
    ys = y[idx]
    plt.plot(xs, ys)
    plt.scatter(xs, ys, linewidth=1, s=100)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    x_lab = variable_message(rc.refined_variable_type[i])
    ax.set_xlabel(x_lab, size=24)
    ax.set_ylabel('Figure of merit', size=24)
    plt.grid(True)
    plt.show()
