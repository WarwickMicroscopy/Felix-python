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
from skimage.filters import sobel, median
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
# a small number
eps = 1e-10


# =============================================================================
# Pixel-parallel helpers (process based, cross-platform)
# =============================================================================
_PX_BLOCH = None
_PX_RC = None
_PX_NOUT = None


def _init_pixel_worker(bloch, rc, n_out):
    """
    Worker initializer for pixel-parallel Bloch wave calculations.
    Each process gets its own local copy of objects, so mutable state is isolated.
    """
    global _PX_BLOCH, _PX_RC, _PX_NOUT
    _PX_BLOCH = bloch
    _PX_RC = rc
    _PX_NOUT = n_out


def _compute_pixel_task(pixel):
    """
    Compute one pixel of the Bloch-wave simulation.
    Returns (pix_x, pix_y, intensity_slice) where intensity_slice has
    shape [n_thickness, n_out].
    """
    pix_x, pix_y = pixel
    bloch = _PX_BLOCH
    rc = _PX_RC

    bloch.s_g_pix = np.squeeze(bloch.s_g[pix_x, pix_y, :])
    bloch.k_dot_n_pix = bloch.k_dot_n[pix_x, pix_y]

    px.wave_functions(bloch, rc)
    intensity = np.abs(bloch.wave_function) ** 2
    return pix_x, pix_y, intensity[:, :_PX_NOUT]


def simulate(xtal, basis, cell, hkl, bloch, cbed, rc):

    typ = rc.refined_variable_type // 10  # array of variable types

    # some setup calculations
    bloch.electron_velocity = (np.sqrt(e*rc.accelerating_voltage_kv*1000*
                                       (e*rc.accelerating_voltage_kv*1000
                                        + 2*m_e*c*c))
                               / (m_e*c + e*rc.accelerating_voltage_kv*1000/c))

    bloch.relativistic_correction = 1.0 / np.sqrt(1.0
                                                  - (bloch.electron_velocity/c)**2)

    # wave vector k in m^-1
    bloch.big_k = bloch.relativistic_correction*m_e*bloch.electron_velocity / h
    # and in 1/A
    bloch.big_k_mag = np.linalg.norm(bloch.big_k)*angstrom

    # output hkl values [hkl indices, h, k, l]
    # if no_of_ugs >0 then first no_of_ugs hkl values from hkl.input_hkls are used
    # if no_of_ugs =0 then output hkl values from hkl.input_hkls where
a>0 are used
    if rc.no_of_ugs > 0:
        no_of_ugs = rc.no_of_ugs
    else:
        no_of_ugs = hkl.n_hkls

    if no_of_ugs > hkl.n_hkls:
        no_of_ugs = hkl.n_hkls
        print("n_output_reflexions reduced to", no_of_ugs)

    bloch.hkl_output = np.zeros([no_of_ugs+1, 3], dtype='int')

    # add 000 beam
    bloch.hkl_output[0, :] = [0, 0, 0]
    n_hkl = 1

    # either first n hkl values or all where use_flag >0
    i = 0
    while n_hkl <= no_of_ugs:
        if i >= hkl.n_hkls:
            break
        if rc.no_of_ugs > 0 or hkl.i_obs[i] > 0:
            bloch.hkl_output[n_hkl] = hkl.input_hkls[i]
            n_hkl += 1
        i += 1

    # remove any all-zero rows and sort by q
    # bloch.hkl_output = bloch.hkl_output[~np.all(bloch.hkl_output == 0, axis=1)]
    # using not all makes sure we keep 000 as first index
    bloch.hkl_output = bloch.hkl_output[np.any(bloch.hkl_output != 0, axis=1)
                                        | np.all(bloch.hkl_output == 0, axis=1)]
    q = np.linalg.norm(bloch.hkl_output @ xtal.r_cart, axis=1)
    sorted_indices = np.argsort(q)
    bloch.hkl_output = bloch.hkl_output[sorted_indices]

    # hkl values for all g vectors in the beam pool
    bloch.hkl_indices = px.hkl_make(bloch, hkl, cell, rc)
    bloch.n_hkl = len(bloch.hkl_indices)
    # g-vectors in reciprocal space [hkl indices, x, y, z]
    # integer hkl -> Cartesian in m^-1
    bloch.g_pool = bloch.hkl_indices @ xtal.b_recip
    # and in 1/A
    bloch.g_pool_mag = np.linalg.norm(bloch.g_pool, axis=1)*angstrom

    # output hkl list sorted by q
    if rc.iter_count == 0:
        if rc.print_flag > 0:
            print("Output hkl values")
            for hkl in bloch.hkl_output:
                print(hkl)

    # if output reflection has no corresponding g in pool use nearest
    # takes almost no time so no need to check if output reflections changed
    d2 = np.sum((bloch.hkl_output[:, None, :] - bloch.hkl_indices[None, :, :])**2,
                axis=2)
    best_match = np.argmin(d2, axis=1)
    d2_min = d2[np.arange(len(bloch.hkl_output)), best_match]

    # all exact?
    # no need to print if no_of_ugs > 0 as output are in hkl.input_hkls
    if np.max(d2_min) == 0 and rc.no_of_ugs == 0:
        print("Output hkl values all in beam pool")

    # print substitutions where no exact match
    if np.max(d2_min) > 0 and rc.no_of_ugs == 0:
        print("Output hkl values in beam pool")
        for i in np.where(d2_min > 0)[0]:
            print(f"  hkl.output {bloch.hkl_output[i]} replaced by",
                  f" {bloch.hkl_indices[best_match[i]]}")
            bloch.hkl_output[i] = bloch.hkl_indices[best_match[i]]

    # if refinement mode includes E then read and process the experiment
    if rc.iter_count == 0 and 'E' in rc.refine_mode:
        if np.max(d2_min) > 0:
            print("Can't use experiment if output hkl values are not in beam pool")
            exit()
        if rc.debug > 0:
            print("Output hkl values all in beam pool, reading experiment")
        px.read_expt_images(cbed, bloch, rc)
        # process experimental images to match simulation shape
        process_images(cbed, rc)

    # hkl list to index in g_pool for output reflections
    out = px.hkl_lut(bloch.hkl_output, bloch.hkl_indices)
    if np.any(out < 0):
        print("Error in output reflection lookup")
        print("hkl_output\n", bloch.hkl_output)
        print("hkl_indices\n", bloch.hkl_indices)

    # ===============================================
    # make list of g-vectors from output hkl's and nearest neighbours
    # this list is used to construct the matrix for the Bloch wave calc
    # any change to output hkl's results in a new list

    # set of candidate g-vectors for output and nearest neighbours
    # z = [0, 1, -1], y = [0, 1, -1], x = [0, 1, -1]
    nz = np.array([0, 1, -1], dtype='int')
    nx = np.array([0, 1, -1], dtype='int')
    ny = np.array([0, 1, -1], dtype='int')
    # if no holz then z = 0 only
    if rc.holz_flag == 0:
        nz = np.array([0], dtype='int')

    # list of hkl indices in g_pool for output and nearest neighbours
    # (no duplicate values)
    list_hkl = []
    for i in out:
        for z in nz:
            for y in ny:
                for x in nx:
                    this_hkl = bloch.hkl_indices[i] + [x, y, z]
                    this_index = px.hkl_lut(this_hkl, bloch.hkl_indices)
                    if this_index >= 0:
                        list_hkl.append(this_index)

    # remove duplicate values and sort
    list_hkl = np.unique(list_hkl)
    n_hkl = len(list_hkl)

    # now we need to include all g-vectors that connect any pair of vectors
    # in list_hkl and their nearest neighbours

    # g-vectors in matrix [hkl indices in g_pool, x, y, z]
    # make matrix from all differences between candidate vectors
    g_matrix = np.zeros([n_hkl, n_hkl, 3], dtype='int')
    for i in range(n_hkl):
        for j in range(n_hkl):
            g_matrix[i, j] = (bloch.hkl_indices[list_hkl[i]]
                              - bloch.hkl_indices[list_hkl[j]])

    # flatten and remove duplicate values
    g_matrix = g_matrix.reshape(n_hkl*n_hkl, 3)
    g_matrix = np.unique(g_matrix, axis=0)

    # add nearest neighbours for each g-vector in matrix
    list_g = []
    for g in g_matrix:
        for z in nz:
            for y in ny:
                for x in nx:
                    this_g = g + [x, y, z]
                    this_index = px.hkl_lut(this_g, bloch.hkl_indices)
                    if this_index >= 0:
                        list_g.append(this_index)

    # remove duplicate values and sort
    list_g = np.unique(list_g)

    # list of g-vectors in reciprocal space [hkl indices, x, y, z]
    bloch.g_matrix = bloch.hkl_indices[list_g]

    # unique |g| values for scattering factors
    bloch.uniq_gmag = np.unique(np.round(bloch.g_pool_mag[list_g], 8))

    # set up complex matrix for Ug and others
    n_g = len(list_g)
    bloch.ug_matrix = np.zeros([n_g, n_g], dtype=np.complex128)

    # if debug plot the g-vectors in pool and in matrix
    if rc.debug > 1:
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
        # masks to weight the refinement
        if rc.correlation_type == 6 and 'X' not in rc.refine_mode:
            px.read_mask(cbed, bloch, xtal, rc)
            print("    Weighting masks loaded")
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
    # pixel-by-pixel calculations from here
    n_xy = 2 * rc.image_radius
    n_pixels = n_xy * n_xy
    n_out = len(bloch.hkl_output)

    # Optional parallelisation across pixels:
    # rc.n_jobs > 1 or rc.n_jobs == -1 (all cores) triggers process parallel.
    n_jobs_raw = getattr(rc, "n_jobs", 1)
    try:
        n_jobs = 1 if n_jobs_raw is None else int(n_jobs_raw)
    except (TypeError, ValueError):
        n_jobs = 1
    use_parallel = n_jobs == -1 or n_jobs > 1

    if use_parallel:
        cpu_total = os.cpu_count() or 1
        if n_jobs == -1:
            n_jobs_eff = cpu_total
        else:
            n_jobs_eff = max(1, min(n_jobs, cpu_total))

        # Cross-platform process context:
        # - Windows: spawn (safe default)
        # - POSIX: fork (lower overhead)
        from multiprocessing import get_context
        ctx = get_context("spawn" if os.name == "nt" else "fork")

        pixels = [(pix_x, pix_y) for pix_x in range(n_xy) for pix_y in range(n_xy)]
        chunksize = max(1, n_pixels // (8 * n_jobs_eff))

        done = 0
        with ctx.Pool(
            processes=n_jobs_eff,
            initializer=_init_pixel_worker,
            initargs=(bloch, rc, n_out),
        ) as pool:
            for pix_x, pix_y, intensity_out in pool.imap_unordered(
                _compute_pixel_task, pixels, chunksize=chunksize
            ):
                # Map diffracted intensity to required output g vectors
                # note x and y swapped!
                cbed.lacbed_sim[:, -pix_y, pix_x, :] = intensity_out

                done += 1
                if done % max(1, n_pixels // 100) == 0 or done == n_pixels:
                    print(f"\rBloch wave calculation... {100*done/n_pixels:.0f}%", end="")
    else:
        for pix_x in range(n_xy):
            # progress by row
            print(f"\rBloch wave calculation... {100*pix_x/n_xy:.0f}%", end="")

            for pix_y in range(n_xy):
                bloch.s_g_pix = np.squeeze(bloch.s_g[pix_x, pix_y, :])
                bloch.k_dot_n_pix = bloch.k_dot_n[pix_x, pix_y]

                # works for multiple thicknesses
                px.wave_functions(bloch, rc)

                intensity = np.abs(bloch.wave_function) ** 2

                # Map diffracted intensity to required output g vectors
                # note x and y swapped!
                cbed.lacbed_sim[:, -pix_y, pix_x, :] = intensity[:, :n_out]
    # = = = = = = = = = = = = = = = = = = = = = = = =

    # timings
    # setup = mid-strt
    bwc = time.time()-mid
    print(f"\rBloch wave calculation... done in {bwc:.1f}s")  # " (beam pool setup {setup:.1f} s)")
    if rc.iter_count == 0:
        print(f"    {1000*(bwc)/(4*rc.image_radius**2):.2f} ms/pixel")

    # increment iteration counter
    rc.iter_count += 1

    return


# ...
