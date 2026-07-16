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
    # i_obs>0 are used
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
    g_matrix = np.array([bloch.hkl_indices[i]-bloch.hkl_indices[j]
                         for i in list_hkl for j in list_hkl], dtype='int')

    # list of hkl indices in g_pool for matrix vectors and nearest neighbours
    # (no duplicate values)
    list_hkl = []
    for g in g_matrix:
        for z in nz:
            for y in ny:
                for x in nx:
                    this_hkl = g + [x, y, z]
                    this_index = px.hkl_lut(this_hkl, bloch.hkl_indices)
                    if this_index >= 0:
                        list_hkl.append(this_index)

    # remove duplicate values
    list_hkl = np.unique(list_hkl)

    # hkl values for g-vectors in matrix
    bloch.g_matrix = bloch.hkl_indices[list_hkl]
    bloch.n_beams = len(bloch.g_matrix)

    # g-vectors in reciprocal space [hkl indices, x, y, z]
    # integer hkl -> Cartesian in m^-1
    bloch.g_matrix = bloch.g_matrix @ xtal.b_recip

    # g vector lengths in 1/A and unique values sorted
    g_length = np.linalg.norm(bloch.g_matrix, axis=1)*angstrom
    bloch.uniq_gmag = np.unique(np.round(g_length, 4))

    if rc.debug > 1:
        print(f"{len(bloch.hkl_indices)} vectors in beam pool")
        print(f"{bloch.n_beams} vectors in matrix")
        print(f"{len(out)} output vectors")
        print(f"{len(bloch.uniq_gmag)} unique g lengths")

    # output hkl values in matrix
    out = px.hkl_lut(bloch.hkl_output, bloch.hkl_indices[list_hkl])
    if np.any(out < 0):
        print("Error in matrix output lookup")
        print("hkl_output\n", bloch.hkl_output)

    # matrix of U(g-h) [h, g]
    # 1. make matrix of all differences between matrix vectors
    dg = bloch.g_matrix[:, None, :] - bloch.g_matrix[None, :, :]
    # 2. convert to hkl and look up U values
    hkl_dg = np.rint(dg @ xtal.r_recip).astype(int)
    ug_lut = px.hkl_lut(hkl_dg.reshape(-1, 3), hkl.input_hkls)
    ug_lut = ug_lut.reshape(bloch.n_beams, bloch.n_beams)
    bloch.ug_matrix = np.zeros((bloch.n_beams, bloch.n_beams), dtype='complex')

    valid = ug_lut >= 0
    bloch.ug_matrix[valid] = hkl.ug[ug_lut[valid]]

    # absorption term
    if rc.absorption_method > 0:
        if rc.absorption_method == 1:
            bloch.ug_matrix *= (1 + 1j*rc.absorption_per/100.0)
        elif rc.absorption_method == 2:
            abs_ug = np.abs(bloch.ug_matrix)
            bloch.ug_matrix += 1j*rc.absorption_per/100.0*abs_ug

    # strong beam list
    # strong beam cutoff from min_reflection_pool
    strong_cutoff = rc.min_strong_beams
    weak_cutoff = rc.min_weak_beams

    # if no data in hkl.obs use matrix g magnitude criteria
    # always include direct beam index 0
    if hkl.n_hkls == 0 or np.max(hkl.i_obs) <= 0:
        bloch.strong_beam = np.where(g_length < strong_cutoff/angstrom)[0]
        if len(bloch.strong_beam) == 0:
            bloch.strong_beam = np.array([0], dtype='int')
        bloch.strong_beam = np.unique(np.append(0, bloch.strong_beam))

    else:
        # from observed reflections, map to matrix indices
        obs_hkl = hkl.input_hkls[hkl.i_obs > 0]
        strong = px.hkl_lut(obs_hkl, np.rint(bloch.g_matrix @ xtal.r_recip).astype(int))
        strong = strong[strong >= 0]
        bloch.strong_beam = np.unique(np.append(0, strong))

    # remove weak beams from matrix if requested
    if rc.min_weak_beams > 0:
        keep = g_length < weak_cutoff/angstrom
        keep[0] = True
        idx = np.where(keep)[0]
        bloch.g_matrix = bloch.g_matrix[idx]
        bloch.ug_matrix = bloch.ug_matrix[np.ix_(idx, idx)]
        # remap output and strong beams
        old_to_new = -np.ones(np.max(idx)+1, dtype='int')
        old_to_new[idx] = np.arange(len(idx))
        out = old_to_new[out]
        bloch.strong_beam = old_to_new[bloch.strong_beam]
        bloch.strong_beam = bloch.strong_beam[bloch.strong_beam >= 0]
        bloch.n_beams = len(idx)

    # dot products with surface normal
    bloch.g_dot_norm = bloch.g_matrix @ xtal.normal

    # setup thickness array
    if rc.iter_count == 0:
        if rc.n_thickness <= 0:
            rc.thickness = np.arange(rc.initial_thickness,
                                     rc.final_thickness+0.5*rc.delta_thickness,
                                     rc.delta_thickness)
            rc.n_thickness = len(rc.thickness)

    # shape arrays
    n_xy = 2*rc.image_radius
    n_out = len(out)

    # initialize output arrays
    cbed.lacbed_sim = np.zeros((rc.n_thickness, n_xy, n_xy, n_out), dtype='float')

    # make s_g and k·n for each pixel for all beams
    # angular coordinates in mrad
    x = np.arange(-rc.image_radius, rc.image_radius)
    y = np.arange(-rc.image_radius, rc.image_radius)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # tilt vectors
    alpha = xx * rc.convergence_angle / rc.image_radius / 1000.0
    beta = yy * rc.convergence_angle / rc.image_radius / 1000.0

    bloch.tilted_k = (bloch.big_k[None, None, :]
                      + alpha[..., None]*xtal.x_direction[None, None, :]
                      + beta[..., None]*xtal.y_direction[None, None, :])

    # excitation errors s_g [x, y, g]
    # s_g = - (k·g + g²/2) / |k|
    g2 = np.sum(bloch.g_matrix**2, axis=1)
    k_dot_g = np.einsum('xyc,gc->xyg', bloch.tilted_k, bloch.g_matrix)
    bloch.s_g = -(k_dot_g + 0.5*g2[None, None, :]) / (bloch.big_k_mag/angstrom)

    # k·n [x, y]
    bloch.k_dot_n = np.einsum('xyc,c->xy', bloch.tilted_k, xtal.normal)

    # Bloch wave calc
    strt = time.time()

    # setup matrix terms common to all pixels
    # (inside px.wave_functions uses bloch.s_g_pix and bloch.k_dot_n_pix)

    mid = time.time()

    # = = = = = = = = = = = = = = = = = = = = = = = =
    # pixel by pixel calculations from here
    for pix_x in range(n_xy):

        # simple progress by row
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


def process_images(cbed, rc):
    """
    Process experimental images to align shape and preprocessing with simulation.
    """

    if cbed.lacbed_expt_raw is None:
        return

    img = np.array(cbed.lacbed_expt_raw, copy=True)

    # optionally median or sobel preprocessing
    if getattr(rc, 'image_processing', 0) == 1:
        for k in range(img.shape[-1]):
            img[..., k] = median(img[..., k])
    elif getattr(rc, 'image_processing', 0) == 2:
        for k in range(img.shape[-1]):
            img[..., k] = sobel(img[..., k])

    # optional blur
    if getattr(rc, 'blur_radius', 0) > 0:
        img = gaussian_filter(img, sigma=(rc.blur_radius, rc.blur_radius, 0))

    cbed.lacbed_expt = img


# =============================================================================
# Correlation and refinement helpers
# =============================================================================

def correlations(cbed, rc):
    """
    Build signature images and pairwise correlation matrix used in refinement.
    """

    # signature images must exist [n_variables, x, y, n_out]
    if cbed.lacbed_sig is None:
        return

    nv = cbed.lacbed_sig.shape[0]
    n_out = cbed.lacbed_sig.shape[-1]

    # normalise signatures by RMS over x,y
    rms = np.sqrt(np.mean(cbed.lacbed_sig**2, axis=(1, 2), keepdims=True))
    rms[rms < eps] = 1.0
    s = cbed.lacbed_sig / rms

    # pairwise correlations per output image
    n_pairs = nv*(nv-1)//2
    cbed.correlation_matrix = np.zeros((n_pairs, n_out), dtype='float')

    k = 0
    for i in range(nv):
        for j in range(i+1, nv):
            num = np.sum(s[i]*s[j], axis=(0, 1))
            den = np.sqrt(np.sum(s[i]**2, axis=(0, 1))*np.sum(s[j]**2, axis=(0, 1)))
            den[den < eps] = 1.0
            cbed.correlation_matrix[k] = num/den
            k += 1


def figure_of_merit(cbed, rc):
    """
    Compute overall figure of merit.
    """
    if cbed.lacbed_expt is None or cbed.lacbed_sim is None:
        rc.fom = 0.0
        return

    # compare at first thickness by default
    sim = cbed.lacbed_sim[0]
    expt = cbed.lacbed_expt

    # ensure shapes are compatible
    if sim.shape != expt.shape:
        # basic crop/pad fallback not intended as final solution
        minx = min(sim.shape[0], expt.shape[0])
        miny = min(sim.shape[1], expt.shape[1])
        mino = min(sim.shape[2], expt.shape[2])
        sim = sim[:minx, :miny, :mino]
        expt = expt[:minx, :miny, :mino]

    diff = sim - expt
    rc.fom = np.sqrt(np.mean(diff**2))


# =============================================================================
# Plot helpers
# =============================================================================

def save_masks(cbed, rc):
    """
    Save mask images if present.
    """
    if cbed.lacbed_mask is None:
        return

    out_dir = os.path.join(rc.path, 'masks') if rc.path is not None else 'masks'
    os.makedirs(out_dir, exist_ok=True)

    n_var = cbed.lacbed_mask.shape[0]
    n_out = cbed.lacbed_mask.shape[-1]

    for i in range(n_var):
        for k in range(n_out):
            fn = os.path.join(out_dir, f'mask_v{i:03d}_o{k:03d}.npy')
            np.save(fn, cbed.lacbed_mask[i, :, :, k])


def plot_correlations(cbed, rc):
    """
    Display correlation matrix as heatmap.
    """
    if cbed.correlation_matrix is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(cbed.correlation_matrix, aspect='auto', cmap='coolwarm',
                   vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_xlabel('Output pattern index')
    ax.set_ylabel('Parameter pair index')
    text_effect = withStroke(linewidth=2, foreground='black')
    ax.text(0.01, 0.98, f"n_pairs={cbed.correlation_matrix.shape[0]}",
            transform=ax.transAxes, va='top', ha='left',
            color='w', path_effects=[text_effect], fontsize=14)
    tit = "Correlations"
    plt.title(tit, fontsize=24)

    plt.tight_layout()
    plt.show()
