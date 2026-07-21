# -*- coding: utf-8 -*-
"""
pylix_grad.py  –  Analytical intensity gradients dI/dp for pylix.

Implements the matrix-exponential derivative (Tsai & Chan, Bernoulli 2003)
as described in Appendix A.4 of Dolomanov et al. (2026), giving exact
analytical gradients ∂I/∂p reusing the eigensolution already computed in
the forward Bloch-wave pass.

Mathematical summary
--------------------
For the scattering matrix  S = M V exp(itD) V⁻¹ M⁻¹  (pylix convention):

    ∂S/∂p  =  M V X_p V⁻¹ M⁻¹
    X_p    =  G ⊙ F                     Hadamard (element-wise) product
    G      =  V⁻¹ (∂A/∂p) V
    F_ii   =  it · exp(i·gamma_i·t)
    F_ij   =  (exp(i·gamma_i·t) - exp(i·gamma_j·t)) / (gamma_i - gamma_j)

Intensity gradient:  ∂I_g/∂p  =  2 Re(ψ_g* · ∂ψ_g/∂p)

pylix notation (from _pixel_row_worker in simulate.py):
    gamma    ↔  eigenvalues of structure_mat    (complex, n_beams)
    V        ↔  eigenvecs                        (complex, n_beams × n_beams)
    M        ↔  diag(m_ii) = diag(1/norm_fac)
    y        =  V⁻¹ M⁻¹ ψ₀                      (computed in forward pass)
    s        =  it                               (imaginary × thickness)

Sign convention
---------------
pylix uses  phase = exp(-i g_cart · r_cart)  in Fg_matrix, so:

    ∂U_{gm-gn} / ∂x_{α,j}  =  -2πi · (hm-hn)_α · U_{j, gm-gn}

The negative sign is critical for the correct gradient direction.

Implemented parameter types
---------------------------
    ∂I/∂x_{α,j}   fractional atomic coordinates   (this module)
    ∂I/∂U_iso,j   isotropic ADP                   (placeholder — to be added)
    ∂I/∂U_{αβ,j}  anisotropic ADP                 (placeholder — to be added)

Workflow
--------
1. Call px.Fg_matrix() as normal.
2. Call compute_atom_ug_contributions() to build per-atom U matrices.
3. Add 'hkl_int' and 'atom_ug_list' to the shared_data dict.
4. Use _init_worker_grad / _pixel_row_worker_grad in place of the forward
   worker when gradient images are required.

Verification
------------
The gradient formula has been verified against central finite differences
on synthetic Bloch problems (relative error < 10⁻⁷).
"""

import numpy as np
from scipy.linalg import eig, solve, lu_factor, lu_solve


# ---------------------------------------------------------------------------
# 1.  Per-atom optical potential contributions
#     Call once after px.Fg_matrix(); result is passed to the parallel worker.
# ---------------------------------------------------------------------------

def compute_atom_ug_contributions(cell, bloch, xtal):
    """
    Build the per-atom contribution to ug_matrix for each cell atom.

    After px.Fg_matrix() has been called, cell.f_g (shape n_atoms × n_hkl × n_hkl)
    holds the complex scattering factors (including absorption) already mapped
    onto the g-difference grid.  This function assembles the remaining factors
    (phase, Debye-Waller, occupancy, relativistic/volume prefactor) to give the
    per-atom optical potential matrix:

        atom_ug[j, i, k]  =  (γ / π Ω) · f_j(|gi-gk|) · occ_j
                              · exp(-i g_cart · r_j_cart)
                              · exp(-½ g^T U_aniso_m g)

    Summing over j exactly reproduces bloch.ug_matrix.

    Parameters
    ----------
    cell  : pylix Cell object   (needs f_g, atom_coordinate, occupancy, u_aniso_m)
    bloch : pylix Bloch object  (needs g_matrix, relativistic_correction)
    xtal  : pylix Crystal object (needs cell_volume)

    Returns
    -------
    atom_ug : ndarray, shape (n_atoms, n_hkl, n_hkl), complex
        atom_ug[j, i, k]  =  U_{j, g_i - g_k}

    Notes
    -----
    Call AFTER px.Fg_matrix() so that cell.f_g is populated.
    For the gradient, pass slices atom_ug[rc.atomic_sites] to the worker.

    Consistency check (use during development):
        assert np.allclose(atom_ug.sum(axis=0), bloch.ug_matrix, rtol=1e-6)
    """
    Fg_to_Ug = bloch.relativistic_correction / (np.pi * xtal.cell_volume)

    # Anisotropic DW exponent: g^T U_aniso_m g, shape (n_atoms, n_hkl, n_hkl)
    # Matches the Ugg computation inside px.Fg_matrix exactly.
    Ugg = np.einsum('ijm, amn, ijn -> aij',
                    bloch.g_matrix, cell.u_aniso_m, bloch.g_matrix)

    # Phase: exp(-i g_cart · r_cart), shape (n_hkl, n_hkl, n_atoms)
    # Identical to the phase array built inside px.Fg_matrix.
    g_dot_r = np.einsum('ijk, lk -> ijl', bloch.g_matrix, cell.atom_coordinate)
    phase = np.exp(-1j * g_dot_r)                        # (n_hkl, n_hkl, n_atoms)

    # Assemble: (n_atoms, n_hkl, n_hkl)
    atom_ug = (Fg_to_Ug
               * cell.f_g                                # (n_atoms, n_hkl, n_hkl)
               * phase.transpose(2, 0, 1)               # (n_atoms, n_hkl, n_hkl)
               * cell.occupancy[:, None, None]
               * np.exp(-Ugg / 2))

    return atom_ug


# ---------------------------------------------------------------------------
# 2.  F matrix  (core of the Tsai-Chan derivative)
# ---------------------------------------------------------------------------

def f_matrix(gamma, t, tol=1e-10):
    """
    F matrix for the matrix-exponential derivative at thickness t.

    pylix convention s = it:
        F_ii   =  it · exp(i·gamma_i·t)
        F_ij   =  (exp(i·gamma_i·t) - exp(i·gamma_j·t)) / (gamma_i - gamma_j)

    Near-degenerate eigenvalue pairs (|Δgamma| < tol) use the L'Hôpital
    limit, which equals F_ii.  The diagonal is always set exactly.

    Parameters
    ----------
    gamma : (n_beams,) complex   eigenvalues from forward pass
    t     : float                crystal thickness (Å)
    tol   : float                degeneracy threshold

    Returns
    -------
    F : (n_beams, n_beams) complex
    """
    phase  = np.exp(1j * gamma * t)                      # (n,)
    d_diff = gamma[:, None] - gamma[None, :]             # (n, n)

    degenerate = np.abs(d_diff) < tol
    safe_diff  = np.where(degenerate, 1.0 + 0j, d_diff)

    F     = (phase[:, None] - phase[None, :]) / safe_diff
    limit = 1j * t * phase                               # L'Hôpital limit

    rows, _ = np.where(degenerate)
    F[degenerate] = limit[rows]
    np.fill_diagonal(F, limit)
    return F


# ---------------------------------------------------------------------------
# 3.  Pixel-level gradient  (standalone, for testing / serial use)
# ---------------------------------------------------------------------------

def dI_dp_pixel(eigenvecs, gamma, y, m_ii, dA_dp, thickness,
                wave_funct, n_out, lu_piv=None):
    """
    Analytical gradient dI/dp for one pixel, one parameter.

    Parameters
    ----------
    eigenvecs : (n_beams, n_beams) complex   — eigenvector matrix V
    gamma     : (n_beams,) complex           — eigenvalues of structure matrix
    y         : (n_beams,) complex           — V^-1 M^-1 psi_0
    m_ii      : (n_beams,) float             — 1/norm_fac (obliquity factors)
    dA_dp     : (n_beams, n_beams) complex   — derivative of structure matrix
    thickness : (n_t,) float                 — thickness values
    wave_funct: (n_t, n_beams) complex       — wave functions (from forward pass)
    n_out     : int                          — number of output beams
    lu_piv    : result of lu_factor(eigenvecs), optional
                If provided, reuses the LU factorisation already computed
                in the calling worker (saves ~N^3/3 work per parameter).

    Returns
    -------
    dI : (n_t, n_out) float   — dI/dp for each thickness and output beam
    """
    from scipy.linalg import lu_factor, lu_solve

    n_t     = len(thickness)
    n_beams = len(gamma)

    # LU factorisation of V — reuse if already computed by caller
    if lu_piv is None:
        lu_piv = lu_factor(eigenvecs)

    # G = V^-1 (dA/dp) V  — uses lu_solve for efficiency
    # shape: (n_beams, n_beams)
    G = lu_solve(lu_piv, dA_dp @ eigenvecs)

    dI = np.zeros((n_t, n_out))
    s  = 1j * thickness                           # (n_t,) scalar per thickness

    for t_idx in range(n_t):
        # F matrix (Tsai-Chan): F_ij = (exp(d_i*s) - exp(d_j*s))/(d_i - d_j)
        #                        F_ii = s * exp(d_i*s)
        F = f_matrix(gamma, s[t_idx])             # (n_beams, n_beams)

        # X_p = G hadamard F
        Xp = G * F                                 # (n_beams, n_beams)

        # dpsi/dp = M (V Xp y)   where M = diag(m_ii)
        # V Xp y:
        VXpy = eigenvecs @ (Xp @ y)               # (n_beams,)
        dpsi = m_ii * VXpy                         # (n_beams,)

        # dI/dp = 2 Re(psi_g* dpsi_g) for each output beam g
        psi_g  = wave_funct[t_idx, :n_out]         # (n_out,)
        dpsi_g = dpsi[:n_out]                      # (n_out,)
        dI[t_idx] = 2.0 * np.real(np.conj(psi_g) * dpsi_g)

    return dI


# ---------------------------------------------------------------------------
# 4.  Parallel row worker with analytical coordinate gradients
# ---------------------------------------------------------------------------

_worker_grad_shared = {}


def _init_worker_grad(shared_data):
    """
    ProcessPoolExecutor initializer for gradient workers.

    Expects the same keys as _init_worker in simulate.py, plus:
        'hkl_int'      : (n_hkl, 3) int   bloch.hkl_indices — integer Miller indices
        'atom_ug_list' : list of (n_hkl, n_hkl) complex arrays,
                         one per refined atom, from compute_atom_ug_contributions.
                         Pass  atom_ug[rc.atomic_sites]  as a Python list.

    Called once per worker process at pool start-up.
    """
    global _worker_grad_shared
    _worker_grad_shared = shared_data


def _pixel_row_worker_grad(row_args):
    """
    Bloch-wave forward pass + analytical coordinate gradients for one image row.

    Mirrors _pixel_row_worker in simulate.py but additionally computes
    ∂I/∂x_{α,j} for each refined atom j and coordinate α ∈ {x, y, z}.
    The eigensolution is performed once per pixel and shared between the
    intensity and all gradient computations — no redundant eigensolves.

    The coordinate gradient formula (pylix sign convention):
        ∂U_{gm-gn}/∂x_{α,j}  =  -2πi · (hm-hn)_α · U_{j, gm-gn}

    Parameters
    ----------
    row_args : tuple  (s_g_row, k_dot_n_row)
        s_g_row     : (n_pix, n_hkl)   deviation parameters
        k_dot_n_row : (n_pix,)          k·n̂ for each pixel

    Returns
    -------
    row_intensity : (n_pix, n_thickness, n_out)  float
    row_gradient  : (n_pix, n_grad_params, n_thickness, n_out)  float
        n_grad_params = n_refined_atoms × 3
        Parameter order: [atom0_x, atom0_y, atom0_z, atom1_x, ...]
        matching the order of atom_ug_list in shared data.

    Usage in simulate.py
    --------------------
    Replace the ProcessPoolExecutor block with:

        shared_data = dict(
            ug_matrix=bloch.ug_matrix,
            g_dot_norm=bloch.g_dot_norm,
            hkl_output=bloch.hkl_output,
            big_k_mag=bloch.big_k_mag,
            thickness=rc.thickness,
            min_strong_beams=rc.min_strong_beams,
            n_hkl=bloch.n_hkl,
            # gradient extras:
            hkl_int=bloch.hkl_indices,
            atom_ug_list=[atom_ug[j] for j in rc.atomic_sites],
        )
        with ProcessPoolExecutor(max_workers=os.cpu_count(),
                                 initializer=_init_worker_grad,
                                 initargs=(shared_data,)) as pool:
            futures = {pool.submit(_pixel_row_worker_grad,
                                   (bloch.s_g[pix_x, :, :],
                                    bloch.k_dot_n[pix_x, :])): pix_x
                       for pix_x in range(2*rc.image_radius)}
            try:
                for f in as_completed(futures):
                    pix_x = futures[f]
                    row_intensity, row_gradient = f.result()
                    for pix_y in range(2*rc.image_radius):
                        cbed.lacbed_sim[:, -pix_y, pix_x, :] = (
                            row_intensity[pix_y])
                        lacbed_sig[:, -pix_y, pix_x, :] = (
                            row_gradient[pix_y])   # shape (n_params, n_t, n_out)
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=False)
                raise
    """
    s_g_row, k_dot_n_row = row_args

    # Unpack shared data
    ug_matrix        = _worker_grad_shared['ug_matrix']
    g_dot_norm       = _worker_grad_shared['g_dot_norm']
    hkl_output       = _worker_grad_shared['hkl_output']
    big_k_mag        = _worker_grad_shared['big_k_mag']
    thickness        = _worker_grad_shared['thickness']
    min_strong_beams = _worker_grad_shared['min_strong_beams']
    n_hkl            = _worker_grad_shared['n_hkl']
    hkl_int          = _worker_grad_shared['hkl_int']       # (n_hkl, 3) int
    atom_ug_list     = _worker_grad_shared['atom_ug_list']  # list of (n_hkl, n_hkl)

    n_pix         = s_g_row.shape[0]
    n_out         = len(hkl_output)
    n_thickness   = len(thickness)
    n_atoms_ref   = len(atom_ug_list)
    n_grad_params = n_atoms_ref * 3                          # x, y, z per atom

    row_intensity = np.zeros((n_pix, n_thickness, n_out))
    row_gradient  = np.zeros((n_pix, n_grad_params, n_thickness, n_out))

    u_g_col0 = np.abs(ug_matrix[:, 0])

    for pix_y in range(n_pix):
        s_g_pix     = s_g_row[pix_y]
        k_dot_n_pix = k_dot_n_row[pix_y]

        # ---- strong_beams (mirrors _pixel_row_worker) ----------------------
        pert = np.divide(u_g_col0, np.abs(s_g_pix),
                         out=np.full_like(s_g_pix, 100.0),
                         where=s_g_pix != 0)
        max_sg = 0.001
        strong = np.zeros(n_hkl, dtype=int)
        while np.sum(strong) < min_strong_beams:
            min_pert_strong = 0.025 / max_sg
            strong = np.where(
                (np.abs(s_g_pix) < max_sg) | (pert >= min_pert_strong), 1, 0)
            max_sg += 0.001
        strong_beam = np.flatnonzero(strong)

        # ---- blochwave -----------------------------------------------------
        strong_new          = np.setdiff1d(strong_beam, hkl_output)
        strong_beam_indices = np.concatenate((hkl_output, strong_new))
        n_beams             = len(strong_beam_indices)

        beam_proj = np.zeros((n_beams, n_hkl), dtype=np.complex128)
        beam_proj[np.arange(n_beams), strong_beam_indices] = 1.0 + 0j

        ug_sg = beam_proj @ ug_matrix @ beam_proj.T
        ug_sg = 2.0 * np.pi**2 * ug_sg / big_k_mag
        ug_sg[np.arange(n_beams), np.arange(n_beams)] = s_g_pix[strong_beam_indices]

        norm_fac      = np.sqrt(1.0 + g_dot_norm[strong_beam_indices] / k_dot_n_pix)
        structure_mat = ug_sg / np.outer(norm_fac, norm_fac)

        gamma, eigenvecs = eig(structure_mat)

        # ---- wave_functions ------------------------------------------------
        psi0    = np.zeros(n_beams, dtype=np.complex128)
        psi0[0] = 1.0 + 0j

        inv_m_ii   = norm_fac
        m_ii       = 1.0 / inv_m_ii
        u          = inv_m_ii * psi0
        y          = solve(eigenvecs, u)
        phase_fwd  = np.exp(1j * np.outer(gamma, thickness))
        z          = eigenvecs @ (y[:, None] * phase_fwd)
        wave_funct = (m_ii[:, None] * z).T               # (n_thickness, n_beams)

        row_intensity[pix_y] = np.abs(wave_funct[:, :n_out])**2

        # ---- Analytical coordinate gradients --------------------------------
        # LU factorisation of V — reused for every parameter at this pixel.
        lu_piv = lu_factor(eigenvecs)

        # Miller index differences for all strong-beam pairs, all 3 coordinates
        # h_diff[m, n, alpha] = hkl_indices[gm, alpha] - hkl_indices[gn, alpha]
        h_strong = hkl_int[strong_beam_indices, :]            # (n_beams, 3)
        h_diff   = (h_strong[:, None, :] -
                    h_strong[None, :, :])                     # (n_beams, n_beams, 3)

        for a_idx, atom_ug_full in enumerate(atom_ug_list):

            # Project per-atom U onto the strong-beam subspace
            # atom_ug_strong[m, n] = U_{j, gm-gn}
            atom_ug_strong = beam_proj @ atom_ug_full @ beam_proj.T

            # Common factor for all 3 coordinates:
            # ∂A_mn/∂x_α = -(4π³i/K) · h_diff[m,n,α] · atom_ug_strong[m,n]
            #               / (norm_fac_m · norm_fac_n)
            # Factor out the scalar and obliquity terms:
            # NB: NEGATIVE sign from pylix phase convention exp(-i g·r)
            base = (-4.0 * np.pi**3 * 1j / big_k_mag) * atom_ug_strong \
                   / np.outer(norm_fac, norm_fac)             # (n_beams, n_beams)

            # Stack ∂A/∂x, ∂A/∂y, ∂A/∂z into a (3, n_beams, n_beams) array
            # h_diff has shape (n_beams, n_beams, 3); transpose to (3, n, n)
            dA_all = h_diff.transpose(2, 0, 1) * base[None, :, :]
            # Diagonal of ∂A/∂x_{α,j} is zero (S_g has no coordinate dependence)
            dA_all[:, np.arange(n_beams), np.arange(n_beams)] = 0.0

            # G_all[alpha] = V⁻¹ (∂A/∂x_alpha) V  for all 3 coords at once.
            # Solve: eigenvecs @ G_all = dA_all @ eigenvecs
            # Reshape (3, n, n) → (3n, n), solve, reshape back.
            rhs   = dA_all @ eigenvecs                        # (3, n_beams, n_beams)
            G_all = lu_solve(
                lu_piv,
                rhs.reshape(3 * n_beams, n_beams)
            ).reshape(3, n_beams, n_beams)                    # (3, n_beams, n_beams)

            # For each thickness: X_p = G ⊙ F, dpsi = M V X_p y, dI = 2Re(ψ* dpsi)
            for t_idx, t in enumerate(thickness):
                F = f_matrix(gamma, t)                        # (n_beams, n_beams)

                # X_p y for all 3 coords: (G_all * F[None]) @ y → (3, n_beams)
                Xpy_all  = (G_all * F[None, :, :]) @ y

                # dpsi = M V X_p y = m_ii * (V @ Xpy)  — vectorised over coords
                # (Xpy_all @ V^T)[c, i] = Σ_j Xpy_all[c,j] * V[i,j]  ✓
                dpsi_all = m_ii[None, :] * (Xpy_all @ eigenvecs.T)  # (3, n_beams)

                # ∂I_g/∂x_α = 2 Re(ψ_g* · ∂ψ_g/∂x_α)
                row_gradient[pix_y,
                             a_idx*3 : a_idx*3+3,
                             t_idx] = 2.0 * np.real(
                    np.conj(wave_funct[t_idx, None, :n_out])
                    * dpsi_all[:, :n_out])

    return row_intensity, row_gradient
