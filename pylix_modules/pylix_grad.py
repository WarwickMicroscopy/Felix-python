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

Sign convention
---------------
pylix uses exp(-i g·r) in Fg_matrix, so

    ∂U_{gm-gn} / ∂x_{α,j}  =  -2πi · (hm-hn)_α · U_{j, gm-gn}

Coordinate refinement — symmetry and projection
------------------------------------------------
The refinement variable is  p = r_j · v  (projection of basis atom j's
fractional coordinate onto its allowed movement direction v, from atom_move).

The off-diagonal element A[m,n] of the structure matrix is:

    A[m,n]  =  (2π²/K) / (ν_m ν_n)  ×  U[g_m − g_n]

where ν = norm_fac and U = bloch.ug_matrix.  Differentiating:

    dA[m,n]/dp  =  (2π²/K) / (ν_m ν_n)  ×
                   Σ_{k: cell.basis_atom_index[k] = j}
                   (−2πi) · (h_mn · R_k v) · atom_ug[k,m,n]

where:
    h_mn   = hkl_indices[m] − hkl_indices[n]   (integer Miller differences)
    R_k    = xtal.symmetry_matrix[cell.symop_index[k]]   (3×3 rotation)
    ν      = norm_fac  (pixel-dependent — applied in the worker, not here)

build_atom_ug_grad pre-computes the pixel-independent part for each variable:

    atom_ug_grad[var_idx][m,n]  =
        −2πi · Σ_{k: basis_atom_index[k] == j}  (h_mn · R_k v) · atom_ug[k,m,n]

The worker applies the pixel-dependent obliquity factor:
    dA/dp = atom_ug_grad[var_idx][strong subset] × (2π²/K) × m_ii⊗m_ii

Verification
------------
Central finite differences, step δ = 1e-5 in fractional coordinates:
    dI_fd = (I_plus − I_minus) / (2δ)
    expect |dI_analytical − dI_fd| / |dI_fd| < 1e-5 per pixel
"""

import numpy as np
from scipy.linalg import eig, solve, lu_factor, lu_solve


def compute_atom_ug_contributions(cell, bloch, xtal):
    """
    Per-cell-atom optical potential matrices.

    Reproduces the per-atom terms summed by px.Fg_matrix so that coordinate
    gradients can be formed analytically.

    After px.Fg_matrix() has been called, cell.f_g[k,m,n] holds the complex
    scattering factor (including absorption) for cell atom k and g-difference
    (m,n).  This function applies the remaining phase, Debye-Waller,
    occupancy, and Fg→Ug prefactor:

        atom_ug[k,m,n]  =  (γ_rel / π Ω)
                            × f_k(|g_m − g_n|)
                            × occ_k
                            × exp(−i g_{mn} · r_k)
                            × exp(−½ g_{mn}^T U_aniso_m[k] g_{mn})

    Summing over k reproduces bloch.ug_matrix exactly.

    Parameters
    ----------
    cell  : Cell   (f_g, atom_coordinate, occupancy, u_aniso_m, n_atoms)
    bloch : Bloch  (g_matrix, relativistic_correction)
    xtal  : Crystal (cell_volume)

    Returns
    -------
    atom_ug : ndarray, shape (cell.n_atoms, n_hkl, n_hkl), complex128

    Notes
    -----
    Must be called after px.Fg_matrix() so that cell.f_g is populated.
    Consistency check (during development):
        assert np.allclose(atom_ug.sum(axis=0), bloch.ug_matrix, rtol=1e-6)
    """
    Fg_to_Ug = bloch.relativistic_correction / (np.pi * xtal.cell_volume)

    # anisotropic DW exponent: g^T U_aniso_m[k] g, shape (n_atoms, n_hkl, n_hkl)
    Ugg = np.einsum('ijm, amn, ijn -> aij',
                    bloch.g_matrix, cell.u_aniso_m, bloch.g_matrix)

    # phase exp(−i g · r_k), shape (n_hkl, n_hkl, n_atoms)
    g_dot_r = np.einsum('ijk, lk -> ijl', bloch.g_matrix, cell.atom_coordinate)
    phase = np.exp(-1j * g_dot_r)               # (n_hkl, n_hkl, n_atoms)

    atom_ug = (Fg_to_Ug
               * cell.f_g                        # (n_atoms, n_hkl, n_hkl)
               * phase.transpose(2, 0, 1)        # (n_atoms, n_hkl, n_hkl)
               * cell.occupancy[:, None, None]
               * np.exp(-Ugg / 2))

    return atom_ug   # (n_atoms, n_hkl, n_hkl), complex128


def build_atom_ug_grad(cell, bloch, xtal, rc, atom_ug):
    """
    Pre-compute rotation-weighted, symmetry-summed, v-projected gradient Ug.

    For refinement variable var_idx with allowed direction v and basis atom j:

        atom_ug_grad[var_idx][m, n]  =
            −2πi · Σ_{k: cell.basis_atom_index[k] == j}
                   (h_mn · R_k @ v) · atom_ug[k, m, n]

    where:
        h_mn  = hkl_indices[m] − hkl_indices[n]   (integer Miller differences)
        R_k   = xtal.symmetry_matrix[cell.symop_index[k]]   (3×3 rotation)

    The obliquity factor (2π²/K) / (ν_m ν_n) is NOT included here — it is
    pixel-dependent and applied per-pixel in the worker.

    Call once after compute_atom_ug_contributions(), before the parallel pool.

    Parameters
    ----------
    cell     : Cell    (n_atoms, basis_atom_index, symop_index)
    bloch    : Bloch   (hkl_indices, n_hkl)
    xtal     : Crystal (symmetry_matrix)
    rc       : RunControl (n_variables, refined_variable_type,
                           atom_refine_flag, atom_refine_vec)
    atom_ug  : ndarray (n_atoms, n_hkl, n_hkl) complex128
               from compute_atom_ug_contributions()

    Returns
    -------
    atom_ug_grad : dict { var_idx (int) : ndarray (n_hkl, n_hkl) complex128 }
        Only contains entries for variables with refined_variable_type == 20.
        Diagonal is identically zero (h_diff[m,m] = 0 by construction).
    """
    # integer Miller-index differences h_mn = hkl[m] − hkl[n]
    # shape (n_hkl, n_hkl, 3)
    hkl = bloch.hkl_indices.astype(float)
    h_diff_all = hkl[:, None, :] - hkl[None, :, :]

    atom_ug_grad = {}

    for var_idx in range(rc.n_variables):
        if rc.refined_variable_type[var_idx] != 20:
            continue

        atom_id = rc.atom_refine_flag[var_idx]               # basis atom index
        v = np.array(rc.atom_refine_vec[var_idx], dtype=float)  # (3,) fractional

        weighted = np.zeros((bloch.n_hkl, bloch.n_hkl), dtype=np.complex128)

        for k in range(cell.n_atoms):
            if cell.basis_atom_index[k] != atom_id:
                continue
            R_k = xtal.symmetry_matrix[cell.symop_index[k]]   # (3, 3)
            Rv = R_k @ v                                       # (3,) rotated direction

            # scalar factor[m,n] = h_diff[m,n,:] · Rv
            # h_diff_all shape (n_hkl, n_hkl, 3); @ Rv gives (n_hkl, n_hkl)
            factor = h_diff_all @ Rv
            weighted += factor * atom_ug[k]                  # atom_ug[k] (n_hkl, n_hkl)

        # absorb the −2πi prefactor
        # diagonal is exactly zero because h_diff[m,m] = [0,0,0]
        atom_ug_grad[var_idx] = -2j * np.pi * weighted

    return atom_ug_grad

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
    phase = np.exp(1j * gamma * t)  # [n,]
    d_diff = gamma[:, None] - gamma[None, :]  # [n, n]
    degenerate = np.abs(d_diff) < tol
    safe_diff = np.where(degenerate, 1.0 + 0j, d_diff)
    F = (phase[:, None] - phase[None, :]) / safe_diff
    limit = 1j * t * phase  # L'Hôpital limit
    rows, _ = np.where(degenerate)
    F[degenerate] = limit[rows]
    np.fill_diagonal(F, limit)
    return F


def dI_dp_pixel(eigenvecs, gamma, y, m_ii, dA_dp, thickness,
                wave_funct, n_out, lu_piv=None):
    """
    Analytical gradient dI/dp for one pixel, one refinement variable.

    Parameters
    ----------
    eigenvecs : (n_beams, n_beams) complex   eigenvector matrix V
    gamma     : (n_beams,) complex           eigenvalues
    y         : (n_beams,) complex           V⁻¹ M⁻¹ ψ₀  (from forward pass)
    m_ii      : (n_beams,) float             1/norm_fac
    dA_dp     : (n_beams, n_beams) complex   dA/dp for the strong-beam subset,
                already including the (2π²/K) · m_ii⊗m_ii obliquity factor
    thickness : (n_t,) float                 thickness values (Å)
    wave_funct: (n_t, n_beams) complex       wave functions from forward pass
    n_out     : int                          number of output beams
    lu_piv    : lu_factor result, optional   reuse LU from the forward pass

    Returns
    -------
    dI : (n_t, n_out) float
    """
    if lu_piv is None:
        lu_piv = lu_factor(eigenvecs)

    # G = V⁻¹ (dA/dp) V
    G = lu_solve(lu_piv, dA_dp @ eigenvecs)        # (n_beams, n_beams)

    n_t = len(thickness)
    dI = np.zeros((n_t, n_out))

    for t_idx in range(n_t):
        F = f_matrix(gamma, thickness[t_idx])      # (n_beams, n_beams)

        # X_p = G ⊙ F  (Hadamard product)
        Xp = G * F                                 # (n_beams, n_beams)

        # dψ/dp = M V X_p y
        dpsi = m_ii * (eigenvecs @ (Xp @ y))       # (n_beams,)

        # dI_g/dp = 2 Re(ψ_g* · dψ_g/dp)
        psi_g  = wave_funct[t_idx, :n_out]         # (n_out,)
        dpsi_g = dpsi[:n_out]                      # (n_out,)
        dI[t_idx] = 2.0 * np.real(np.conj(psi_g) * dpsi_g)

    return dI


_worker_grad_shared = {}


def _init_worker_grad(shared_data):
    """
    Parallel row worker with analytical coordinate gradients
    ProcessPoolExecutor initializer for gradient workers.

    Expects the same keys as _init_worker in simulate.py, plus:
        'hkl_int'      : (n_hkl, 3) int
                          bloch.hkl_indices — integer Miller indices
        'atom_ug_list' : list of (n_hkl, n_hkl) complex arrays,
                         one per refined atom,
                         from compute_atom_ug_contributions.
                         Pass  atom_ug[rc.atomic_sites]  as a Python list.

    Called once per worker process at pool start-up.
    """
    global _worker_grad_shared
    _worker_grad_shared = shared_data


def _pixel_row_worker_grad(row_args):
    """
    Bloch-wave forward pass + analytical coordinate gradients for one image row

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
    ug_matrix = _worker_grad_shared['ug_matrix']
    g_dot_norm = _worker_grad_shared['g_dot_norm']
    hkl_output = _worker_grad_shared['hkl_output']
    big_k_mag = _worker_grad_shared['big_k_mag']
    thickness = _worker_grad_shared['thickness']
    min_strong_beams = _worker_grad_shared['min_strong_beams']
    n_hkl = _worker_grad_shared['n_hkl']
    hkl_int = _worker_grad_shared['hkl_int']  # [n_hkl, 3]
    atom_ug_list = _worker_grad_shared['atom_ug_list']  # [n_hkl, n_hkl]

    n_pix = s_g_row.shape[0]
    n_out = len(hkl_output)
    n_thickness = len(thickness)
    n_atoms_ref = len(atom_ug_list)
    n_grad_params = n_atoms_ref * 3  # x, y, z per atom

    row_intensity = np.zeros((n_pix, n_thickness, n_out))
    row_gradient = np.zeros((n_pix, n_grad_params, n_thickness, n_out))
    u_g_col0 = np.abs(ug_matrix[:, 0])

    for pix_y in range(n_pix):
        s_g_pix = s_g_row[pix_y]
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
        strong_new = np.setdiff1d(strong_beam, hkl_output)
        strong_beam_indices = np.concatenate((hkl_output, strong_new))
        n_beams = len(strong_beam_indices)

        beam_proj = np.zeros((n_beams, n_hkl), dtype=np.complex128)
        beam_proj[np.arange(n_beams), strong_beam_indices] = 1.0 + 0j

        ug_sg = beam_proj @ ug_matrix @ beam_proj.T
        ug_sg = 2.0 * np.pi**2 * ug_sg / big_k_mag
        ug_sg[np.arange(n_beams),
              np.arange(n_beams)] = s_g_pix[strong_beam_indices]

        norm_fac = np.sqrt(1.0 + g_dot_norm[strong_beam_indices] / k_dot_n_pix)
        structure_mat = ug_sg / np.outer(norm_fac, norm_fac)

        gamma, eigenvecs = eig(structure_mat)

        # ---- wave_functions ------------------------------------------------
        psi0 = np.zeros(n_beams, dtype=np.complex128)
        psi0[0] = 1.0 + 0j

        inv_m_ii = norm_fac
        m_ii = 1.0 / inv_m_ii
        u = inv_m_ii * psi0
        y = solve(eigenvecs, u)
        phase_fwd = np.exp(1j * np.outer(gamma, thickness))
        z = eigenvecs @ (y[:, None] * phase_fwd)
        wave_funct = (m_ii[:, None] * z).T  # [n_thickness, n_beams]

        row_intensity[pix_y] = np.abs(wave_funct[:, :n_out])**2

        # ---- Analytical coordinate gradients --------------------------------
        # LU factorisation of V — reused for every parameter at this pixel.
        lu_piv = lu_factor(eigenvecs)

        # Miller index differences for all strong-beam pairs, all 3 coordinates
        # h_diff[m, n, alpha] = hkl_indices[gm, alpha] - hkl_indices[gn, alpha]
        h_strong = hkl_int[strong_beam_indices, :]  # [n_beams, 3]
        h_diff = (h_strong[:, None, :] -
                  h_strong[None, :, :])  # [n_beams, n_beams, 3]

        for a_idx, atom_ug_full in enumerate(atom_ug_list):

            # Project per-atom U onto the strong-beam subspace
            # atom_ug_strong[m, n] = U_{j, gm-gn}
            atom_ug_strong = beam_proj @ atom_ug_full @ beam_proj.T

            # Common factor for all 3 coordinates:
            # ∂A_mn/∂x_α = -(4π³i/K) · h_diff[m,n,α] · atom_ug_strong[m,n]
            #               / (norm_fac_m · norm_fac_n)
            # Factor out the scalar and obliquity terms:
            # NB: NEGATIVE sign from pylix phase convention exp(-i g·r)
            base = (-4.0 * np.pi**3 * 1j / big_k_mag) * atom_ug_strong / \
                np.outer(norm_fac, norm_fac)  # [n_beams, n_beams]

            # Stack ∂A/∂x, ∂A/∂y, ∂A/∂z into a [3, n_beams, n_beams] array
            # h_diff has shape [n_beams, n_beams, 3]; transpose to [3, n, n]
            dA_all = h_diff.transpose(2, 0, 1) * base[None, :, :]
            # Diagonal of ∂A/∂x_{α,j} is 0 (S_g has no coordinate dependence)
            dA_all[:, np.arange(n_beams), np.arange(n_beams)] = 0.0

            # G_all[alpha] = V⁻¹ (∂A/∂x_alpha) V  for all 3 coords at once.
            # Solve: eigenvecs @ G_all = dA_all @ eigenvecs
            # Reshape (3, n, n) → (3n, n), solve, reshape back.
            rhs = dA_all @ eigenvecs  # [3, n_beams, n_beams]
            G_all = lu_solve(
                lu_piv,
                rhs.reshape(3 * n_beams, n_beams)
            ).reshape(3, n_beams, n_beams)  # [3, n_beams, n_beams]

            # For each t: X_p = G ⊙ F, dpsi = M V X_p y, dI = 2Re(ψ* dpsi)
            for t_idx, t in enumerate(thickness):
                F = f_matrix(gamma, t)  # [n_beams, n_beams]

                # X_p y for all 3 coords: (G_all * F[None]) @ y → [3, n_beams]
                Xpy_all = (G_all * F[None, :, :]) @ y

                # dpsi = M V X_p y = m_ii * (V @ Xpy) — vectorised over coords
                # (Xpy_all @ V^T)[c, i] = Σ_j Xpy_all[c,j] * V[i,j]  ✓
                dpsi_all = m_ii[None, :] * (Xpy_all @ eigenvecs.T)  # [3, n_beams]

                # ∂I_g/∂x_α = 2 Re(ψ_g* · ∂ψ_g/∂x_α)
                row_gradient[pix_y,
                             a_idx*3:a_idx*3+3,
                             t_idx] = 2.0 * np.real(
                    np.conj(wave_funct[t_idx, None, :n_out])
                    * dpsi_all[:, :n_out])

    return row_intensity, row_gradient
