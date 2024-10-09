import numpy as np

def hkl_make(ar_vec_m, br_vec_m, cr_vec_m, lattice_type, min_reflection_pool,
             min_strong_beams, g_limit, electron_wave_vector_magnitude):
    """
    Generates Miller indices that satisfy the selection rules for a given lattice type and are
    close to the Bragg condition, using efficient vectorized operations.

    Parameters:
    ar_vec_m (ndarray): Reciprocal lattice vector a
    br_vec_m (ndarray): Reciprocal lattice vector b
    cr_vec_m (ndarray): Reciprocal lattice vector c
    lattice_type (str): Lattice type (from space group name)
    min_strong_beams (int): Minimum number of strong beams required
    g_limit (float): Upper limit for g-vectors magnitude
    electron_wave_vector_magnitude (float): Magnitude of the electron wave vector

    Returns:
    ndarray: Array of selected Miller indices satisfying the selection rules and Bragg condition
    """
    # k-vector for the incident beam (k is along z in the microscope frame)
    k_vector = np.array([0.0, 0.0, electron_wave_vector_magnitude])

    # Calculate the magnitude of reciprocal lattice basis vectors
    ar_mag = np.linalg.norm(ar_vec_m)
    br_mag = np.linalg.norm(br_vec_m)
    cr_mag = np.linalg.norm(cr_vec_m)

    # Determine shell size (smallest basis vector)
    shell = min(ar_mag, br_mag, cr_mag)

    if g_limit < 1e-10:
        # Use default value and set min_reflection_pool as cutoff
        g_limit = 10.0 * 2 * np.pi
    else:
        # Make min_reflection_pool large and use g_limit as the cutoff
        min_reflection_pool = 6666

    # Maximum indices to consider based on g_limit
    max_h = int(np.ceil(g_limit / ar_mag))
    max_k = int(np.ceil(g_limit / br_mag))
    max_l = int(np.ceil(g_limit / cr_mag))

    # Generate grid of h, k, l values
    h_range = np.arange(-max_h, max_h + 1)
    k_range = np.arange(-max_k, max_k + 1)
    l_range = np.arange(-max_l, max_l + 1)
    h_, k_, l_ = np.meshgrid(h_range, k_range, l_range, indexing='ij')
    hkl_pool = np.stack((h_.ravel(), k_.ravel(), l_.ravel()), axis=-1)

    # Apply selection rules using a mask
    if lattice_type == "F":
        mask = (hkl_pool[:, 0] + hkl_pool[:, 1]) % 2 == 0
        mask &= (hkl_pool[:, 1] + hkl_pool[:, 2]) % 2 == 0
        mask &= (hkl_pool[:, 2] + hkl_pool[:, 0]) % 2 == 0
    elif lattice_type == "I":
        mask = (hkl_pool.sum(axis=1) % 2) == 0
    elif lattice_type == "A":
        mask = (hkl_pool[:, 1] + hkl_pool[:, 2]) % 2 == 0
    elif lattice_type == "B":
        mask = (hkl_pool[:, 0] + hkl_pool[:, 2]) % 2 == 0
    elif lattice_type == "C":
        mask = (hkl_pool[:, 0] + hkl_pool[:, 1]) % 2 == 0
    elif lattice_type == "R":
        mask = (hkl_pool[:, 0] - hkl_pool[:, 1] + hkl_pool[:, 2]) % 3 == 0
    elif lattice_type == "V":
        mask = (-hkl_pool[:, 0] + hkl_pool[:, 1] + hkl_pool[:, 2]) % 3 == 0
    elif lattice_type == "P":
        mask = np.ones(len(hkl_pool), dtype=bool)
    else:
        raise ValueError("Space group not recognised")
    hkl_pool = hkl_pool[mask]

    # Calculate g-vectors and their magnitudes
    g_vectors = hkl_pool @ np.array([ar_vec_m, br_vec_m, cr_vec_m])
    g_magnitudes = np.linalg.norm(g_vectors, axis=1) + 1.0e-10

    # Calculate deviation from Bragg condition
    g_plus_k = g_vectors + k_vector
    deviations = np.abs(electron_wave_vector_magnitude -
                        np.linalg.norm(g_plus_k, axis=1)) / g_magnitudes

    # we choose reflections by building up masks until we have enough
    # (limited by g_limit or min_reflection_pool, whichever is smallest)

    # first shell
    lnd = 1.0  # Number of the shell
    current_g_limit = shell*lnd
    mask = (g_magnitudes <= current_g_limit) & (deviations < 0.08)
    hkl = hkl_pool[mask]
    # expand until we have enough
    while (len(hkl) < min_reflection_pool) and (lnd * shell < g_limit):
        lnd += 1.0
        current_g_limit = shell*lnd
        mask = (g_magnitudes <= current_g_limit) & (deviations < 0.08)
        hkl = hkl_pool[mask]

    g_vectors = g_vectors[mask]
    g_magnitudes = g_magnitudes[mask]
    # clean up
    del hkl_pool
    del g_plus_k
    del deviations

    # Check if enough beams are present
    if len(hkl) < min_strong_beams:
        raise ValueError("Beam pool is too small, please increase g_limit!")

    return hkl, g_vectors, g_magnitudes
