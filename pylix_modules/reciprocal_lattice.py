import numpy as np


def reciprocal_lattice(cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                       cell_gamma, diffraction_flag, space_group_name,
                       x_dir_c, z_dir_c, norm_dir_c):
    """
    Produces reciprocal lattice vectors and related parameters

    Parameters:
    diffraction_flag : int
        Flag to indicate whether diffraction calculations are required.
    space_group_name : str
        Space group name (will be modified in some cases).
    cell_alpha, cell_beta, cell_gamma : float
        Lattice angles in radians.
    cell_a, cell_b, cell_c : float
        Lattice lengths in Angstroms.
    x_dir_c, z_dir_c : ndarray
        Reciprocal lattice vectors that define the x-axis of the diffraction
        pattern and the beam direction.
    norm_dir_c : ndarray
        Normal direction in the crystal reference frame.
    norm_dir_m : ndarray
        Normal direction in the microscope reference frame.
    """

    tiny = 1e-10
    two_pi = 2.0 * np.pi

    # Direct lattice vectors in an orthogonal reference frame, Angstrom units
    a_vec_o = np.array([cell_a, 0.0, 0.0])
    b_vec_o = np.array([cell_b * np.cos(cell_gamma),
                        cell_b * np.sin(cell_gamma), 0.0])
    c_vec_o = np.array([
        cell_c * np.cos(cell_beta),
        cell_c * (np.cos(cell_alpha) - np.cos(cell_beta) *
                  np.cos(cell_gamma)) / np.sin(cell_gamma),
        cell_c * np.sqrt(1.0 - np.cos(cell_alpha)**2 -
                         np.cos(cell_beta)**2 - np.cos(cell_gamma)**2 +
                         2.0 * np.cos(cell_alpha) *
                         np.cos(cell_beta) * np.cos(cell_gamma)) /
        np.sin(cell_gamma)])

    # Some checks for rhombohedral cells
    if diffraction_flag == 0:
        r_test = (
            np.dot(a_vec_o / np.dot(a_vec_o, a_vec_o),
                   b_vec_o / np.dot(b_vec_o, b_vec_o)) *
            np.dot(b_vec_o / np.dot(b_vec_o, b_vec_o),
                   c_vec_o / np.dot(c_vec_o, c_vec_o)) *
            np.dot(c_vec_o / np.dot(c_vec_o, c_vec_o),
                   a_vec_o / np.dot(a_vec_o, a_vec_o))
        )
        if 'r' in space_group_name.lower():
            if abs(r_test) < tiny:
                space_group_name = "V"
                # Assume the crystal is Obverse
            else:
                space_group_name = "P"
                # Primitive setting (Rhombohedral axes)

    # Reciprocal lattice vectors: orthogonal frame in 1/Angstrom units
    ar_vec_o = (two_pi * np.cross(b_vec_o, c_vec_o) /
                np.dot(b_vec_o, np.cross(c_vec_o, a_vec_o)))
    br_vec_o = (two_pi * np.cross(c_vec_o, a_vec_o) /
                np.dot(c_vec_o, np.cross(a_vec_o, b_vec_o)))
    cr_vec_o = (two_pi * np.cross(a_vec_o, b_vec_o) /
                np.dot(a_vec_o, np.cross(b_vec_o, c_vec_o)))

    ar_vec_o[np.abs(ar_vec_o) < tiny] = 0.0
    br_vec_o[np.abs(br_vec_o) < tiny] = 0.0
    cr_vec_o[np.abs(cr_vec_o) < tiny] = 0.0

    # Transformation matrix from crystal to orthogonal reference frame
    t_mat_c2o = np.column_stack((a_vec_o, b_vec_o, c_vec_o))

    # Unit reciprocal lattice vectors in orthogonal frame
    x_dir_o = np.dot(x_dir_c, np.column_stack((ar_vec_o, br_vec_o, cr_vec_o)))
    x_dir_o /= np.linalg.norm(x_dir_o)
    z_dir_o = np.dot(z_dir_c, t_mat_c2o)
    z_dir_o /= np.linalg.norm(z_dir_o)
    y_dir_o = np.cross(z_dir_o, x_dir_o)

    # Transformation matrix from orthogonal to microscope reference frame
    t_mat_o2m = np.column_stack((x_dir_o, y_dir_o, z_dir_o))

    # Unit normal to the specimen in microscope frame
    norm_dir_m = np.dot(t_mat_o2m, np.dot(t_mat_c2o, norm_dir_c))
    norm_dir_m /= np.linalg.norm(norm_dir_m)

    # Transform from crystal reference frame to microscope frame
    a_vec_m = np.dot(t_mat_o2m, a_vec_o)
    b_vec_m = np.dot(t_mat_o2m, b_vec_o)
    c_vec_m = np.dot(t_mat_o2m, c_vec_o)

    # Reciprocal lattice vectors: microscope frame in 1/Angstrom units
    ar_vec_m = (two_pi * np.cross(b_vec_m, c_vec_m) /
                np.dot(b_vec_m, np.cross(c_vec_m, a_vec_m)))
    br_vec_m = (two_pi * np.cross(c_vec_m, a_vec_m) /
                np.dot(c_vec_m, np.cross(a_vec_m, b_vec_m)))
    cr_vec_m = (two_pi * np.cross(a_vec_m, b_vec_m) /
                np.dot(a_vec_m, np.cross(b_vec_m, c_vec_m)))

    return a_vec_m, b_vec_m, c_vec_m, ar_vec_m, br_vec_m, cr_vec_m
