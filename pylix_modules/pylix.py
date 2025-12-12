import ast
import re
import sympy as sp
import subprocess
import numpy as np
from scipy.constants import c
from scipy.linalg import eig, inv
from CifFile import CifFile
import struct
from pylix_modules import pylix_dicts as fu
from numba import njit, prange
import math


def read_inp_file(filename):
    """
    Reads in the file felix.inp and assigns values based on text labels.
    Each line in the file should have the format: variable_name = value.
    The order of variables in the file does not matter.

    Parameters:
    filename (str): The path to the input file.
    Returns:
    inp_dict: A dictionary with variable names and values.
    """

    # Dictionary to store the variable values
    inp_dict = {}

    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or '=' not in line:
                    continue

                # Split the line into variable name and value
                var_name, var_value = line.split('=', 1)
                var_name = var_name.strip()
                var_value = var_value.strip()
                inp_dict[var_name] = ast.literal_eval(var_value)

    except FileNotFoundError:
        raise ValueError(f"File not found: {filename}")
    except IOError as e:
        raise ValueError(f"IO error ({e}) reading file: {filename}")

    return inp_dict


def read_hkl_file(filename):
    """
    Reads in the file felix.hkl and returns the list of reflexions to output

    Parameters:
    filename (str): The path to the input file.
    Returns:
    inp_dict: A dictionary with variable names and values.
    """
    input_hkls = []
    i_obs = []
    sigma_obs = []

    cRED = False  # cRED data has Iobs & sigma as extra columns
    with open(filename, 'r') as file:
        # Check the first line to decide the file structure
        first_line = file.readline().strip()
        if ',' in first_line.split(']')[-1]:  # comma after ]
            cRED = True
        # Go back to the beginning of the file
        file.seek(0)

        for line in file:
            # Remove brackets and split the line
            line = line.strip().replace('[', '').replace(']', '').replace(',', '')
            if line:  # skip blank lines
                parts = line.split()

                # Extract Miller indices
                g_ = list(parts[0:3])
                input_hkls.append(g_)

                if cRED:  # Extract g, i_obs and sigma_obs
                    intensity = float(parts[1].replace(',', ''))
                    sigma = float(parts[2])
                    i_obs.append(intensity)
                    sigma_obs.append(sigma)

    # Convert lists to numpy arrays
    input_hkls = np.array(input_hkls, dtype=int)
    i_obs = np.array(i_obs) if cRED else None
    sigma_obs = np.array(sigma_obs) if cRED else None

    return input_hkls, i_obs, sigma_obs


def extract_cif_parameter(item):
    """
    Parses a value string with uncertainty, e.g., '8.6754(3)',
    and returns the value and uncertainty as floats.

    Args:
    - item (str): The string containing the value with uncertainty.

    Returns:
    - tuple: A tuple containing the value and the uncertainty as floats.
    """

    # Check if the value contains an uncertainty part (i.e., contains '(')
    if '(' in item and ')' in item:
        value_str, pm_str = item.split('(')
        value_str = value_str.strip()  # Remove any extra spaces
        pm_str = pm_str.strip(')')  # Remove the closing parenthesis
    else:
        # If no uncertainty is provided, return zero pm
        value_str = item
        pm_str = None
    if value_str == '.':
        value = 0.0
    else:
        value = float(value_str)

    # uncertainty
    if pm_str:
        # Number of decimal places in the main value string
        decimal_places = value_str[::-1].find('.')
        pm = int(pm_str) * (10 ** -decimal_places)
    else:
        pm = 0

    return value, pm


def dlst(param):
    # sometimes ReadCif returns lists (why?), this extracts the single value
    while isinstance(param, (list, tuple)):
        # print(param)
        param = param[0]
    return float(param)


def read_cif(filename):
    """
    Extracts and returns the names and values of specified items in a
    CIF file, names omit the initial underscore, values in a dictionary

    Args: filename (str): The path to the CIF file

    Returns: cif_dict (dictionary): variable names and their values, as a
        tuple if numeric with a given uncertainty
    """
    # the dictionary to be returned
    cif_dict = {}

    cf = CifFile(filename)
    data_block_names = cf.keys()
    data_block = cf[data_block_names[0]]

    # NB cif keys are converted to all lower case
    # numeric items to be returned
    numeric_items = [
        "_symmetry_int_tables_number",
        "_space_group_it_number",
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
        "_cell_volume",
        "_atom_type_oxidation_number",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_b_iso_or_equiv",
        "_atom_site_u_iso_or_equiv",
        "_atom_site_occupancy",
        "_atom_site_aniso_u_11",
        "_atom_site_aniso_u_22",
        "_atom_site_aniso_u_33",
        "_atom_site_aniso_u_12",
        "_atom_site_aniso_u_13",
        "_atom_site_aniso_u_23"]
    # string items to be returned
    string_items = [
        "_space_group_name_h-m_alt",
        "_symmetry_space_group_name_h-m",
        "_symmetry_space_group_name_hall",
        "_chemical_formula_structural",
        "_chemical_formula_iupac",
        "_chemical_formula_sum",
        "_symmetry_equiv_pos_as_xyz",
        "_space_group_symop_operation_xyz",
        "_atom_type_symbol",
        "_atom_site_wyckoff_symbol",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_aniso_label",
        "_atom_site_aniso_type_symbol"]

    # extract numeric values and uncertainties, add them to the dictionary
    for item in numeric_items:
        if item in data_block.keys():
            value_list = []
            var_name = item.lstrip('_')
            param = data_block[item]
            # is it a single value
            if isinstance(param, str):
                cif_dict[var_name] = extract_cif_parameter(param)
            # or is it a list
            elif isinstance(param, list):
                for val in param:
                    # issue where first call comes back as a nested list
                    # can't see why that happens!
                    data = extract_cif_parameter(val)
                    value_list.append(data)
                cif_dict[var_name] = value_list

    # extract strings, add them to the dictionary
    for item in string_items:
        if item in data_block.keys():
            var_name = item.lstrip('_')
            param = data_block[item]
            # is it a single value
            if isinstance(param, str):
                cif_dict[var_name] = param
            # or is it a list
            elif isinstance(param, list):
                value_list = ([])
                for val in param:
                    value_list.append(val)
                cif_dict[var_name] = value_list

    # modify to remove invalid characters
    original_keys = list(cif_dict.keys())
    for key in original_keys:
        new_key = key.replace('-', '_')
        if new_key != key:
            cif_dict[new_key] = cif_dict[key]
            del cif_dict[key]

    # tidy up cell parameters that have been read as lists not tuples
    # I feel there should be a more elegant way of fixing this issue!
    if isinstance(cif_dict['cell_length_a'], list):
        cif_dict['cell_length_a'] = cif_dict['cell_length_a'][0]
    if isinstance(cif_dict['cell_length_b'], list):
        cif_dict['cell_length_b'] = cif_dict['cell_length_b'][0]
    if isinstance(cif_dict['cell_length_a'], list):
        cif_dict['cell_length_c'] = cif_dict['cell_length_c'][0]

    # tidy up chemical formula
    if "chemical_formula_structural" in cif_dict:
        cif_dict['chemical_formula_structural'] = \
            re.sub(r'(?<!\d)1(?!\d)', '',
                   cif_dict['chemical_formula_structural'].replace(' ', ''))
    if "chemical_formula_sum" in cif_dict:  # preferred
        cif_dict['chemical_formula_sum'] = \
            re.sub(r'(?<!\d)1(?!\d)', '',
                   cif_dict['chemical_formula_sum'].replace(' ', ''))

    return cif_dict


def symop_convert(symop_xyz):
    # Converts symmetry operation xyz form into matrix+vector form
    symmetry_count = len(symop_xyz)
    mat = np.zeros((symmetry_count, 3, 3), dtype="float")
    vec = np.zeros((symmetry_count, 3), dtype="float")
    x, y, z = sp.symbols("x y z")
    vars = [x, y, z]
    # we expect comma-delimited symmetry operations
    for i in range(symmetry_count):
        symop = symop_xyz[i]
        # Remove any numbers, extra spaces, and quotation marks
        # remove leading number/label if present
        tokens = symop.strip().split(maxsplit=1)
        if tokens[0].replace("-", "").isdigit() and len(tokens) > 1:
            symop = tokens[1]  # drop the leading number
        # split into 3 parts
        parts = [sp.sympify(e.strip()) for e in symop.split(",")]
        for j, pt in enumerate(parts):
            symop_expanded = sp.expand(pt)
            # translation part
            vec[i, j] = symop_expanded.subs({x:0, y:0, z:0})
            # matrix part
            for k, var in enumerate(vars):
                # print(symop_expanded.coeff(var))
                mat[i, j, k] = symop_expanded.coeff(var)

    return mat, vec


def unique_atom_positions(symmetry_matrix, symmetry_vector, basis_atom_label, basis_atom_type_label,
                          basis_atom_name, basis_atom_position, basis_u_ij,
                          basis_occupancy, basis_pv, basis_kappa, debug):
    """
    Fills the unit cell by applying symmetry operations to the basis

    Parameters:
    symmetry_matrix float(n_symmetry_operations x 3x3): n_symmetry_operations
    symmetry_vector float(n_symmetry_operations x 3):  associated translations
    basis_atom_label (str): a label for each basis atom
    basis_atom_name (str): element symbol for each basis atom
    basis_atom_position float(n_basis_atoms x 3): fractional coordinates
    basis_u_ij float(n_basis_atoms, 3, 3): anisotropic displacement parameter tensor for each basis atom
    basis_occupancy float(n_basis_atoms): occupancy for each basis atom

    Returns:
    atom_position, atom_label, atom_name, B_iso, occupancy
    """
    if debug:
        np.set_printoptions(precision=5, suppress=True)
        for i in range(3):
            print(f"Basis anisotropic u_ij [{i}]")
            print(f"{basis_u_ij[i, :5, :5]}")

    # tolerance in fractional coordinates to consider atoms to be the same
    tol = 0.000001
    # Determine the size of the all_atom_position array
    n_symmetry_operations = symmetry_vector.shape[0]
    n_basis_atoms = basis_atom_position.shape[0]
    total_atoms = n_symmetry_operations * n_basis_atoms

    # Initialize arrays to store all atom positions, including duplicates
    all_atom_label = np.tile(basis_atom_label, n_symmetry_operations)
    # print (basis_atom_type_label)

    all_atom_type_label = np.tile(basis_atom_type_label, n_symmetry_operations)

    all_atom_name = np.tile(basis_atom_name, n_symmetry_operations)
    # print(all_atom_name)
    all_occupancy = np.tile(basis_occupancy, n_symmetry_operations)
    all_kappa = np.tile(basis_kappa, n_symmetry_operations)
    all_pv = np.tile(basis_pv, n_symmetry_operations)

    # Generate all equivalent positions by applying symmetry
    symmetry_applied = \
        np.einsum('ijk,lk->ilj', symmetry_matrix, basis_atom_position) +\
        symmetry_vector[:, np.newaxis, :]
    all_atom_position = symmetry_applied.reshape(total_atoms, 3)

    # Anisotropic Displacement Parameters, ADPs
    # apply symops to u_ij's, size [n_symmetry_operations, n_basis_atoms, 3, 3]
    tmp = np.matmul(symmetry_matrix[:, None], basis_u_ij[None, :])
    # array of inverse operations, size [n_symmetry_operations, 3, 3]
    # NB we can't use the transpose as symops are not always orthonormal
    Minv = np.linalg.inv(symmetry_matrix)
    # finsh the calculation, size [n_symmetry_operations, n_basis_atoms, 3, 3]
    all_u_ij = np.matmul(tmp, Minv[:, None])
    # reshape, size [n_symmetry_operations*n_basis_atoms, 3, 3]
    all_u_ij = all_u_ij.reshape(-1, 3, 3)

    # Normalize positions to be within [0, 1]
    all_atom_position %= 1.0
    # make small values precisely zero
    all_atom_position[np.abs(all_atom_position) < tol] = 0.0

    # Reduce to the set of unique fractional atomic positions using tol
    dist_matrix = np.linalg.norm(all_atom_position[:, np.newaxis, :] -
                                 all_atom_position[np.newaxis, :, :], axis=-1)
    unique_mask = np.ones(len(all_atom_position), dtype=bool)
    i = []  # indices of unique atom positions
    for j in range(total_atoms):
        if unique_mask[j]:  # If this point is still unique
            i.append(j)
            # Mark all points within tol as not unique
            unique_mask &= (dist_matrix[j] > tol)

    # Apply the same reduction to the labels, names, occupancies, and B_iso
    atom_position = all_atom_position[i]
    atom_label = all_atom_label[i]
    atom_type = all_atom_type_label[i]
    atom_name = all_atom_name[i]
    occupancy = all_occupancy[i]
    u_ij = all_u_ij[i]
    kappa = all_kappa[i]
    pv = all_pv[i]
    if debug:
        np.set_printoptions(precision=5, suppress=True)
        for i in range(3):
            print(f"Anisotropic u_ij [{i}]")
            print(f"{u_ij[i, :5, :5]}")

    return atom_position, atom_label,atom_type, atom_name, u_ij, occupancy, pv, kappa


def reference_frames(cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma,
                     space_group, x_dir_c, z_dir_c, norm_dir_c, debug):
    """
    Produces reciprocal lattice vectors and related parameters

    Parameters:
    space_group : str
        Space group name (will be modified in some cases).
    cell_alpha, cell_beta, cell_gamma : float
        Lattice angles in radians.
    cell_a, cell_b, cell_c : float
        Lattice lengths in Angstroms.
    z_dir_c : ndarray
        Direct lattice vector that defines the beam direction.
    x_dir_c : ndarray
        Reciprocal lattice vector that defines the x-axis of the diffraction
        pattern.
    norm_dir_c : ndarray
        Normal direction in the crystal reference frame.
    """

    tiny = 1e-10

    # Direct lattice vectors in an orthogonal reference frame, Angstrom units
    a_vec_o = np.array([cell_a, 0.0, 0.0])  # x_o is // to a
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
    # if diffraction_flag == 0:
    #     r_test = (
    #         np.dot(a_vec_o / np.dot(a_vec_o, a_vec_o),
    #                b_vec_o / np.dot(b_vec_o, b_vec_o)) *
    #         np.dot(b_vec_o / np.dot(b_vec_o, b_vec_o),
    #                c_vec_o / np.dot(c_vec_o, c_vec_o)) *
    #         np.dot(c_vec_o / np.dot(c_vec_o, c_vec_o),
    #                a_vec_o / np.dot(a_vec_o, a_vec_o))
    #     )
    #     if 'r' in space_group.lower():
    #         if abs(r_test) < tiny:
    #             space_group = "V"
    #             # Assume the crystal is Obverse
    #         else:
    #             space_group = "P"
    #             # Primitive setting (Rhombohedral axes)

    # Reciprocal lattice vectors: orthogonal frame in 1/Angstrom units
    ar_vec_o = (2.0*np.pi * np.cross(b_vec_o, c_vec_o) /
                np.dot(b_vec_o, np.cross(c_vec_o, a_vec_o)))
    br_vec_o = (2.0*np.pi * np.cross(c_vec_o, a_vec_o) /
                np.dot(c_vec_o, np.cross(a_vec_o, b_vec_o)))
    cr_vec_o = (2.0*np.pi * np.cross(a_vec_o, b_vec_o) /
                np.dot(a_vec_o, np.cross(b_vec_o, c_vec_o)))
    # not strictly needed but aren't those e-17 things annoying
    ar_vec_o[np.abs(ar_vec_o) < tiny] = 0.0
    br_vec_o[np.abs(br_vec_o) < tiny] = 0.0
    cr_vec_o[np.abs(cr_vec_o) < tiny] = 0.0

    # Transformation matrix from crystal to orthogonal reference frame
    t_mat_c2o = np.column_stack((a_vec_o, b_vec_o, c_vec_o))
    # And the same for reciprocal frames
    t_mat_cr2or = np.column_stack((ar_vec_o, br_vec_o, cr_vec_o))

    # Unit reciprocal lattice vectors in orthogonal frame
    x_dir_o = t_mat_cr2or @ x_dir_c
    x_dir_o /= np.linalg.norm(x_dir_o)
    z_dir_o = t_mat_c2o @ z_dir_c
    z_dir_o /= np.linalg.norm(z_dir_o)
    y_dir_o = np.cross(z_dir_o, x_dir_o)

    # Transformation matrix from orthogonal to microscope reference frame
    t_mat_o2m = np.column_stack((x_dir_o, y_dir_o, z_dir_o)).T

    # Unit normal to the specimen in microscope frame
    norm_dir_m = t_mat_o2m @ t_mat_c2o @ norm_dir_c
    norm_dir_m /= np.linalg.norm(norm_dir_m)

    # Transform from crystal reference frame to microscope frame
    a_vec_m = t_mat_o2m @ a_vec_o
    b_vec_m = t_mat_o2m @ b_vec_o
    c_vec_m = t_mat_o2m @ c_vec_o

    # Reciprocal lattice vectors: microscope frame in 1/Angstrom units
    ar_vec_m = (2.0*np.pi * np.cross(b_vec_m, c_vec_m) /
                np.dot(a_vec_m, np.cross(b_vec_m, c_vec_m)))
    br_vec_m = (2.0*np.pi * np.cross(c_vec_m, a_vec_m) /
                np.dot(b_vec_m, np.cross(c_vec_m, a_vec_m)))
    cr_vec_m = (2.0*np.pi * np.cross(a_vec_m, b_vec_m) /
                np.dot(c_vec_m, np.cross(a_vec_m, b_vec_m)))

    # Output to check
    if debug:
        print(" ")
        np.set_printoptions(precision=5, suppress=True)
        print(f"a = {cell_a}, b = {cell_b}, c = {cell_c}")
        print(f"alpha = {cell_alpha*180.0/np.pi:.2f}, beta = {cell_beta*180.0/np.pi:.2f}, gamma = {cell_gamma*180.0/np.pi:.2f}")
        print(f"X = {x_dir_c} (reciprocal space)")
        print(f"Z = {z_dir_c} (direct space)")
        print(" ")
        print("Transformation crystal to orthogonal (O) frame:")
        print(t_mat_c2o)
        print(t_mat_cr2or)
        print("O frame:")
        print(f" a = {a_vec_o}")
        print(f" b = {b_vec_o}")
        print(f" c = {c_vec_o}")
        print(f" a* = {ar_vec_o}")
        print(f" b* = {br_vec_o}")
        print(f" c* = {cr_vec_o}")
        print(" ")
        print(f" X = {x_dir_o}")
        print(f" Y = {y_dir_o}")
        print(f" Z = {z_dir_o}")
        print(" ")
        print("Transformation orthogonal to microscope frame:")
        print(t_mat_o2m)
        print("Microscope frame:")
        print(f" a = {a_vec_m}")
        print(f" b = {b_vec_m}")
        print(f" c = {c_vec_m}")
        print(f" a* = {ar_vec_m}")
        print(f" b* = {br_vec_m}")
        print(f" c* = {cr_vec_m}")
        print(f"Specimen surface normal = {norm_dir_m}")

    return a_vec_m, b_vec_m, c_vec_m, ar_vec_m, br_vec_m, cr_vec_m, norm_dir_m, t_mat_o2m, t_mat_c2o


def change_origin(space_group, basis_atom_position, basis_wyckoff):
    """
    Change to the preferred origin for a given space group.

    Parameters:
    space_group (int): The space group number.

    Returns:
    basis_atom_position (ndarray): the updated basis atom fractional coords.
    """
    change_flag = 0
    n_basis_atoms = basis_atom_position.shape[0]  # Number of basis atoms

    # Only needed for space group #142 (I41/acd) so far
    # will be needed for others with origin choices!!!
    if space_group == 142:
        # Change from choice 1 (origin -4 at [0,0,0])
        # to choice 2 (origin -1 at [0,0,0])
        # Look for an 'a' site incompatible with choice 2
        for i in range(n_basis_atoms):
            # 'a' is -4 at [000],[0,1/2,1/2],[0,1/2,1/4],[1/2,0,1/4] in 1
            # and [0,1/4,3/8], [0,3/4,5/8], [1/2,1/4,5/8], [1/2,3/4,5/8] in 2
            if basis_wyckoff[i] == 'a':
                # Check for origin 1
                # by multiplying by 4 and checking if it is an integer
                if np.mod(4 * basis_atom_position[i, 2], 1.0) < 1e-10:
                    change_flag = 1
        # add [0,1/4,3/8] if change_flag is set
        if change_flag == 1:
            basis_atom_position[:, 1] = np.mod(basis_atom_position[:, 1]
                                               + 0.25, 1.0)
            basis_atom_position[:, 2] = np.mod(basis_atom_position[:, 2]
                                               + 0.375, 1.0)
    return basis_atom_position


def preferred_basis(space_group, basis_atom_position, basis_wyckoff):

    n_basis_atoms = len(basis_atom_position)
    basis_atom_position = change_origin(space_group, basis_atom_position,
                                        basis_wyckoff)

    # Loop over basis atoms
    for i in range(n_basis_atoms):
        wyckoff_symbol = basis_wyckoff[i]

        if space_group == 1:  # P1
            if wyckoff_symbol != 'a':
                raise ValueError("Wyckoff Symbol for P1 not recognised")
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
        elif space_group == 15:  # C2/c
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for C 2/c not recognised")
# 16
# 17
# 18
# 19
# 20
# 21
# 22
# 23
# 24
# 25
# 26
# 27
# 28
# 29
# 30
# 31
# 32
# 33
# 34
# 35
        elif space_group == 36:  # C m c 21
            if wyckoff_symbol in ['a', 'b']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for C m c 21 not recognised")
# 37
# 38
# 39
# 40
# 41
# 42
# 43
# 44
# 45
# 46
# 47
# 48
# 49
# 50
# 51
# 52
# 53
# 54
# 55
# 56
# 57
# 58
# 59
# 60
# 61
# 62
        elif space_group == 63:  # C m c m
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for C m c m not recognised")
        elif space_group == 64:  # C m c a
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for C m c a not recognised")
# 65
# 66
# 67
        elif space_group == 68:  # C c c a
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for Ccca not recognised")
# 69
# 70
# 71
# 72
# 73
# 74
# 75
# 76
# 77
# 78
# 79
# 80
# 81
# 82
# 83
# 84
# 85
# 86
# 87
# 88
# 89
# 90
# 91
# 92
# 93
# 94
# 95
# 96
# 97
# 98
        elif space_group == 99:  # P 4 m m
            if wyckoff_symbol in ['a', 'b', 'c', 'g']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'd':
                # Change equivalent coordinate [x,-x,z] to [x,x,z]
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'e':
                # Change equivalent coordinate [0,x,z] to [x,0,z]
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'f':
                # Change equivalent coordinate [1/2,x,z] to [x,1/2,z]
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            else:
                raise ValueError("Wyckoff Symbol for P 4 m m not recognised")
# 100
# 101
# 102
# 103
# 104
# 105
# 106
# 107
# 108
# 109
# 110
# 111
# 112
# 113
# 114
# 115
# 116
# 117
# 118
# 119
# 120
# 121
# 122
# 123
# 124
# 125
# 126
# 127
# 128
# 129
# 130
# 131
# 132
# 133
# 134
# 135
# 136
# 137
# 138
        elif space_group == 139:  # I4/m m m
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'l', 'o']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'h':
                # Change to [x,x,0]
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'i':
                # Change equivalent coordinate [0,y,0]
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'j':
                # Change equivalent coordinate [1/2,y,0]
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            elif wyckoff_symbol == 'k':
                # Change to [x,1/2+x,1/4]
                basis_atom_position[i, 1] = basis_atom_position[i, 0] + 0.5
            elif wyckoff_symbol == 'm':
                # Change to [x,x,z]
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'n':
                # Change equivalent coordinate [0,y,z]
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            else:
                raise ValueError("Wyckoff Symbol for I4/m m m not recognised")
# 140
# 141
        elif space_group == 142:  # I41/acd
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'g']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'e':
                if abs(basis_atom_position[i, 2]) < 1e-10:
                    if abs(basis_atom_position[i, 0] - 0.25) < 1e-10:
                        basis_atom_position[i, 0] = 0.25 - basis_atom_position[i, 1]
                        basis_atom_position[i, 1] = 0.0
                        basis_atom_position[i, 2] = 0.25
                    elif abs(basis_atom_position[i, 0] - 0.75) < 1e-10:
                        basis_atom_position[i, 0] = basis_atom_position[i, 1] - 0.25
                        basis_atom_position[i, 1] = 0.0
                        basis_atom_position[i, 2] = 0.25
            elif wyckoff_symbol == 'f':
                if abs(basis_atom_position[i, 2] - 0.375) < 1e-10:
                    basis_atom_position[i, 1] = basis_atom_position[i, 0] + 0.25
                    basis_atom_position[i, 2] = 0.125
                elif abs(basis_atom_position[i, 2] - 0.625) < 1e-10:
                    basis_atom_position[i, 1] = basis_atom_position[i, 0] + 0.25
                    basis_atom_position[i, 2] = 0.125
            else:
                raise ValueError("Wyckoff Symbol for 142, I41/a c d not recognised")
# 143
# 144
# 145
# 146
# 147
# 148
# 149
# 150
# 151
# 152
# 153
# 154
# 155
# 156
# 157
# 158
# 159
# 160
        elif space_group == 161:  # R 3 c
            if wyckoff_symbol in ['a', 'b']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for R3c not recognised")
# 162
# 163
# 164
# 165
# 166
# 167
# 168
# 169
# 170
# 171
# 172
# 173
# 174
# 175
# 176
# 177
# 178
# 179
# 180
# 181
# 182
# 183
# 184
# 185
# 186
# 187
# 188
# 189
# 190
        elif space_group == 191:  # P 6 m m
            if wyckoff_symbol in ['a', 'd', 'g']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'b':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'c':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'e':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            elif wyckoff_symbol == 'f':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            else:
                raise ValueError("Wyckoff Symbol for P6mm not recognised")
# 192
# 193
        elif space_group == 194:  # P 6/mmm
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e',
                                  'f', 'g', 'h', 'm', 'r']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'i':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'j':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'k':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            elif wyckoff_symbol == 'l':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'n':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'o':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'p':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            elif wyckoff_symbol == 'q':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            else:
                raise ValueError("Wyckoff Symbol for P6/mmm not recognised")
# 195
# 196
# 197
# 198
# 199
# 200
# 201
# 202
# 203
# 204
# 205
# 206
# 207
# 208
# 209
# 210
# 211
# 212
# 213
# 214
# 215
# 216
# 217
# 218
# 219
# 220
        elif space_group == 221:  # Pm-3m
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'm']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'g':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
                basis_atom_position[i, 2] = 0.0
            elif wyckoff_symbol == 'h':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
                basis_atom_position[i, 2] = 0.0
            elif wyckoff_symbol == 'i':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
                basis_atom_position[i, 2] = 0.0
            elif wyckoff_symbol == 'j':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'k':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'l':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            else:
                raise ValueError("Wyckoff Symbol for Pm-3m not recognised")
# 222
# 223
# 224
# 225
# 226
        elif space_group == 227:  # F d -3 m
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e',
                                  'f', 'g', 'h', 'i', 'j']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for Fd-3m not recognised")
# 228
# 229
# 230
        else:
            raise ValueError("Space group not implemented")

        return basis_atom_position


def hkl_make(ar_vec_m, br_vec_m, cr_vec_m, big_k, lattice_type,
             min_reflection_pool, min_strong_beams, g_limit, input_hkls,
             electron_wave_vector_magnitude):
    """
    Generates Miller indices that satisfy the selection rules for a given
    lattice type and are close to the Bragg condition.

    Parameters:
    ar_vec_m (ndarray): Reciprocal lattice vector a
    br_vec_m (ndarray): Reciprocal lattice vector b
    cr_vec_m (ndarray): Reciprocal lattice vector c
    lattice_type (str): Lattice type (from space group name)
    min_strong_beams (int): Minimum number of strong beams required
    g_limit (float): Upper limit for g-vector magnitude
    electron_wave_vector_magnitude (float):

    Returns:
    hkl (ndarray): Miller indices close to the Bragg condition
    g_pool (ndarray): in the microscope reference frame
    g_magnitudes (ndarray): their magnitudes
    """

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
    g_pool = hkl_pool @ np.array([ar_vec_m, br_vec_m, cr_vec_m])
    g_mag = np.linalg.norm(g_pool, axis=1) + 1.0e-12
    sorter = np.argsort(g_mag)
    g_pool = g_pool[sorter]
    hkl_pool = hkl_pool[sorter]
    g_mag = np.linalg.norm(g_pool, axis=1) + 1.0e-12

    # Calculate deviation from Bragg condition
    g_plus_k = g_pool + big_k
    deviations = np.abs(electron_wave_vector_magnitude -
                        np.linalg.norm(g_plus_k, axis=1)) / g_mag
    deviations[0] = 0  # 000 beam
    
    # we choose reflections by increasing the radius of reciprocal space
    # explored until we have enough
    # (limited by g_limit or min_reflection_pool, whichever is smallest)

    # first shell
    lnd = 1.0  # Number of the shell
    current_g_limit = shell*lnd
    mask = (g_mag <= current_g_limit) & (deviations < 0.08)
    hkl = hkl_pool[mask]
    # expand until we have enough
    while (len(hkl) < min_reflection_pool) and (lnd * shell < g_limit):
        lnd += 1.0
        current_g_limit = shell*lnd
        mask = (g_mag <= current_g_limit) & (deviations < 0.08)
        hkl = hkl_pool[mask]
    g_pool = g_pool[mask]
    g_mag = g_mag[mask]

    # Check if enough beams are present
    if len(hkl) < min_strong_beams:
        raise ValueError("Beam pool is too small, please increase g_limit!")

    # Check if required output HKLs are in the hkl list
    g_output = ([])
    for g in input_hkls:
        comparison = np.all(hkl == g, axis=1)
        if not np.any(comparison):
            print(f"Input hkl not found: {g}")
        idx = np.where(comparison)
        if len(idx[0]) > 0:
            g_output.append(idx[0][0])
    g_output = np.append([0],np.array(g_output))

    return hkl, g_pool, g_mag, np.array(g_output)


def Fg_matrix(n_hkl, scatter_factor_method, n_atoms, atom_coordinate,
              atomic_number, occupancy, u_ij, g_matrix, absorption_method,
              absorption_per, electron_velocity, kappas, pv,
              Debye, model, debug):
    """
    Parameters
    ----------
    n_hkl : int, number of hkls
    scatter_factor_method : int, cchoise of scattering factor calculation
    n_atoms : int, number of atoms in the cell
    atom_coordinate : float, fractional atom coordinates, size [n_atoms, 3]
    atomic_number : int, size [n_atoms]
    occupancy : float, size [n_atoms]
    u_ij : float array of ADPs, size [n_atoms,3,3]
    g_matrix : array of g-vectors in the microscope frame, size [n_hkl,n_hkl]
    absorption_method : int, flag for absorption calculation
    absorption_per : float, % absorption, if that method is used
    electron_velocity : float
    kappas : TYPE DESCRIPTION.
    pv : TYPE DESCRIPTION.
    Debye : TYPE DESCRIPTION.
    model : TYPE DESCRIPTION.

    Raises
    ------
    ValueError if no scatter factor method chosen

    Returns
    -------
    Fg_matrix : size [n_hkl,n_hkl]
    """

    # calculate g.r for all g-vectors and atom posns [n_hkl, n_hkl, n_atoms]
    g_dot_r = np.einsum('ijk,lk->ijl', g_matrix, atom_coordinate)
    # exp(i g.r) [n_hkl, n_hkl, n_atoms]
    phase = np.exp(-1j * g_dot_r)

    # NB scattering factor methods accept and return 2D[n_hkl, n_hkl] array of
    # g magnitudes but only one atom type.

    # g-vector magnitudes, size [n_hkl, n_hkl]
    g_magnitude = np.sqrt(np.sum(g_matrix**2, axis=2))

    # anisotropic DP U*g[i]*g[j], size [n_atoms, n_hkl, n_hkl]
    Ugg = np.einsum('ijm, a mn, ij n -> aij', g_matrix, u_ij, g_matrix)
    # equivalent anisotropic B, size [n_atoms, n_hkl, n_hkl]
    B_aniso = np.divide(Ugg, np.square(g_magnitude), out=np.zeros_like(Ugg),
                        where=(g_magnitude != 0)) * 8 * np.pi**2
    if debug:
        np.set_printoptions(precision=3, suppress=True)
        print("g_magnitudes")
        print(g_magnitude[:5, :5])
        print("  ")
        for i in range(3):
            print(f"Anisotropic u_ij*g[i]*g[j] [{i}]")
            print(f"{Ugg[i, :5, :5]}")
        print("  ")
        for i in range(3):
            print(f"Anisotropic B[{i}]")
            print(f"{B_aniso[i, :5, :5]}")
        print("  ")

    Fg_matrix = np.zeros([n_hkl, n_hkl], dtype=np.complex128)
    # scattering factor f_g, size [n_hkl, n_hkl], atom by atom
    for i in range(n_atoms):
        # get the scattering factor
        if scatter_factor_method == 0:
            f_g = f_kirkland(atomic_number[i], g_magnitude)
        elif scatter_factor_method == 1:
            f_g = f_lobato(atomic_number[i], g_magnitude)
        elif scatter_factor_method == 2:
            f_g = f_peng(atomic_number[i], g_magnitude)
        elif scatter_factor_method == 3:
            f_g = f_doyle_turner(atomic_number[i], g_magnitude)
        elif scatter_factor_method == 4:
            print("Calculating scattering factors for atom", i+1, "/", n_atoms)
            f_g = kappa_factors(g_magnitude, atomic_number[i], pv[i], kappas[i])
        else:
            raise ValueError("No scattering factors chosen in felix.inp")

        # get the absorptive scattering factor
        # f_g_prime, size [n_hkl, n_hkl], atom by atom
        # no absorption
        if absorption_method == 0:
            f_g_prime = np.zeros_like(f_g)
        # proportional model
        elif absorption_method == 1:
            f_g_prime = 1j * f_g * absorption_per/100.0
        # Bird & King model, parameterised by Thomas (Acta Cryst 2023)
        elif absorption_method == 2:
            f_g_prime = 1j * f_thomas(g_magnitude, 0.4,
                                      atomic_number[i], electron_velocity)
            # f_g_prime = 1j * f_thomas(g_magnitude, B_aniso[i, :, :],
            #                           atomic_number[i], electron_velocity)
        if debug and i < 4:
            print(f"f_g [{i}]")
            print(f"{f_g[:5, :5]}")
            print("  ")
            print(f"f_g_prime [{i}]")
            print(f"{f_g_prime[:5, :5]}")

        # The Structure Factor Equation
        Fg_matrix = Fg_matrix+((f_g + f_g_prime) * phase[:, :, i] *
                               occupancy[i] *
                               # np.exp(-B_aniso[i, :, :] * (g_magnitude**2) /
                               # (16*np.pi**2)))
                               np.exp(-Ugg[i, :, :] / 2))

    return Fg_matrix


def deviation_parameter(convergence_angle, image_radius, big_k_mag, g_pool,
                        g_pool_mag):
    # for LACBED pattern of size [m, m] and a set of g-vectors [n, 3]
    # this returns a 3D array of deviation parameters [m, m, n]

    # resolution in k-space
    delta_K = 2.0*np.pi * convergence_angle/image_radius

    # pixel grids
    x_pix = np.arange(-image_radius+.5, image_radius+.5)
    y_pix = np.arange(-image_radius+.5, image_radius+.5)
    # k_x and k_y
    k_x, k_y = np.meshgrid(x_pix * delta_K, y_pix * delta_K)

    # k-vector for all pixels k'
    k_z = np.sqrt(big_k_mag**2 - k_x**2 - k_y**2)
    tilted_k = np.stack([k_x, k_y, k_z], axis=-1)

    # g excuding g_pool[0], which is always [000]
    g_pool1 = g_pool[1:]  # shape: (n_hkl-1, 3)
    g_pool_mag1 = g_pool_mag[1:]  # shape: (n_hkl-1)
    # Components of k_0
    k_0_0 = -g_pool_mag1 / 2
    k_0_2 = np.sqrt(big_k_mag**2 - k_0_0**2)
    # Components of k_prime
    k_prime_0 = np.einsum('ijk,lk->ijl', tilted_k, g_pool1) / g_pool_mag1
    k_prime_2 = np.sqrt(big_k_mag**2 - k_prime_0**2)

    k_0_dot_k_prime = (k_0_0[None, None, :] * k_prime_0 +
                       k_0_2[None, None, :] * k_prime_2)
    pm = -np.sign(2*k_prime_0 + g_pool_mag1)
    s_g = pm * np.sqrt(2*abs(big_k_mag**2 - k_0_dot_k_prime)) * \
        (g_pool_mag1/big_k_mag)

    # add in 000
    s_0 = np.expand_dims(np.zeros_like(k_x), axis=-1)
    s_g = np.concatenate([s_0, s_g], axis=-1)  # add 000

    return s_g, tilted_k


def strong_beams(s_g_pix, ug_matrix, min_strong_beams):
    """
    returns a list of strong beams according to their perturbation strength
    NB s_g_pix here is a 1D array of values for a given pixel, and is different
    to the s_g in the main code which is for all pixels & g-vectors

    Perturbation Strength Eq. 8 Zuo Ultramicroscopy 57 (1995) 375, |Ug/2KSg|
    Here use |Ug/Sg| since 2K is a constant
    NB pert is an array of perturbation strengths for all reflections
    """

    # Perturbation strength |Ug/Sg|, put 100 where we have s_g=0
    u_g = np.abs(ug_matrix[:, 0])
    pert = np.divide(u_g, np.abs(s_g_pix),
                     out=np.full_like(s_g_pix, 100.0), where=s_g_pix != 0)
    # deviation parameter and perturbation thresholds for strong beams
    max_sg = 0.001

    # Determine strong beams: Increase max_sg until enough beams are found
    strong = np.zeros_like(s_g_pix, dtype=int)
    while np.sum(strong) < min_strong_beams:
        min_pert_strong = 0.025 / max_sg
        strong = np.where((np.abs(s_g_pix) < max_sg)
                          | (pert >= min_pert_strong), 1, 0)
        max_sg += 0.001

    # Create strong beam list
    return np.flatnonzero(strong)


def bloch(g_output, s_g_pix, ug_matrix, min_strong_beams, n_hkl,
          big_k_mag, g_dot_norm, k_dot_n_pix, debug):

    # strong_beam_indices gives the index of a strong beam in the beam pool
    # Use Sg and perturbation strength to define strong beams
    strong_beam = strong_beams(s_g_pix, ug_matrix, min_strong_beams)
    # which ones are new (i.e. not already in the output list)
    strong_new = np.setdiff1d(strong_beam, g_output)
    # make the structure matrix for this pixel, outputs top of the list
    strong_beam_indices = np.concatenate((g_output, strong_new))
    n_beams = len(strong_beam_indices)

    # Make a Ug matrix for this pixel by selecting only strong beams
    beam_projection_matrix = np.zeros((n_beams, n_hkl), dtype=np.complex128)
    beam_projection_matrix[np.arange(n_beams), strong_beam_indices] = 1+0j
    # reduce the matrix using some nifty matrix multiplication
    beam_transpose = beam_projection_matrix.T
    ug_matrix_partial = np.dot(ug_matrix, beam_transpose)
    ug_sg_matrix = np.dot(beam_projection_matrix, ug_matrix_partial)

    # Final normalization of the matrix
    # off-diagonal elements are Ug/2K, diagonal elements are Sg
    # Spence's (1990) 'Structure matrix'
    ug_sg_matrix = 2.0*np.pi**2 * ug_sg_matrix / big_k_mag
    # replace the diagonal with strong beam deviation parameters
    ug_sg_matrix[np.arange(n_beams), np.arange(n_beams)] = \
        s_g_pix[strong_beam_indices]

    # weak beam correction (NOT WORKING)
    # px.weak_beams(s_g_pix, ug_matrix, ug_sg_matrix, strong_beam_indices,
    #                min_weak_beams, big_k_mag)

    # surface normal correction part 1
    structure_matrix = np.zeros_like(ug_sg_matrix)
    norm_factor = np.sqrt(1 + g_dot_norm[strong_beam_indices]/k_dot_n_pix)
    structure_matrix = ug_sg_matrix / np.outer(norm_factor, norm_factor)

    # get eigenvalues (gamma), eigenvecs
    gamma, eigenvecs = eig(structure_matrix)
    # Invert using LU decomposition (similar to ZGETRI in Fortran)
    inv_eigenvecs = inv(eigenvecs)

    # if debug:
    #     np.set_printoptions(precision=3, suppress=True)
    #     print("eigenvectors")
    #     print(eigenvecs[:5, :5])

    return n_beams, strong_beam_indices, gamma, eigenvecs, inv_eigenvecs


def wave_functions(g_output, s_g_pix, ug_matrix, min_strong_beams,
                   n_hkl, big_k_mag, g_dot_norm,
                   k_dot_n_pix, thickness, debug):
    # calculates wave functions for a given thickness by calling the bloch
    # subroutine to get the eigenvector matrices
    # and evaluating for a range of thicknesses

    n_beams, strong_beam_indices, gamma, eigenvecs, inv_eigenvecs = bloch(
        g_output, s_g_pix, ug_matrix, min_strong_beams, n_hkl, big_k_mag,
        g_dot_norm, k_dot_n_pix, debug)
    # calculate intensities

    # Initialize incident (complex) wave function psi0
    # all zeros except 000 beam which is 1
    psi0 = np.zeros(n_beams, dtype=np.complex128)
    psi0[0] = 1.0 + 0j

    # surface normal correction part 2
    m_ii = np.sqrt(1 + g_dot_norm[strong_beam_indices] / k_dot_n_pix)
    inverted_m = np.diag(m_ii)
    m_matrix = np.diag(1/m_ii)

    # evaluate for the range of thicknesses
    wave_function = ([])
    if thickness.ndim == 0:  # just one thickness
        gamma_t = np.diag(np.exp(1j * thickness * gamma))
        wave_function.append(m_matrix @ eigenvecs @ gamma_t
                             @ inv_eigenvecs @ inverted_m @ psi0)
    else:  # multiple thicknesses
        for t in thickness:
            gamma_t = np.diag(np.exp(1j * t * gamma))
            # calculate wave functions
            wave_function.append(m_matrix @ eigenvecs @ gamma_t
                                 @ inv_eigenvecs @ inverted_m @ psi0)

    # ... or, for all thicknesses at once! not working, boo
    # would avoid the type change when making wave_functions a numpy array
    # thickness = thickness[:, np.newaxis]
    # gamma_t = np.array([np.diag(np.exp(1j * t * gamma))
    #                          for t in thickness.flatten()])
    # wave_functions = np.einsum(
    #     'ij,jk,tkm,mn,nl,l->ti',
    #     m_matrix,
    #     eigenvecs,
    #     gamma_t,
    #     inv_eigenvecs,
    #     inverted_m,
    #     psi0
    # )

    return np.array(wave_function)


def weak_beams(s_g_pix, ug_matrix, ug_sg_matrix, strong_beam_list,
               min_weak_beams, big_k_mag):
    """
    Updates the Ug-Sg matrix usingweak beams according to their perturbation
    strength. We start with all non-strong beams in the list.
    We then raise the threshold perturbation strength until we have fewer than
    min_weak_beams in the list.  These are the strongest beams not included
    in strong_beam_list.
    """

    # Perturbation strength |Ug/Sg|, put 100 where we have s_g=0
    u_g = np.abs(ug_matrix[:, 0])
    pert = np.divide(u_g, np.abs(s_g_pix),
                     out=np.full_like(s_g_pix, 100.0), where=s_g_pix != 0)

    # We start with all non-strong beams in the list.
    weak = np.ones_like(s_g_pix)
    weak[strong_beam_list] = 0
    max_pert_weak = 0.0001
    # increasing threshold till we have few enough
    while np.sum(weak) > min_weak_beams:
        weak *= np.where((pert >= max_pert_weak), 1, 0)
        max_pert_weak += 0.0001

    # Create weak beam list
    weak_beam_list = np.flatnonzero(weak)
    # n_weak_beams = len(weak_beam_list)
    n_beams = len(weak_beam_list)

    # now update the scattering ug_sg_matrix (not sure if it works)
    # Add weak beams perturbatively for the 1st column (sumC)
    # and diagonal elements (sumD)
    # new version using broadcasting
    # (NOT WORKING, SIZE mismatch ug_wj * ug_wj_weak)
    weak_beam_sg = s_g_pix[weak_beam_list]
    ug_w0 = ug_matrix[weak_beam_list, 0]
    ug_wj = ug_matrix[strong_beam_list[:, None], weak_beam_list]
    ug_wj_weak = ug_matrix[weak_beam_list[:, None], strong_beam_list]

    # Eq. 4 from Zuo & Weickenmeier (Ultramicroscopy 57, 1995)
    sum_c = np.sum(ug_wj * ug_w0 / (2.0 * big_k_mag * weak_beam_sg), axis=1)

    # Eq. 5 (sumD): Broadcasting for diagonal terms
    sum_d = np.einsum('ij,ij->i', ug_wj, ug_wj_weak) / \
        (2.0 * big_k_mag * weak_beam_sg)

    # Update ug_sg_matrix: first column and diagonal terms
    ug_sg_matrix[1:n_beams, 0] -= sum_c  # Update first column (sumC)
    ug_sg_matrix[1:n_beams, 1:n_beams] -= (2.0 * big_k_mag *
                                           sum_d[:, None]) / (4.0 * np.pi**2)
    # old version using loops
    # for j in range(1, n_beams):
    #     sum_c = 0 + 0j  # Complex zero
    #     sum_d = 0 + 0j  # Complex zero

    #     for i in range(n_weak_beams):
    #         # Eq. 4 from Zuo & Weickenmeier (Ultramicroscopy 57, 1995)
    #         sum_c += (ug_matrix[strong_beam_list[j], weak[i]] *
    #                   ug_matrix[weak[i], 0] /
    #                   (2.0 * big_k_mag * s_g_pix[weak_beam_list[i]]))

    #         # Eq. 5 from Zuo & Weickenmeier (Ultramicroscopy 57, 1995)
    #         sum_d += (ug_matrix[strong_beam_list[j], weak_beam_list[i]] *
    #                   ug_matrix[weak_beam_list[i], strong_beam_list[j]] /
    #                   (2.0 * big_k_mag * s_g_pix[weak[i]]))

    #     # Update the first column of the ug_sg_matrix
    #     mask = ug_sg_matrix == ug_sg_matrix[j, 0]
    #     ug_sg_matrix[mask] = ug_sg_matrix[j, 0] - sum_c

    #     # Update the diagonal elements (Sg's)
    #     ug_sg_matrix[j, j] = ug_sg_matrix[j, j] - \
    #         2.0*big_k_mag*sum_d/(4.0*np.pi**2)
    return


def f_kirkland(z, g_magnitude):
    """
    calculates atomic scattering factor using the Kirkland model.
    From Appendix C of "Advanced Computing in Electron Microscopy", 2nd ed.

    Parameters:
    g_magnitude (ndarray): Magnitude of the scattering vector in 1/Å.
    (NB exp(-i*g.r), physics negative convention)
    z (int): Atomic number, used to index scattering factors.
    kirkland (np.ndarray): Array of scattering factors from pylix_dicts

    Returns:
    ndarray: The calculated Kirkland scattering factor.
    """
    
    q = g_magnitude / (2*np.pi)
    # coefficients in shape (3, 1, 1) for broadcasting
    a = fu.kirkland[z-1, 0:6:2].reshape(-1, 1, 1)
    b = fu.kirkland[z-1, 1:7:2].reshape(-1, 1, 1)
    c = fu.kirkland[z-1, 6:11:2].reshape(-1, 1, 1)
    d = fu.kirkland[z-1, 7:12:2].reshape(-1, 1, 1)
   
    f_g =   np.sum(a/(q**2+b), axis=0) + np.sum(c*np.exp(-(d*q**2)), axis=0)
    return f_g

# calc scattering factors for core density and valence density seperately


def f_doyle_turner(z, g_magnitude):
    """
    calculates atomic scattering factor using the Doyle & Turner model.

    Parameters:
    g_magnitude (ndarray): Magnitude of the scattering vector in 1/Å.
    (NB exp(-i*g.r), physics negative convention)
    z (int): Atomic number, used to index scattering factors.
    kirkland (np.ndarray): Array of scattering factors from pylix_dicts

    Returns:
    ndarray: The calculated Doyle & Turner scattering factor.
    """
    # Convert g to s
    s = g_magnitude / (2*np.pi)

    a = fu.doyle_turner[z-1, 0:8:2].reshape(-1, 1, 1)
    b = fu.doyle_turner[z-1, 1:8:2].reshape(-1, 1, 1)

    f_g = np.sum(a * np.exp(-(b * s**2)), axis=0)

    return f_g


def f_lobato(z, g_magnitude):
    """
    calculates atomic scattering factor using the Lobato model.
    Lobato & van Dyck Acta Cryst A70, 636 (2014)
    Parameters:
    g_magnitude (ndarray): Magnitude of the scattering vector in 1/Å.
    (NB exp(-i*g.r), physics negative convention)
    z (int): Atomic number, used to index scattering factors.
    lobato (np.ndarray): Array of scattering factors from pylix_dicts

    Returns:
    ndarray: The calculated Lobato scattering factor.
    """
    # Convert physics to crystallography convention
    g = g_magnitude / (2*np.pi)

    a = fu.lobato[z-1, 0:5].reshape(-1, 1, 1)
    b = fu.lobato[z-1, 5:10].reshape(-1, 1, 1)
    bg2 = b*(g*g)
    f_g = np.sum(a*(2+bg2)/((1+bg2)**2), axis=0)

    return f_g


def f_peng(z, g_magnitude):
    """
    calculates atomic scattering factor using the Peng model.
    Peng, Micron 30, 625 (1999)
    Parameters:
    g_magnitude (ndarray): Magnitude of the scattering vector in 1/Å.
    (NB exp(-i*g.r), physics negative convention)
    z (int): Atomic number, used to index scattering factors.
    peng (np.ndarray): Array of scattering factors from pylix_dicts

    Returns:
    ndarray: The calculated Peng scattering factor.
    """
    # Convert g to s
    s = g_magnitude / (4*np.pi)

    a = fu.peng[z-1, 0:4].reshape(-1, 1, 1)
    b = fu.peng[z-1, 4:8].reshape(-1, 1, 1)
    b_ = -b*(s**2)
    f_g = np.sum(a*np.exp(b_), axis=0)

    return f_g


def calc_slater_orbitals(z, orbital, r):
    # ok so now when we calc slater orbital we pass kappa scaled q
    # r, distance of electron from atomic nucleus#
    # N is normalizing constant

    delta = np.array(fu.slater_coefficients[z][orbital]['delta'])
    delta = delta / 0.52917721092
    # convert to angstrom
    C = np.array(fu.slater_coefficients[z][orbital]['coeff'])
    n = int(orbital[0])

    # for now we just state 1s contriutes to the core and 2s contributes
    # to valence with a respective electron occupation of 2,1
    # we need array of R values to sample the electron density from
    # so we can actually evaluate a fourier transform integral
    R_total = 0

    # Total radial finction is a superposition of these primitive radial
    # functions and their corresponding expansion coefficent C_jln given
    # in out hartree fock equation
    # delta is given next to each slater type orbital in the table

    # after we fourier transform radial function to get form factor we
    # use Mott-Bethe formula to get to electron scattering factor
    # then compare with kirkland to check agreement and upscale

    for cj, zj in zip(C, delta):
        Nj = ((2*zj)**(n+0.5))/(np.sqrt(math.factorial(2*n)))
        # each electron is defined by a primitive slater orbital of this form
        S_j = Nj*r**(n-1)*np.exp(-zj*r)
        R_total += cj*S_j
    # return the radial function for our atom, integrated to get form factor
    return R_total


"""
def xray_form_factor_valence(r, rho, S,pv,k):
    # rho = electron density at r, Q = array of Q values
    #S= Q/2*np.pi
    #not vectorized currently 
    fQ = []
    for s in S:
        integrand =  4*np.pi*(k**3)*rho * r**2 * np.sinc(2*s*r)  # np.sinc(x) = sin(pi x)/(pi x)
        fQ.append(np.trapz(integrand, r))
    return pv*np.array(fQ)   #scale by number of electrons in valence



def xray_form_factor_core(r,rho,S,pc):
    fQ = []
    for s in S:
        integrand =  4*np.pi*rho * r**2 * np.sinc(2*s*r)  # np.sinc(x) = sin(pi x)/(pi x)
        fQ.append(np.trapz(integrand, r))
    return pc*np.array(fQ)
    
"""


@njit(fastmath=True)
def _sinc_numba(x):
    if x == 0.0:
        return 1.0
    pix = math.pi * x
    return math.sin(pix) / pix


@njit(fastmath=True, parallel=True)
def _form_factor_kernel(r, rho, S, scale):
    """
    Multi-threaded Numba kernel.
    Parallelized over S.
    Computes:  f_Q(s) = scale * ∫ 4*pi*rho*r^2 * sinc(2*s*r) dr
    """
    Nr = r.shape[0]
    Ns = S.shape[0]

    # Precompute base = 4*pi * rho * r^2
    base = np.empty(Nr, dtype=np.float64)
    for j in range(Nr):
        base[j] = 4.0 * math.pi * rho[j] * (r[j] * r[j])

    out = np.zeros(Ns, dtype=np.float64)

    # Parallel loop over S
    for i in prange(Ns):
        s = S[i]
        acc = 0.0

        # manual trapezoid rule along r
        for j in range(Nr - 1):
            x1 = 2.0 * s * r[j]
            x2 = 2.0 * s * r[j+1]

            f1 = base[j]   * _sinc_numba(x1)
            f2 = base[j+1] * _sinc_numba(x2)

            dr = r[j+1] - r[j]
            acc += 0.5 * (f1 + f2) * dr

        out[i] = scale * acc

    return out


def xray_form_factor_valence(r, rho, S, pv, k):
    r = np.asarray(r, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    scale = pv * (k**3)
    return _form_factor_kernel(r, rho, S, scale)


def xray_form_factor_core(r, rho, S, pc):
    r = np.asarray(r, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    scale = pc
    return _form_factor_kernel(r, rho, S, scale)


def precompute_densities(Z, kappa, pv):
    r_max = 20  # in angstrom
    n_points = 1000
    r = np.linspace(1e-6, r_max, n_points)

    core_orbitals = fu.elements_info[Z]['core_orbitals']
    valence_orbitals = fu.elements_info[Z]['valence_orbitals']
    core_density = 0
    valence_density = 0

    n_e_core = 0
    for orbital in core_orbitals:
        n_e_core += fu.elements_info[Z]['occupation'][orbital]
        R = calc_slater_orbitals(Z, orbital, r)
        core_density += (R**2)
    core_density /= (4*np.pi)
    core_density_n = core_density / np.trapz(4*np.pi*r**2*core_density, r)
    for orbital in valence_orbitals:
        R = calc_slater_orbitals(Z, orbital, r*kappa)
        valence_density += (R**2)
    valence_density /= (4*np.pi)
    # need to normalize to 1 electron then scale by pv after
    valence_density_n = valence_density / np.trapz(4*np.pi*r**2*valence_density, r) 

    # N_core =  (4*np.pi * np.trapz(r**2 * core_density, r))
    # N_valence = (4*np.pi * np.trapz(r**2 * valence_density, r))
    #core_density = (1/(4*np.pi))*(R_core**2)  # this will depend on n,l if multipolar is considered so its more complicate than this , just using this as an example 
    #valence_density = (1/(4*np.pi))*(R_valence**2)# wavefucniton = R(r)Y_l^m(theta,phi)

    pc = fu.elements_info[Z]['pc']
    # p_atom(r) in kappa formalism
    density_total = pc*core_density_n + pv*kappa**3*valence_density_n
    integrand = density_total*np.pi*r**2
    # mean square radius of electrons in the atom
    r2_expect = np.trapz(r**2*integrand, x=r)/np.trapz(integrand, x=r)
    fu.elements_info[Z]["r2"] = r2_expect
    fu.precomputed_densities[Z] = {
        "r": r.copy(),
        "core": core_density_n.copy(),
        "valence": valence_density_n.copy(),
        "r2": r2_expect
        }

    return -1


def calc_scattering_amplitudes(q, Z, pv, kappa):
    rho_core = fu.precomputed_densities[Z]["core"]
    rho_val = fu.precomputed_densities[Z]["valence"]
    r = fu.precomputed_densities[Z]["r"]
    pc = fu.elements_info[Z]['pc']
    # precomputed densities
    f_valence = xray_form_factor_valence(r, rho_val, q, pv, kappa)
    f_core = xray_form_factor_core(r, rho_core, q, pc)
    # fourier transform of the calculated radial funciton in 3d from 0 to inf
    f_x_total = f_core + f_valence

    return f_x_total


def convert_x(Z, f_x, q):
    Bohr = 0.52917721067  # in angstrom
    q = np.asarray(q)
    f_x = np.asarray(f_x)

    # Output array
    f_e = np.zeros_like(q, dtype=float)

    # Mask where q == 0
    mask0 = (q == 0)
    maskN = ~mask0  # q ≠ 0

    # Special case for q = 0 (Ibers correction)
    r2 = fu.elements_info[Z]["r2"]
    f_e[mask0] = (Z * r2) / (3 * Bohr)

    # General Motte–Beth formula
    f_e[maskN] = (Z - f_x[maskN]) / (2 * np.pi**2 * Bohr * q[maskN]**2)

    return f_e


# need to carefully look through and fix scaling of function
# close but not quite
def kappa_factors(g, Z, pv, kappa):
    orig_shape = g.shape
    g_flat = g.flatten()
    S = g_flat / (2*np.pi)
    f_out = np.zeros_like(g_flat, dtype=float)
    f_out = convert_x(
        Z,
        calc_scattering_amplitudes(S, Z, pv, kappa),
        S)

    return f_out.reshape(orig_shape)
# should handle values below 0.5 Q using kirkland values or some type of extrapolation


def four_gauss(x, args):
    # # returns the sum of four Gaussians & a constant for Thomas f_prime
    f = args[0]*np.exp(-abs(args[1])*x**2) + \
        args[2]*np.exp(-abs(args[3])*x**2) + \
        args[4]*np.exp(-abs(args[5])*x**2) + \
        args[6]*np.exp(-abs(args[7])*x**2) + args[8]
    return f
    # x: array (..., 3)
    # args: array (..., 9)
    # return (
    #     args[..., 0]*np.exp(-abs(args[..., 1])*x[..., 0]**2) +
    #     args[..., 2]*np.exp(-abs(args[..., 3])*x[..., 1]**2) +
    #     args[..., 4]*np.exp(-abs(args[..., 5])*x[..., 2]**2) +
    #     args[..., 6]*np.exp(-abs(args[..., 7]) *
    #                         (x[..., 0]**2+x[..., 1]**2+x[..., 2]**2)) +
    #     args[..., 8])


def f_thomas(g, B, Z, v):
    # interpolated of parameterised Bird & King absorptive scattering factors
    # calculation uses s = g/2
    s = g/2
    # returns an interpolated absorptive scattering factor
    Bvalues = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                        0.7, 1, 1.5, 2, 2.75, 4])

    # error checking
    if np.any(s < 0):
        raise ValueError("invalid values of s (must be ≥ 0)")
    if np.any(B < 0):
        raise ValueError("invalid B values (must be ≥ 0)")
    B = np.clip(B, 0.1, None)  # enforce B >= 0.1
    if np.any(B > 4):
        raise ValueError("B values outside parameterised range [0.1, 4]")
    if Z < 1 or Z > 103:
        raise ValueError("invalid Z")
    # If all B are zero: return zero map
    if np.all(B == 0):
        return np.zeros(B.shape)

    # # interpolation - index in Bvalues just above B
    # idx_hi = np.searchsorted(Bvalues, B, side='left')
    # # exact matches (no interpolation needed)
    # exact_mask = (idx_hi < len(Bvalues)) & (Bvalues[idx_hi] == B)
    # # clamp idx_hi to valid interior region (1 ... len-1)
    # idx_hi = np.clip(idx_hi, 1, len(Bvalues) - 1)
    # idx_lo = idx_hi - 1
    # # Retrieve coefficients for the two bounding B-values for each (i,j)
    # params_lo = fu.thomas[Z - 1][idx_lo]     # shape (n_hkl, n_hkl, 9)
    # params_hi = fu.thomas[Z - 1][idx_hi]     # same shape
    # # Compute Gaussian sums for both bounding sets
    # line_lo = four_gauss(s, params_lo)       # shape (n_hkl, n_hkl)
    # line_hi = four_gauss(s, params_hi)       # shape (n_hkl, n_hkl)
    # # linear interpolation factor
    # B_lo = Bvalues[idx_lo]
    # B_hi = Bvalues[idx_hi]
    # t = (B - B_lo) / (B_hi - B_lo)

    # # final interpolated line
    # line = line_lo + t * (line_hi - line_lo)

    # # Replace values where B matched exactly
    # if np.any(exact_mask):
    #     exact_idx = idx_hi   # same index for exact match
    #     line_exact = four_gauss(s[exact_mask],
    #                             fu.thomas[Z - 1][exact_idx[exact_mask]])
    #     line[exact_mask] = line_exact
    # # error checking
    # if isinstance(s, np.ndarray) and np.any(s < 0):
    #     raise Exception("inavlid values of s")
    # elif isinstance(s, (int, float)) and s < 0:
    #     raise Exception("invalid value of s")
    # if B < 0:
    #     raise Exception("invalid value of B")
    # if B < 0.1:
    #     B = 0.1
    # if B > 4:  # or 0 < B < 0.1:
    #     raise Exception(f"B = {B}! Outside range of parameterisation")
    # if Z < 1 or Z > 103:
    #     raise Exception("invalid value of Z")
    # if isinstance(s, np.ndarray) and B == 0:
    #     return np.zeros(np.shape(s))
    # elif isinstance(s, (int, float)) and B == 0:
    #     return 0

    # get f_prime
    if np.any(B == Bvalues):  # we don't need to interpolate
        i = np.where(Bvalues == B)[0][0]
        line = four_gauss(s, fu.thomas[Z-1][i])
    else:  # interpolate between parameterised values
        i = np.where(Bvalues >= B)[0][0]
        bounding_b = Bvalues[i - 1:i + 1]
        line1 = four_gauss(s, fu.thomas[Z-1][i - 1])
        line2 = four_gauss(s, fu.thomas[Z-1][i])
        line = line1 + (B - bounding_b[0])*(line2 - line1) / \
            (bounding_b[1] - bounding_b[0])
    f_prime = np.where(line > 0, line, 0)*c/v

    return f_prime


def read_dm3(file_path, x, debug):
    """
    Reads a .dm3 file, finds tags (ignores their structure and content),
    and extracts image data.

    We assume a square image of size[x, x]

    In principle it is possible to get the dimensions of the image from the
    tags and use this to determine image size x.  However if the image
    size doesn't match the simulation there's not much point (at the moment)

    We look for delimiters %%%% that lies between tag labels and data
    """
    if debug:
        print(f"Reading image {file_path}")
    y = x
    try:
        with open(file_path, 'rb') as f:
            n_tags = 0
            n_datatags = 0
            tag_delimiter = b'%%%%'
            tag_label = ""
            # Initialize an empty image
            image = np.zeros((y, x), dtype=np.float32)

            # Create a buffer for 60 bytes
            buffersize = 60
            prev_bytes = bytearray(buffersize)
            prev_4_bytes = bytearray(4)

            # first go through the tags
            find_tags = True
            while find_tags is True:
                # Read one byte at a time
                byte = f.read(1)
                if not byte:
                    # End of file
                    raise ValueError("Unexpected end of file")
                    break
                prev_bytes = prev_bytes[1:] + byte
                prev_4_bytes = prev_4_bytes[1:] + byte

                if prev_4_bytes == tag_delimiter:
                    n_tags += 1
                    tag_label = ""
                    # The tag label is in the bytes before delimiter
                    for j in range(buffersize-5, 0, -1):
                        char_byte = prev_bytes[j]
                        if chr(char_byte) == '%':  # previous tag
                            break
                        if 32 <= char_byte <= 126:  # ASCII character
                            tag_label = chr(char_byte) + tag_label
                        else:
                            break

                    # 16 bytes of tag information
                    # first 4 bytes = tag type
                    tag_type = struct.unpack('>I', f.read(4))[0]
                    if tag_type == 1:
                        data_type = "int"
                    elif tag_type == 2:
                        data_type = "int16"
                    elif tag_type == 3:
                        data_type = "real 4"
                    elif tag_type == 4:
                        data_type = "uint16"
                    elif tag_type == 5:
                        data_type = "uint32"
                    elif tag_type == 6:
                        data_type = "float32"
                    elif tag_type == 7:
                        data_type = "float64"
                    elif tag_type == 8:
                        data_type = "bool"
                    elif tag_type == 9:
                        data_type = "uint8 character"
                    elif tag_type == 10:
                        data_type = "octet"
                    elif tag_type == 11:
                        data_type = "uint64"
                    else:
                        data_type = str(tag_type)
                        # raise ValueError("Data type not recognised")

                    # next 4 bytes is the tag form (not sure about these)
                    data_form = ""
                    tag_form = struct.unpack('>I', f.read(4))[0]
                    if tag_form == 15:
                        data_form = "1D array"
                    elif tag_form == 5:
                        data_form = "int"
                    elif tag_form == 6:
                        data_form = "float"
                    elif tag_form == 8:
                        data_form = "bool"
                    elif tag_form == 20:
                        data_form = "2D array"
                    if debug:
                        print(f"{tag_label}: {data_type}, {data_form}")

                    # Check if it's the second 'Data' tag
                    if tag_label == 'Data':
                        n_datatags += 1
                        if n_datatags == 2:  # found it
                            find_tags = False

            # Read the image
            # next 4 bytes is data_type (again?_
            tag_type = struct.unpack('>I', f.read(4))[0]
            if tag_type == 2:
                pix_bytes = 2  # Signed 2-byte integer
            elif tag_type == 3:
                pix_bytes = 4  # Signed 4-byte integer
            elif tag_type == 4:
                pix_bytes = 2  # Unsigned 2-byte integer
            elif tag_type == 5:
                pix_bytes = 4  # Unsigned 4-byte integer
            elif tag_type == 6:
                pix_bytes = 4  # int32
            elif tag_type == 7:
                pix_bytes = 8  # int64
            elif tag_type == 9:
                pix_bytes = 1  # Signed 1-byte integer
            elif tag_type == 10:
                pix_bytes = 1  # Unsigned 1-byte integer
            else:
                raise ValueError(f"data type [{tag_type}] not recognised")

            # final 4 bytes is data length
            data_length = struct.unpack('>I', f.read(4))[0]
            # Check data array length, error if things don't match
            expected_data_length = y * x
            if data_length != expected_data_length:
                raise ValueError(f"Data length mismatch. Expected: {expected_data_length}, Got: {data_length}")

            total_bytes = data_length * pix_bytes  # 4 bytes per float32
            raw_data = f.read(total_bytes)
            # Error if things don't match
            if len(raw_data) != total_bytes:
                raise ValueError(f"Error: Incomplete data read. Expected {total_bytes} bytes, got {len(raw_data)} bytes.")

            # put the data into a 2D image according to its data type
            if tag_type == 2:
                image = np.frombuffer(raw_data, dtype='<i2').reshape((x, y))
            if tag_type == 3:
                image = np.frombuffer(raw_data, dtype='<i4').reshape((x, y))
            if tag_type == 4:
                image = np.frombuffer(raw_data, dtype='<u2').reshape((x, y))
            if tag_type == 5:
                image = np.frombuffer(raw_data, dtype='<u4').reshape((x, y))
            if tag_type == 6:
                image = np.frombuffer(raw_data, dtype='<f4').reshape((x, y))
            if tag_type == 7:
                image = np.frombuffer(raw_data, dtype='<f8').reshape((x, y))
            if tag_type == 9:
                image = np.frombuffer(raw_data, dtype=np.int8).reshape((x, y))
            if tag_type == 10:
                image = np.frombuffer(raw_data, dtype=np.uint8).reshape((x, y))
        return image

    except FileNotFoundError:
        print(f"{file_path} not found")


def parabo3(x, y):
    # y=a*x^2+b*x+c
    d = x[0]*x[0]*(x[1]-x[2]) + x[1]*x[1]*(x[2]-x[0]) + x[2]*x[2]*(x[0]-x[1])
    if (abs(d) > 1e-10):  # we get zero d if all three inputs are the same
        a =(x[0]*(y[2]-y[1]) + x[1]*(y[0]-y[2]) + x[2]*(y[1]-y[0])) / d
        b =(x[0]*x[0]*(y[1]-y[2]) + x[1]*x[1]*(y[2]-y[0]) +
            x[2]*x[2]*(y[0]-y[1])) / d
        c =(x[0]*x[0]*(x[1]*y[2]-x[2]*y[1]) + x[1]*x[1]*(x[2]*y[0]-x[0]*y[2]) +
            x[2]*x[2]*(x[0]*y[1]-x[1]*y[0])) / d
        x_v = -b/(2*a)  # x-coord
        y_v = c-b*b/(4*a)  # y-coord
    else:
        x_v = x[np.argmin(y)]
        y_v = y[np.argmin(y)]

    return x_v, y_v


def convex(r3_x, r3_y):
    # Checks the three points coming in to see if a parabolic fit for a
    # minimum is possible.  If so, returns the predicted minimum (minny=True).
    # If not, returns the next point to check (minny=False).
    tol = 1e-10
    x_max = np.argmax(r3_x)  # index of lowest x
    x_min = np.argmin(r3_x)  # index of highest x
    if r3_x[x_max] - r3_x[x_min] > tol:
        x_mid = 3 - x_max - x_min  # index of mid x
        if r3_x[x_max] - r3_x[x_mid] > tol and r3_x[x_mid] - r3_x[x_min] > tol:
            convexity_test = -abs(r3_y[x_max] - r3_y[x_min])
            # convexity is y at the mid x
            # if there was a straight line between lowest and highest x
            convexity = r3_y[x_mid] - (
                r3_y[x_min] + (r3_x[x_mid] - r3_x[x_min]) *
                (r3_y[x_max] - r3_y[x_min]) /
                (r3_x[x_max] - r3_x[x_min]))
        else:
            raise ValueError("Parabolic refinement failed")
    else:
        raise ValueError("Parabolic refinement failed")
    if convexity > 0.1 * convexity_test:
        # find the size of the step between the two lowest y
        y_max = np.argmax(r3_y)  # index of highest y
        y_min = np.argmin(r3_y)  # index of lowest y
        y_mid = 3 - y_max - y_min  # index of mid y
        last_dx = r3_x[y_min] - r3_x[y_mid]
        # use exp to give an irrational step size and avoid going to the same
        # point twice, exp(0.75)~=2.12
        next_x = r3_x[y_min] + np.exp(0.75) * last_dx
        minny = False
        print("Convex, will contine")  # going to {next_x:.2f}")
    else:
        next_x, next_y = parabo3(r3_x, r3_y)
        print(f"Concave, predict minimum at {next_x:.4f} with fit index {100*next_y:.2f}%")
        minny = True

    return next_x, minny


def get_git():
    try:
        # Run the git command to get the latest commit ID
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']
                                            ).strip().decode('utf-8')
        return commit_id
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving commit ID: {e}")
        return None


def hkl_string(hkl):
    # Initialize strings for h, k, l values
    string = ""
    for value in hkl:
        if value >= 0:
            formatted_value = f"+{value}"
        else:
            formatted_value = f"{value}"
        string += formatted_value.strip()
    return string


def atom_move(space_group_number, wyckoff):
    moves = np.zeros([3, 3])
    if space_group_number == 1:  # P1
        if ('x') in wyckoff:
            moves[0, :] = ([0.0, 1.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('b') in wyckoff:
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        else:
            raise ValueError("Wyckoff Symbol for space group P1 not recognised")   
    # if space_group_number == 2:
    # if space_group_number == 3:
    # if space_group_number == 4:
    # if space_group_number == 5:
    # if space_group_number == 6:
    # if space_group_number == 7:
    # if space_group_number == 8:
    # if space_group_number == 9:
    # if space_group_number == 10:
    # if space_group_number == 11:
    # if space_group_number == 12:
    # if space_group_number == 13:
    # if space_group_number == 14:
    # if space_group_number == 15:
    if space_group_number == 15:  # C 1 2/c 1
        if ('e') in wyckoff:  # point symmetry 2 along y
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd') not in wyckoff:  # point symmetry -1
            raise ValueError("Wyckoff Symbol for space group C2/c, not recognised")
    # if space_group_number == 16:
    # if space_group_number == 17:
    # if space_group_number == 18:
    # if space_group_number == 19:
    # if space_group_number == 20:
    # if space_group_number == 21:
    # if space_group_number == 22:
    # if space_group_number == 23:
    # if space_group_number == 24:
    # if space_group_number == 25:
    # if space_group_number == 26:
    # if space_group_number == 27:
    # if space_group_number == 28:
    # if space_group_number == 29:
    # if space_group_number == 30:
    # if space_group_number == 31:
    # if space_group_number == 32:
    # if space_group_number == 33:
    # if space_group_number == 34:
    # if space_group_number == 35:
    if space_group_number == 36:  # C m c 21
        # NEED TO CODE ALTERNATIVE SETTINGS Ccm21,Bb21m,Bm21b,A21ma,A21am
        if ('a') in wyckoff:  # point symmetry 1, coordinate [x,y,z],
          moves[0, :] = ([0.0, 1.0, 0.0])
          moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('b') in wyckoff:  # point symmetry 1, coordinate [x,y,z],
          moves[0, :] = ([1.0, 0.0, 0.0])
          moves[1, :] = ([0.0, 1.0, 0.0])
          moves[2, :] = ([0.0, 0.0, 1.0])
        else:
            raise ValueError("Wyckoff Symbol for space group Cmc21 not recognised")
    # if space_group_number == 37:
    # if space_group_number == 38:
    # if space_group_number == 39:
    # if space_group_number == 40:
    # if space_group_number == 41:
    # if space_group_number == 42:
    # if space_group_number == 43:
    # if space_group_number == 44:
    # if space_group_number == 45:
    # if space_group_number == 46:
    # if space_group_number == 47:
    # if space_group_number == 48:
    # if space_group_number == 49:
    # if space_group_number == 50:
    # if space_group_number == 51:
    # if space_group_number == 52:
    # if space_group_number == 53:
    # if space_group_number == 54:
    # if space_group_number == 55:
    # if space_group_number == 56:
    # if space_group_number == 57:
    # if space_group_number == 58:
    # if space_group_number == 59:
    # if space_group_number == 60:
    # if space_group_number == 61:
    # if space_group_number == 62:
    if space_group_number == 63:  # Cmcm
        # a: point symmetry 2/m, coordinate [0,0,0] & eq, no movement
        # b: point symmetry 2/m, coordinate [0,1/2,0] & eq, no movement
        # d: point symmetry -1, coordinate [1/4,1/4,0] & eq, no movement
        if ('c') in wyckoff:  # point symmetry mm2, coordinate [0,y,1/4] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('e') in wyckoff:  # point symmetry 2(x), coordinate [x,0,0] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry m(x), coordinate [0,y,z] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('g') in wyckoff:  # point symmetry m(z), coordinate [x,y,1/4] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
        elif ('h') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group Cmc2m not recognised")
    if space_group_number == 64:  # Cmca
        # a: point symmetry 2/m, coordinate [0,0,0] & eq, no movement
        # b: point symmetry 2/m, coordinate [1/2,0,0] & eq, no movement
        # c: point symmetry -1, coordinate [1/4,1/4,0] & eq, no movement
        if ('d') in wyckoff:  # point symmetry 2(x), coordinate [x,0,0] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('e') in wyckoff:  # point symmetry 2(y), coordinate [1/4,y,1/4] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry m(x), coordinate [0,y,z] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('g') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group Cmca not recognised")
    # if space_group_number == 65:
    # if space_group_number == 66:
    # if space_group_number == 67:
    if space_group_number == 68:  # Ccca
        # N.B. multiple origin choices allowed, here origin at 222, -1 at [1/4,0,1/4]
        # a: point symmetry 222, coordinate [0,0,0] & eq, no movement
        # b: point symmetry 222, coordinate [0,0,1/2] & eq, no movement
        # c: point symmetry -1, coordinate [1/4,0,1/4] & eq, no movement
        # d: point symmetry -1, coordinate [0,1/4,1/4] & eq, no movement
        if ('e') in wyckoff:  # point symmetry 2(x), coordinate [x,0,0] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 2(y), coordinate [0,y,0] & eq
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('g') in wyckoff:  # point symmetry 2(z), coordinate [0,0,z] & eq
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('h') in wyckoff:  # point symmetry 2(z), coordinate [1/4,1/4,z] & eq
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('i') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group Ccca not recognised")
    # if space_group_number == 69:
    # if space_group_number == 70:
    # if space_group_number == 71:
    # if space_group_number == 72:
    # if space_group_number == 73:
    # if space_group_number == 74:
    # if space_group_number == 75:
    # if space_group_number == 76:
    # if space_group_number == 77:
    # if space_group_number == 78:
    # if space_group_number == 79:
    # if space_group_number == 80:
    # if space_group_number == 81:
    # if space_group_number == 82:
    # if space_group_number == 83:
    # if space_group_number == 84:
    # if space_group_number == 85:
    # if space_group_number == 86:
    # if space_group_number == 87:
    # if space_group_number == 88:
    # if space_group_number == 89:
    # if space_group_number == 90:
    # if space_group_number == 91:
    # if space_group_number == 92:
    # if space_group_number == 93:
    # if space_group_number == 94:
    # if space_group_number == 95:
    # if space_group_number == 96:
    # if space_group_number == 97:
    # if space_group_number == 98:
    if space_group_number == 99:  # P 4 m m 
        if ('a') in wyckoff:  # point symmetry 4mm, coordinate [0,0,z], allowed movement along [001]
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('b') in wyckoff:  # point symmetry 4mm, coordinate [1/2,1/2,z], allowed movement along [001]
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('c') in wyckoff:  # point symmetry mm, coordinate [1/2,0,z] & eq, allowed movement along [001]
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('d') in wyckoff:  # point symmetry m, coordinate [x,x,z] & eq, allowed movement along [110] & [001]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('e') in wyckoff:  # point symmetry m, coordinate [x,0,z] & eq, allowed movement along [100] & [001]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('f') in wyckoff:  # point symmetry m, coordinate [x,1/2,z] & eq, allowed movement along [100] & [001]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('g') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        else:
            raise ValueError("Wyckoff Symbol for space group P4mm not recognised")
    # if space_group_number == 100:
    # if space_group_number == 101:
    # if space_group_number == 102:
    # if space_group_number == 103:
    # if space_group_number == 104:
    # if space_group_number == 105:
    # if space_group_number == 106:
    # if space_group_number == 107:
    # if space_group_number == 108:
    # if space_group_number == 109:
    # if space_group_number == 110:
    # if space_group_number == 111:
    # if space_group_number == 112:
    # if space_group_number == 113:
    # if space_group_number == 114:
    # if space_group_number == 115:
    # if space_group_number == 116:
    # if space_group_number == 117:
    # if space_group_number == 118:
    # if space_group_number == 119:
    # if space_group_number == 120:
    # if space_group_number == 121:
    # if space_group_number == 122:
    # if space_group_number == 123:
    # if space_group_number == 124:
    # if space_group_number == 125:
    # if space_group_number == 126:
    # if space_group_number == 127:
    # if space_group_number == 128:
    # if space_group_number == 129:
    # if space_group_number == 130:
    # if space_group_number == 131:
    # if space_group_number == 132:
    # if space_group_number == 133:
    # if space_group_number == 134:
    # if space_group_number == 135:
    # if space_group_number == 136:
    # if space_group_number == 137:
    # if space_group_number == 138:
    if space_group_number == 139:  # I4/m m m
        # a: point symmetry 4/mmm, coordinate [0,0,0], no allowed movements
        # b: point symmetry 4/mmm, coordinate [0,0,1/2], no allowed movements
        # c: point symmetry mmm, coordinate [0,1/2,0] or [1/2,0,0], no allowed movements
        # d: point symmetry -4m2, coordinate [0,1/2,1/4] or [1/2,0,1/4], no allowed movements
        # f: point symmetry 2/m, coordinate [1/4,1/4,1/4] & eq, no allowed movements
        if ('e') in wyckoff:  # point symmetry 4mm, coordinate [0,0,z] allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('g') in wyckoff:  # point symmetry mm, coordinate [0,1/2,z] & eq, allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('h') in wyckoff:  # point symmetry mm, coordinate [x,x,0] & eq, allowed movement along [110]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
        elif ('i') in wyckoff:  # point symmetry mm, coordinate [x,0,0] & eq, allowed movement along [100]
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('j') in wyckoff:  # point symmetry mm, coordinate [x,1/2,0] & eq, allowed movement along [100]
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('k') in wyckoff:  # point symmetry 2, coordinate [x,1/2+x,1/4] & eq, allowed movement along [110]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
        elif ('l') in wyckoff:  # point symmetry m, coordinate [x,y,0] & eq, allowed movement along [100] & [010]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
        elif ('m') in wyckoff:  # point symmetry m, coordinate [x,x,z] & eq, allowed movement along [110] & [001]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('n') in wyckoff:  # point symmetry m, coordinate [x,0,z] & eq, allowed movement along [100] & [001]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 0.0, 1.0])
        elif ('o') in wyckoff:  # point symmetry 1, coordinate [x,y,z] & eq, allowed movement along [100], [010] & [001]
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd' or 'f') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group I4/mmm not recognised")
    # if space_group_number == 140:
    # if space_group_number == 141:
    if space_group_number == 142:  # I41/acd
        # a: point symmetry -4, no allowed movements
        # b: point symmetry 222, no allowed movements
        # c: point symmetry -1, no allowed movements
        if ('d') in wyckoff:  # point symmetry 2, allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('e') in wyckoff:  # point symmetry 2, allowed movement along x
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 2, allowed movement along [x,x,0]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
        elif ('g') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group I41/acd not recognised")
    # if space_group_number == 143:
    # if space_group_number == 144:
    # if space_group_number == 145:
    # if space_group_number == 146:
    # if space_group_number == 147:
    # if space_group_number == 148:
    # if space_group_number == 149:
    # if space_group_number == 150:
    # if space_group_number == 151:
    # if space_group_number == 152:
    # if space_group_number == 153:
    # if space_group_number == 154:
    # if space_group_number == 155:
    # if space_group_number == 156:
    # if space_group_number == 157:
    # if space_group_number == 158:
    # if space_group_number == 159:
    # if space_group_number == 160:
    if space_group_number == 161:  # R3c, ***Hexagonal setting only***
        # b: point symmetry -3, no allowed movements
        if ('a') in wyckoff:  # point symmetry 3, allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('b') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group R-3c not recognised")

    # if space_group_number == 162:
    # if space_group_number == 163:
    # if space_group_number == 164:
    # if space_group_number == 165:
    # if space_group_number == 166:
    # if space_group_number == 167:
    if space_group_number == 167:  # R-3c
        # a: point symmetry 32, no allowed movements
        # b: point symmetry -3, no allowed movements
        if ('c') in wyckoff:  # point symmetry 3, allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        # d: point symmetry -1, no allowed movements
        elif ('e') in wyckoff:  # point symmetry 2, allowed movement along [x,0,0]
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group R-3c not recognised")
    # if space_group_number == 168:
    # if space_group_number == 169:
    # if space_group_number == 170:
    # if space_group_number == 171:
    # if space_group_number == 172:
    # if space_group_number == 173:
    # if space_group_number == 174:
    # if space_group_number == 175:
    # if space_group_number == 176:
    # if space_group_number == 177:
    # if space_group_number == 178:
    # if space_group_number == 179:
    # if space_group_number == 180:
    # if space_group_number == 181:
    # if space_group_number == 182:
    # if space_group_number == 183:
    # if space_group_number == 184:
    # if space_group_number == 185:
    # if space_group_number == 186:
    # if space_group_number == 187:
    # if space_group_number == 188:
    # if space_group_number == 189:
    # if space_group_number == 190:
    # if space_group_number == 191:
    # if space_group_number == 192:
    # if space_group_number == 193:
    # if space_group_number == 194:
    # if space_group_number == 195:
    # if space_group_number == 196:
    # if space_group_number == 197:
    # if space_group_number == 198:
    # if space_group_number == 199:
    # if space_group_number == 200:
    # if space_group_number == 201:
    # if space_group_number == 202:
    # if space_group_number == 203:
    # if space_group_number == 204:
    # if space_group_number == 205:
    # if space_group_number == 206:
    # if space_group_number == 207:
    # if space_group_number == 208:
    # if space_group_number == 209:
    # if space_group_number == 210:
    # if space_group_number == 211:
    # if space_group_number == 212:
    # if space_group_number == 213:
    # if space_group_number == 214:
    # if space_group_number == 215:
    if space_group_number == 216:  # F-43m
        # a: point symmetry -43m, coordinate [0,0,0], no allowed movements
        # b: point symmetry -43m, coordinate [1/2,1/2,1/2], no allowed movements
        # c: point symmetry -43m, coordinate [1/4,1/4,1/4], no allowed movements
        # d: point symmetry -43m, coordinate [3/4,3/4,3/4], no allowed movements
        if ('e') in wyckoff:  # point symmetry 3m, coordinate [x,x,x] allowed movement along [111]
            moves[0, :] = ([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
        elif ('f') in wyckoff:  # point symmetry mm, coordinate [x,0,0] allowed movements along x
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('g') in wyckoff:  # point symmetry mm, coordinate [x,1/4,1/4] allowed movement along x
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('h') in wyckoff:  # point symmetry m, coordinate [x,x,z], allowed movement along [110] and [001]
            moves[0, :] = ([1/np.sqrt(2), 1/np.sqrt(2), 0.0])
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('i') in wyckoff:  # point symmetry 1, coordinate [x,y,z], allowed movement along x,y,z
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group F-43m not recognised")
    # if space_group_number == 217:
    # if space_group_number == 218:
    # if space_group_number == 219:
    # if space_group_number == 220:
    # if space_group_number == 221:
    # if space_group_number == 222:
    # if space_group_number == 223:
    # if space_group_number == 224:
    # if space_group_number == 225:
    # if space_group_number == 226:
    # if space_group_number == 227:
    # if space_group_number == 228:
    # if space_group_number == 229:
    # if space_group_number == 230:
    
    return moves