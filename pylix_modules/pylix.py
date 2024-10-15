import ast
import re
import numpy as np
from scipy.constants import c, h, e, m_e
from CifFile import CifFile
from pylix_modules import pylix_dicts as fu

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
    error_flag = False

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
        print(f"File not found: {filename}")
        error_flag = True
    except IOError as e:
        print(f"IO error ({e}) reading file: {filename}")
        error_flag = True

    return inp_dict, error_flag


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
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_b_iso_or_equiv",
        "_atom_site_u_iso_or_equiv",
        "_atom_site_occupancy"]
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
        "_atom_site_wyckoff_symbol",
        "_atom_site_label",
        "_atom_site_type_symbol"]

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
                    if isinstance(data, list):
                        data = data[0]
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

    return cif_dict


def symop_convert(symop_xyz):
    # Converts symmetry operation xyz form into matrix+vector form
    symmetry_count = len(symop_xyz)
    mat = np.zeros((symmetry_count, 3, 3), dtype="float")
    vec = np.zeros((symmetry_count, 3), dtype="float")
    coord_map = {'x': 0, 'y': 1, 'z': 2}
    # we expect comma-delimited symmetry operations
    for i in range(symmetry_count):
        symop = symop_xyz[i]
        # Remove any numbers, extra spaces, and quotation marks
        symop = re.sub(r'^[^a-zA-Z-]*', '', symop).replace("'", "").replace('"', '').strip()
        # split into 3 parts
        parts = symop.split(',')
        for j, pt in enumerate(parts):
            pt = pt.strip()  # Remove extra spaces
            # Regex to capture the k/l/m (x, y, z) and the fractional part
            match = re.match(r'([+-]?[xyz])?([+-]\d+/\d+)?([+-]?\d+)?', pt)
            if match:
                # Extract the variable part (x, y, z)
                var_part = match.group(1)
                if var_part:
                    pm1 = -1 if var_part.startswith('-') else 1
                    axis = coord_map[var_part[-1]]  # Get the axis from 'x', 'y', or 'z'
                    mat[i, j, axis] = pm1
                # Extract the fractional part
                frac_part = match.group(2)
                if frac_part:
                    numerator, denominator = map(int, frac_part.split('/'))
                    vec[i, j] = numerator / denominator

    return mat, vec


def unique_atom_positions(symmetry_matrix, symmetry_vector, basis_atom_label,
                          basis_atom_name, basis_atom_position, basis_B_iso,
                          basis_occupancy):
    """
    Fills the unit cell by applying symmetry operations to the basis

    Parameters:
    symmetry_matrix float(n_symmetry_operations x 3x3): n_symmetry_operations
    symmetry_vector float(n_symmetry_operations x 3):  associated translations
    basis_atom_label (str): a label for each basis atom
    basis_atom_name (str): element symbol for each basis atom
    basis_atom_position float(n_basis_atoms x 3): fractional coordinates
    basis_B_iso float(n_basis_atoms): Debye-Waller factors for each basis atom
    basis_occupancy float(n_basis_atoms): occupancy for each basis atom

    Returns:
    atom_position, atom_label, atom_name, B_iso, occupancy
    
    """

    # tolerance in fractional coordinates to consider atoms to be the same
    tol = 0.001
    # Determine the size of the all_atom_position array
    n_symmetry_operations = symmetry_vector.shape[0]
    n_basis_atoms = basis_atom_position.shape[0]
    total_atoms = n_symmetry_operations * n_basis_atoms

    # Initialize arrays to store all atom positions, including duplicates
    all_atom_label = np.tile(basis_atom_label, n_symmetry_operations)
    all_atom_name = np.tile(basis_atom_name, n_symmetry_operations)
    all_occupancy = np.tile(basis_occupancy, n_symmetry_operations)
    all_B_iso = np.tile(basis_B_iso, n_symmetry_operations)

    # # Generate all equivalent positions by applying symmetry
    symmetry_applied = np.einsum('ijk,lk->ilj', symmetry_matrix,
                                 basis_atom_position) + symmetry_vector[:, np.newaxis, :]
    all_atom_position = symmetry_applied.reshape(total_atoms, 3)

    # Normalize positions to be within [0, 1]
    all_atom_position %= 1.0
    # make small values precisely zero
    all_atom_position[np.abs(all_atom_position) < tol] = 0.0
    # Reduce to the set of unique fractional atomic positions using tol
    dist_matrix = np.linalg.norm(all_atom_position[:, np.newaxis, :] - 
                                 all_atom_position[np.newaxis, :, :], axis=-1)
    unique_mask = np.ones(len(all_atom_position), dtype=bool)
    i = []  # indices of unique atom positiona
    for j in range(total_atoms):
        if unique_mask[j]:  # If this point is still unique
            i.append(j)
            # Mark all points within tol as not unique
            unique_mask &= (dist_matrix[j] > tol)

    # Apply the same reduction to the labels, names, occupancies, and B_iso
    atom_position = all_atom_position[i]
    atom_label = all_atom_label[i]
    atom_name = all_atom_name[i]
    occupancy = all_occupancy[i]
    B_iso = all_B_iso[i]

    return atom_position, atom_label, atom_name, B_iso, occupancy


def reference_frames(cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                     cell_gamma, space_group, x_dir_c, z_dir_c, norm_dir_c):
    """
    Produces reciprocal lattice vectors and related parameters

    Parameters:
    space_group : str
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
    """

    tiny = 1e-10

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
    t_mat_o2m = np.column_stack((x_dir_o, y_dir_o, z_dir_o)).T

    # Unit normal to the specimen in microscope frame
    norm_dir_m = t_mat_o2m @ t_mat_c2o @ norm_dir_c
    norm_dir_m /= np.linalg.norm(norm_dir_m)

    # Transform from crystal reference frame to microscope frame
    a_vec_m = np.dot(t_mat_o2m, a_vec_o)
    b_vec_m = np.dot(t_mat_o2m, b_vec_o)
    c_vec_m = np.dot(t_mat_o2m, c_vec_o)

    # Reciprocal lattice vectors: microscope frame in 1/Angstrom units
    ar_vec_m = (2.0*np.pi * np.cross(b_vec_m, c_vec_m) /
                np.dot(b_vec_m, np.cross(c_vec_m, a_vec_m)))
    br_vec_m = (2.0*np.pi * np.cross(c_vec_m, a_vec_m) /
                np.dot(c_vec_m, np.cross(a_vec_m, b_vec_m)))
    cr_vec_m = (2.0*np.pi * np.cross(a_vec_m, b_vec_m) /
                np.dot(a_vec_m, np.cross(b_vec_m, c_vec_m)))

    return a_vec_m, b_vec_m, c_vec_m, ar_vec_m, br_vec_m, cr_vec_m, norm_dir_m


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
# 161
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
              atomic_number, occupancy, B_iso, g_matrix, g_magnitude,
              absorption_method, absorption_per, electron_velocity):
    Fg_matrix = np.zeros([n_hkl, n_hkl], dtype=np.complex128)
    # calculate g.r for all g-vectors and atom posns [n_hkl, n_hkl, n_atoms]
    g_dot_r = np.einsum('ijk,lk->ijl', g_matrix, atom_coordinate)
    # exp(i g.r) [n_hkl, n_hkl, n_atoms]
    phase = np.exp(-1j * g_dot_r)
    # scattering factor for all g-vectors, to be used atom by atom
    # NB scattering factor methods accept and return 2D[n_hkl, n_hkl] array of
    # g magnitudes but only one atom type.  Potential speed up by broadcasting
    # all atom types and modifying scattering factor methods to accept 2D + 1D
    # arrays [n_hkl, n_hkl] & [n_atoms],
    # returning an array [n_hkl, n_hkl, n_atoms]
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
        else:
            raise ValueError("No scattering factors chosen in felix.inp")

        # get the absorptive scattering factor (null for absorption_method==0)
        # no absorption
        if absorption_method == 0:
            f_g_prime = np.zeros_like(f_g)
        # proportional model
        elif absorption_method == 1:
            f_g_prime = 1j * f_g * absorption_per/100.0
        # Bird & King model, parameterised by Thomas (Acta Cryst 2023)
        elif absorption_method == 2:
            f_g_prime = 1j * f_thomas(g_magnitude, B_iso[i],
                                      atomic_number[i], electron_velocity)

        # The Structure Factor Equation
        # multiply by Debye-Waller factor, phase and occupancy
        Fg_matrix = Fg_matrix+((f_g + f_g_prime) * phase[:, :, i] *
                               occupancy[i] *
                               np.exp(-B_iso[i] *
                                      (g_magnitude**2)/(16*np.pi**2)))

    return Fg_matrix


def deviation_parameter(convergence_angle, image_radius, big_k_mag, g_pool, g_pool_mag):
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
    g_pool_mag1 = g_pool_mag[1:] # shape: (n_hkl-1)

    # Components of k_0
    k_0_0 = -g_pool_mag1 / 2
    k_0_2 = np.sqrt(big_k_mag**2 - k_0_0**2)
    # Components of k_prime
    k_prime_0 = np.einsum('ijk,lk->ijl', tilted_k, g_pool1) / g_pool_mag1
    k_prime_2 = np.sqrt(big_k_mag**2 - k_prime_0**2)

    k_0_dot_k_prime = (k_0_0[None, None, :] * k_prime_0 +
                       k_0_2[None, None, :] * k_prime_2)
    pm = -np.sign(2*k_prime_0 + g_pool_mag1)
    s_g = pm * np.sqrt(2*(big_k_mag**2 -k_0_dot_k_prime)) * (g_pool_mag1/big_k_mag)

    # add in 000
    s_0 = np.expand_dims(np.zeros_like(k_x), axis=-1)  
    s_g = np.concatenate([s_0, s_g], axis = -1)  # add 000 
   
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
    sum_d = np.einsum('ij,ij->i', ug_wj, ug_wj_weak) / (2.0 * big_k_mag * weak_beam_sg)
    
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

    f_g = np.sum(a/(q**2+b), axis=0) + np.sum(c*np.exp(-(d*q**2)), axis=0)

    return f_g


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
    s = g_magnitude / (4*np.pi)

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


def four_gauss(x, args):
    # returns the sum of four Gaussians & a constant for Thomas f_prime
    f = args[0]*np.exp(-abs(args[1])*x**2) + \
        args[2]*np.exp(-abs(args[3])*x**2) + \
        args[4]*np.exp(-abs(args[5])*x**2) + \
        args[6]*np.exp(-abs(args[7])*x**2) + args[8]
    return f


def f_thomas(g, B, Z, v):
    # interpolated of parameterised Bird & King absorptive scattering factors
    # calculation uses s = g/2
    s = g/2
    # returns an interpolated absorptive scattering factor
    Bvalues = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                        0.7, 1, 1.5, 2, 2.75, 4])

    # error checking
    if isinstance(s, np.ndarray) and np.any(s < 0):
        raise Exception("inavlid values of s")
    elif isinstance(s, (int, float)) and s < 0:
        raise Exception("invalid value of s")
    if B < 0:
        raise Exception("invalid value of B")
    if Z < 1 or Z > 103:
        raise Exception("invalid value of Z")
    if B > 4 or 0 < B < 0.1:
        raise Exception("B outside range of parameterisation")
    if isinstance(s, np.ndarray) and B == 0:
        return np.zeros(np.shape(s))
    elif isinstance(s, (int, float)) and B == 0:
        return 0

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
    f_prime = c*np.where(line > 0, line, 0)/v

    return f_prime


def read_dm3(file_path, y_pixels, x_pixels):
    """
    Reads a DM3 file and extracts the image matrix.
    Parameters:
    file_path (str): The path to the DM3 file.
    y_pixels (int): Number of pixels in the Y dimension.
    x_pixels (int): Number of pixels in the X dimension.
    
    Returns:
    image_dm3 (ndarray): A 2D numpy array containing the image data.
    
    Raises:
    ValueError: If there are any issues with reading the file or the data format.
    """

    # Parameters
    print_tags = True
    image_data_tag = 2
    
    # Initialize output variables
    image_dm3 = np.zeros((y_pixels, x_pixels), dtype=np.float32)
    
    # Initialize local variables
    finding_tags = True
    looking_for_data_tags = True
    
    n_bytes = 0
    n_tags = 0
    n_data_tags = 0
    n_header_bytes = 0
    iy = 0
    ix = 0
    previous_bytes = np.zeros(40, dtype=np.uint8)
    previous_4_bytes = np.zeros(4, dtype=np.uint8)
    
    data_length = 0
    tag_label = ""
    in_tag_reading_mode = False
    
    # Try to open the file
    try:
        with open(file_path, 'rb') as file:
            while n_bytes < 4000000:
                if finding_tags:
                    # Shift bytes and read new byte
                    previous_bytes = np.roll(previous_bytes, -1)
                    previous_4_bytes = np.roll(previous_4_bytes, -1)
                    byte = np.fromfile(file, dtype=np.uint8, count=1)
                    if byte.size == 0:  # End of file
                        if print_tags:
                            print(f'End Of File Reached, Total bytes = {n_bytes}')
                        break
                    n_bytes += 1
                    previous_bytes[-1] = byte[0]
                    previous_4_bytes[-1] = byte[0]
                    
                    # Check if at a DM3 tag delimiter '%%%%' (ASCII value 37)
                    if np.all(previous_4_bytes == [37, 37, 37, 37]):
                        if in_tag_reading_mode:
                            # Capture bytes leading up to the delimiter (those that are valid ASCII)
                            label = ''
                            for j in range(len(previous_bytes)-5, -1, -1):
                                if 32 <= previous_bytes[j] <= 126:
                                    label = chr(previous_bytes[j]) + label
                                else:
                                    break  # Stop on non-ASCII
                            tag_label = label.strip()
                            # Print tag if needed
                            n_tags += 1
                            if print_tags:
                                print(f'{tag_label}  {n_tags}  {n_bytes}')
                                
                            # Check if it's the 'Data' tag
                            if looking_for_data_tags:
                                if 'Data' in tag_label:
                                    n_data_tags += 1
                                if n_data_tags == image_data_tag:
                                    looking_for_data_tags = False
                                    finding_tags = False
                            
                            # Reset tag label for next tag
                            tag_label = ""
                            in_tag_reading_mode = False
                        
                        # Set the mode to read tag label after finding delimiter
                        in_tag_reading_mode = True
                        read_after_delimiter = True
                    elif in_tag_reading_mode:
                        # Collect ASCII characters to form the tag label
                        if 32 <= byte[0] <= 126 and read_after_delimiter:
                            tag_label += chr(byte[0])
                        # Stop reading after tag is fully captured
                        elif not read_after_delimiter:
                            in_tag_reading_mode = False
                            tag_label = ""
                        else:
                            read_after_delimiter = False

                else:
                    # read_image_data:
                    if n_header_bytes < 16:
                        # Read 4-byte integer
                        pre_data = np.fromfile(file, dtype=np.uint32, count=1)
                        if pre_data.size == 0:
                            raise ValueError(f'Error reading pre-data bytes, Byte number = {n_bytes}')
    
                        if n_header_bytes == 12:
                            # Convert to big endian format
                            data_length = int.from_bytes(pre_data.tobytes(), 'big')
                            if data_length != y_pixels * x_pixels:
                                raise ValueError(f'Data array length does not match input pixel size. '
                                                 f'Data length = {data_length}, '
                                                 f'Expected = {y_pixels * x_pixels}')
    
                        n_bytes += 4
                        n_header_bytes += 4
                    else:
                        # Read image data
                        data_bytes = np.fromfile(file, dtype=np.float32, count=1)
                        if data_bytes.size == 0:
                            raise ValueError(f'Error reading image data, Byte number = {n_bytes}')
                        
                        n_bytes += 4
                        iy += 1
                        if iy == y_pixels + 1:
                            iy = 1
                            ix += 1
    
                        if ix < x_pixels + 1:
                            image_dm3[ix - 1, iy - 1] = data_bytes[0]

    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    
    return image_dm3


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
    # if space_group_number == 2)
    # if space_group_number == 3)
    # if space_group_number == 4)
    # if space_group_number == 5)
    # if space_group_number == 6)
    # if space_group_number == 7)
    # if space_group_number == 8)
    # if space_group_number == 9)
    # if space_group_number == 10)
    # if space_group_number == 11)
    # if space_group_number == 12)
    # if space_group_number == 13)
    # if space_group_number == 14)
    # if space_group_number == 15)
    if space_group_number == 15:  # C 1 2/c 1
        if ('e') in wyckoff:  # point symmetry 2 along y
            moves[0, :] = ([0.0, 1.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'c' or 'd') not in wyckoff:  # point symmetry -1
            raise ValueError("Wyckoff Symbol for space group C2/c, not recognised")
    # if space_group_number == 16)
    # if space_group_number == 17)
    # if space_group_number == 18)
    # if space_group_number == 19)
    # if space_group_number == 20)
    # if space_group_number == 21)
    # if space_group_number == 22)
    # if space_group_number == 23)
    # if space_group_number == 24)
    # if space_group_number == 25)
    # if space_group_number == 26)
    # if space_group_number == 27)
    # if space_group_number == 28)
    # if space_group_number == 29)
    # if space_group_number == 30)
    # if space_group_number == 31)
    # if space_group_number == 32)
    # if space_group_number == 33)
    # if space_group_number == 34)
    # if space_group_number == 35)
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
    # if space_group_number == 37)
    # if space_group_number == 38)
    # if space_group_number == 39)
    # if space_group_number == 40)
    # if space_group_number == 41)
    # if space_group_number == 42)
    # if space_group_number == 43)
    # if space_group_number == 44)
    # if space_group_number == 45)
    # if space_group_number == 46)
    # if space_group_number == 47)
    # if space_group_number == 48)
    # if space_group_number == 49)
    # if space_group_number == 50)
    # if space_group_number == 51)
    # if space_group_number == 52)
    # if space_group_number == 53)
    # if space_group_number == 54)
    # if space_group_number == 55)
    # if space_group_number == 56)
    # if space_group_number == 57)
    # if space_group_number == 58)
    # if space_group_number == 59)
    # if space_group_number == 60)
    # if space_group_number == 61)
    # if space_group_number == 62)
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
    # if space_group_number == 65)
    # if space_group_number == 66)
    # if space_group_number == 67)
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
    # if space_group_number == 69)
    # if space_group_number == 70)
    # if space_group_number == 71)
    # if space_group_number == 72)
    # if space_group_number == 73)
    # if space_group_number == 74)
    # if space_group_number == 75)
    # if space_group_number == 76)
    # if space_group_number == 77)
    # if space_group_number == 78)
    # if space_group_number == 79)
    # if space_group_number == 80)
    # if space_group_number == 81)
    # if space_group_number == 82)
    # if space_group_number == 83)
    # if space_group_number == 84)
    # if space_group_number == 85)
    # if space_group_number == 86)
    # if space_group_number == 87)
    # if space_group_number == 88)
    # if space_group_number == 89)
    # if space_group_number == 90)
    # if space_group_number == 91)
    # if space_group_number == 92)
    # if space_group_number == 93)
    # if space_group_number == 94)
    # if space_group_number == 95)
    # if space_group_number == 96)
    # if space_group_number == 97)
    # if space_group_number == 98)
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
    # if space_group_number == 100)
    # if space_group_number == 101)
    # if space_group_number == 102)
    # if space_group_number == 103)
    # if space_group_number == 104)
    # if space_group_number == 105)
    # if space_group_number == 106)
    # if space_group_number == 107)
    # if space_group_number == 108)
    # if space_group_number == 109)
    # if space_group_number == 110)
    # if space_group_number == 111)
    # if space_group_number == 112)
    # if space_group_number == 113)
    # if space_group_number == 114)
    # if space_group_number == 115)
    # if space_group_number == 116)
    # if space_group_number == 117)
    # if space_group_number == 118)
    # if space_group_number == 119)
    # if space_group_number == 120)
    # if space_group_number == 121)
    # if space_group_number == 122)
    # if space_group_number == 123)
    # if space_group_number == 124)
    # if space_group_number == 125)
    # if space_group_number == 126)
    # if space_group_number == 127)
    # if space_group_number == 128)
    # if space_group_number == 129)
    # if space_group_number == 130)
    # if space_group_number == 131)
    # if space_group_number == 132)
    # if space_group_number == 133)
    # if space_group_number == 134)
    # if space_group_number == 135)
    # if space_group_number == 136)
    # if space_group_number == 137)
    # if space_group_number == 138)
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
    # if space_group_number == 140)
    # if space_group_number == 141)
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
    # if space_group_number == 143)
    # if space_group_number == 144)
    # if space_group_number == 145)
    # if space_group_number == 146)
    # if space_group_number == 147)
    # if space_group_number == 148)
    # if space_group_number == 149)
    # if space_group_number == 150)
    # if space_group_number == 151)
    # if space_group_number == 152)
    # if space_group_number == 153)
    # if space_group_number == 154)
    # if space_group_number == 155)
    # if space_group_number == 156)
    # if space_group_number == 157)
    # if space_group_number == 158)
    # if space_group_number == 159)
    # if space_group_number == 160)
    # if space_group_number == 161)
    # if space_group_number == 162)
    # if space_group_number == 163)
    # if space_group_number == 164)
    # if space_group_number == 165)
    # if space_group_number == 166)
    # if space_group_number == 167)
    if space_group_number == 167:  # R-3c
        # a: point symmetry 32, no allowed movements
        # b: point symmetry -3, no allowed movements
        # d: point symmetry -1, no allowed movements
        if ('c') in wyckoff:  # point symmetry 3, allowed movement along z
            moves[0, :] = ([0.0, 0.0, 1.0])
        elif ('e') in wyckoff:  # point symmetry 2, allowed movement along [x,0,0]
            moves[0, :] = ([1.0, 0.0, 0.0])
        elif ('f') in wyckoff:  # point symmetry 1
            moves[0, :] = ([1.0, 0.0, 0.0])
            moves[1, :] = ([0.0, 1.0, 0.0])
            moves[2, :] = ([0.0, 0.0, 1.0])
        elif ('a' or 'b' or 'd') not in wyckoff:
            raise ValueError("Wyckoff Symbol for space group R-3c not recognised")
    # if space_group_number == 168)
    # if space_group_number == 169)
    # if space_group_number == 170)
    # if space_group_number == 171)
    # if space_group_number == 172)
    # if space_group_number == 173)
    # if space_group_number == 174)
    # if space_group_number == 175)
    # if space_group_number == 176)
    # if space_group_number == 177)
    # if space_group_number == 178)
    # if space_group_number == 179)
    # if space_group_number == 180)
    # if space_group_number == 181)
    # if space_group_number == 182)
    # if space_group_number == 183)
    # if space_group_number == 184)
    # if space_group_number == 185)
    # if space_group_number == 186)
    # if space_group_number == 187)
    # if space_group_number == 188)
    # if space_group_number == 189)
    # if space_group_number == 190)
    # if space_group_number == 191)
    # if space_group_number == 192)
    # if space_group_number == 193)
    # if space_group_number == 194)
    # if space_group_number == 195)
    # if space_group_number == 196)
    # if space_group_number == 197)
    # if space_group_number == 198)
    # if space_group_number == 199)
    # if space_group_number == 200)
    # if space_group_number == 201)
    # if space_group_number == 202)
    # if space_group_number == 203)
    # if space_group_number == 204)
    # if space_group_number == 205)
    # if space_group_number == 206)
    # if space_group_number == 207)
    # if space_group_number == 208)
    # if space_group_number == 209)
    # if space_group_number == 210)
    # if space_group_number == 211)
    # if space_group_number == 212)
    # if space_group_number == 213)
    # if space_group_number == 214)
    # if space_group_number == 215)
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
    # if space_group_number == 217)
    # if space_group_number == 218)
    # if space_group_number == 219)
    # if space_group_number == 220)
    # if space_group_number == 221)
    # if space_group_number == 222)
    # if space_group_number == 223)
    # if space_group_number == 224)
    # if space_group_number == 225)
    # if space_group_number == 226)
    # if space_group_number == 227)
    # if space_group_number == 228)
    # if space_group_number == 229)
    # if space_group_number == 230)
    
    return moves