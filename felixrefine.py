# -*- coding: utf-8 -*-
# %% modules and subroutines
"""
Created August 2024

@author: R Beanland

"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
from matplotlib.ticker import PercentFormatter
import time

start = time.time()
# %% Main felix program
# go to the pylix folder
# path = r"C:\Users\rbean\Documents\GitHub\Felix-python"
path = r"C:\Users\Richard\Documents\GitHub\Felix-python"
os.chdir(path)

# felix modules
from pylix_modules import pylix as px
from pylix_modules import simulate as sim
from pylix_modules import pylix_dicts as fu
latest_commit_id = px.get_git()
# outputs
print("-----------------------------------------------------------------")
print(f"felixrefine:  version {latest_commit_id[:8]}")
print("felixrefine: see https://github.com/WarwickMicroscopy/Felix-python")
print("-----------------------------------------------------------------")

# variables to get from felix.inp
accelerating_voltage_kv = None
acceptance_angle = None
absorption_method = None
absorption_per = None
atomic_sites = None
blur_radius = None
byte_size = None
convergence_angle = None
correlation_type = None
debye_waller_constant = None
debug = None
delta_thickness = None
exit_criteria = None
final_thickness = None
g_limit = None
holz_flag = None
image_processing = None
image_radius = None
incident_beam_direction = None
initial_thickness = None
min_reflection_pool = None
min_strong_beams = None
min_weak_beams = None
no_of_ugs = None
normal_direction = None
plot = None
precision = None
print_flag = None
refine_method_flag = None
refine_mode = None
scatter_factor_method = None
simplex_length_scale = None
weighting_flag = None
x_direction = None

# variables to get from felix.cif
atom_site_b_iso_or_equiv = None
atom_site_label = None
atom_site_type_symbol = None
atom_site_fract_x = None
atom_site_fract_y = None
atom_site_fract_z = None
atom_site_occupancy = None
atom_site_u_iso_or_equiv = None
atom_site_wyckoff_symbol = None
cell_angle_alpha = None
cell_angle_beta = None
cell_angle_gamma = None
cell_length_a = None
cell_length_b = None
cell_length_c = None
cell_volume = None
chemical_formula_iupac = None
chemical_formula_structural = None
chemical_formula_sum = None
space_group_it_number = None
space_group_name_h_m_alt = None
space_group_symbol = None
space_group_symop_operation_xyz = None
symmetry_equiv_pos_as_xyz = None
symmetry_space_group_name_h_m = None

# %% read felix.cif

# cif_dict is a dictionary of value-key pairs.  values are given as tuples
# with the second number the uncertainty in the first.  Nothing is currently
# done with these uncertainties...
cif_dict = px.read_cif('felix.cif')
# modifying the dictionary to remove invalid characters
original_keys = list(cif_dict.keys())
for key in original_keys:
    new_key = key.replace('-', '_')
    if new_key != key:
        cif_dict[new_key] = cif_dict[key]
        del cif_dict[key]
# make global variables
for var_name, var_value in cif_dict.items():
    globals()[var_name] = var_value

# ====== extract values
# chemical formula
if "chemical_formula_structural" in cif_dict:
    chemical_formula = re.sub(r'(?<!\d)1(?!\d)', '',
                              chemical_formula_structural.replace(' ', ''))
if "chemical_formula_sum" in cif_dict:  # preferred, replce structural if poss
    chemical_formula = re.sub(r'(?<!\d)1(?!\d)', '',
                              chemical_formula_sum.replace(' ', ''))
print("Material: " + chemical_formula)
# space group number and lattice type
if "space_group_symbol" in cif_dict:
    space_group = space_group_symbol.replace(' ', '')
elif "space_group_name_h_m_alt" in cif_dict:
    space_group = space_group_name_h_m_alt.replace(' ', '')
elif "symmetry_space_group_name_h_m" in cif_dict:
    space_group = symmetry_space_group_name_h_m.replace(' ', '')
elif "space_group_it_number" in cif_dict:
    space_group_number = int(space_group_it_number[0])
    reverse_space_groups = {v: k for k, v in fu.space_groups.items()}
    space_group = reverse_space_groups.get(space_group_number, "Unknown")
else:
    error_flag = True
    raise ValueError("No space group found in .cif")
lattice_type = space_group[0]
space_group_number = fu.space_groups[space_group]

# cell
cell_a = px.dlst(cell_length_a)
cell_b = px.dlst(cell_length_b)
cell_c = px.dlst(cell_length_c)

cell_alpha = px.dlst(cell_angle_alpha)*np.pi/180.0  # angles in radians
cell_beta = px.dlst(cell_angle_beta)*np.pi/180.0
cell_gamma = px.dlst(cell_angle_gamma)*np.pi/180.0
basis_count = len(atom_site_label)
# # moved to simulate ***
# if "cell_volume" in cif_dict:
#     cell_volume = px.dlst(cell_volume)
# else:
#     cell_volume = cell_a*cell_b*cell_c*np.sqrt(1.0-np.cos(cell_alpha)**2
#                   - np.cos(cell_beta)**2 - np.cos(cell_gamma)**2
#                   +2.0*np.cos(cell_alpha)*np.cos(cell_beta)*np.cos(cell_gamma))

# symmetry operations
if space_group_symop_operation_xyz is not None:
    symmetry_matrix, symmetry_vector = px.symop_convert(
        space_group_symop_operation_xyz)
elif symmetry_equiv_pos_as_xyz is not None:
    symmetry_matrix, symmetry_vector = px.symop_convert(
        symmetry_equiv_pos_as_xyz)
else:
    error_flag = True
    raise ValueError("Symmetry operations not found in .cif")

# extract the basis from the raw cif values
# take basis atom labels as given
basis_atom_label = atom_site_label
# atom symbols, stripping any charge etc.
basis_atom_name = [''.join(filter(str.isalpha, name))
                   for name in atom_site_type_symbol]
# take care of any odd symbols, get the case right
for i in range(basis_count):
    name = basis_atom_name[i]
    if len(name) == 1:
        name = name.upper()
    elif len(name) > 1:
        name = name[0].upper() + name[1:].lower()
    basis_atom_name[i] = name
# take basis Wyckoff letters as given (maybe check they are only letters?)
basis_wyckoff = atom_site_wyckoff_symbol

# basis_atom_position = np.zeros([basis_count, 3])
# x_ = np.array([tup[0] for tup in atom_site_fract_x])
# y_ = np.array([tup[0] for tup in atom_site_fract_y])
# z_ = np.array([tup[0] for tup in atom_site_fract_z])
basis_atom_position = np.column_stack((np.array([tup[0] for tup in atom_site_fract_x]),
                                       np.array([tup[0] for tup in atom_site_fract_y]),
                                       np.array([tup[0] for tup in atom_site_fract_z])))
# Debye-Waller factor
if "atom_site_b_iso_or_equiv" in cif_dict:
    basis_B_iso = np.array([tup[0] for tup in atom_site_b_iso_or_equiv])
elif "atom_site_u_iso_or_equiv" in cif_dict:
    basis_B_iso = np.array([tup[0] for tup in
                            atom_site_u_iso_or_equiv])*8*(np.pi**2)
# occupancy, assume it's unity if not specified
if atom_site_occupancy is not None:
    basis_occupancy = np.array([tup[0] for tup in atom_site_occupancy])
else:
    basis_occupancy = np.ones([basis_count])

basis_atom_delta = np.zeros([basis_count, 3])  # ***********what's this

# %% read felix.inp
inp_dict, error_flag = px.read_inp_file('felix.inp')
for var_name, var_value in inp_dict.items():
    globals()[var_name] = var_value  # Create global variables
if error_flag is True:
    raise ValueError("can't read felix.inp")

# thickness array
if (final_thickness > initial_thickness + delta_thickness):
    thickness = np.arange(initial_thickness, final_thickness, delta_thickness)
    n_thickness = len(thickness)
else:
    # need np.array rather than float so wave_functions works for 1 or many t's
    thickness = np.array(initial_thickness)
    n_thickness = 1

# convert arrays to numpy
incident_beam_direction = np.array(incident_beam_direction, dtype='float')
normal_direction = np.array(normal_direction, dtype='float')
x_direction = np.array(x_direction, dtype='float')
atomic_sites = np.array(atomic_sites, dtype='int')

# crystallography exp(2*pi*i*g.r) to physics convention exp(i*g.r)
g_limit = g_limit * 2 * np.pi

# output
print(f"Zone axis: {incident_beam_direction.astype(int)}")
if n_thickness ==1:
    print(f"Specimen thickness {initial_thickness/10} nm")
else:
    print(f"{n_thickness} thicknesses: {', '.join(map(str, thickness/10))} nm")
if 'S' in refine_mode:
    print("Simulation only, S")
elif 'A' in refine_mode:
    print("Refining Structure Factors, A")
    # needs error check for any other refinement
    # raise ValueError("Structure factor refinement
    # incompatible with anything else")
else:
    if 'B' in refine_mode:
        print("Refining Atomic Coordinates, B")
        # redefine the basis if necessary to allow coordinate refinement
        basis_atom_position = px.preferred_basis(space_group_number,
                                                 basis_atom_position,
                                                 basis_wyckoff)
    if 'C' in refine_mode:
        print("Refining Occupancies, C")
    if 'D' in refine_mode:
        print("Refining Isotropic Debye Waller Factors, D")
    if 'E' in refine_mode:
        print("Refining Anisotropic Debye Waller Factors, E")
        raise ValueError("Refinement mode E not implemented")
    if (len(atomic_sites) > basis_count):
        raise ValueError("Number of atomic sites to refine is larger than the \
                         number of atoms")
if 'F' in refine_mode:
    print("Refining Lattice Parameters, F")
if 'G' in refine_mode:
    print("Refining Lattice Angles, G")
if 'H' in refine_mode:
    print("Refining Convergence Angle, H")
if 'I' in refine_mode:
    print("Refining Accelerating Voltage, I")

if scatter_factor_method == 0:
    print("Using Kirkland scattering factors")
elif scatter_factor_method == 1:
    print("Using Lobato scattering factors")
elif scatter_factor_method == 2:
    print("Using Peng scattering factors")
elif scatter_factor_method == 3:
    print("Using Doyle & Turner scattering factors")
else:
    raise ValueError("No scattering factors chosen in felix.inp")


# %% read felix.hkl
input_hkls, i_obs, sigma_obs = px.read_hkl_file("felix.hkl")
n_out = len(input_hkls)+1  # we expect 000 NOT to be in the hkl list


# %% set up refinement, TO BE TESTED
# --------------------------------------------------------------------
# n_variables calculated depending upon Ug and non-Ug refinement
# --------------------------------------------------------------------
# Ug refinement is a special case, cannot do any other refinement alongside
# We count the independent variables:
# independent_variable = variable to be refined
# independent_variable_type = what kind of variable, as follows
# 0 = Ug amplitude
# 1 = Ug phase
# 2 = atom coordinate *** PARTIALLY IMPLEMENTED *** not all space groups
# 3 = occupancy
# 4 = B_iso
# 5 = B_aniso *** NOT YET IMPLEMENTED ***
# 61,62,63 = lattice parameters *** PARTIALLY IMPLEMENTED *** not rhombohedral
# 7 = unit cell angles *** NOT YET IMPLEMENTED ***
# 8 = convergence angle
# 9 = accelerating_voltage_kv *** NOT YET IMPLEMENTED ***
independent_variable = ([])
independent_variable_type = ([])
atom_refine_flag = ([])
if 'S' not in refine_mode:
    n_vars = 0
    # count refinement variables
    if 'B' in refine_mode:  # Atom coordinate refinement
        for i in range(len(atomic_sites)):
            # the [3, 3] matrix 'moves' returned by atom_move gives the
            # allowed movements for an atom (depending on its Wyckoff
            # symbol and space group) as row vectors with magnitude 1.
            # ***NB NOT ALL SPACE GROUPS IMPLEMENTED ***
            moves = px.atom_move(space_group_number, basis_wyckoff[i])
            degrees_of_freedom = int(np.sum(moves**2))
            if degrees_of_freedom == 0:
                raise ValueError("Atom coord refinement not possible")
            for j in range(degrees_of_freedom):
                r_dot_v = np.dot(basis_atom_position[atomic_sites[i]],
                                 moves[j, :])
                independent_variable.append(r_dot_v)
                independent_variable_type.append(2)
                atom_refine_flag.append(atomic_sites[i])

    if 'C' in refine_mode:  # Occupancy
        for i in range(len(atomic_sites)):
            independent_variable.append(basis_occupancy[atomic_sites[i]])
            independent_variable_type.append(3)
            atom_refine_flag.append(atomic_sites[i])

    if 'D' in refine_mode:  # Isotropic DW
        for i in range(len(atomic_sites)):
            independent_variable.append(basis_B_iso[atomic_sites[i]])
            independent_variable_type.append(5)
            atom_refine_flag.append(atomic_sites[i])

    if 'E' in refine_mode:  # Anisotropic DW
        # Not yet implemented!!!
        raise ValueError("Anisotropic Debye-Waller factor refinement \
                         not yet implemented")

    if 'F' in refine_mode:  # Lattice parameters
        # This section needs work to include rhombohedral cells and
        # non-standard settings!!!
        independent_variable.append(cell_a)  # is in all lattice types
        independent_variable_type.append(71)
        if space_group_number < 75:  # Triclinic, monoclinic, orthorhombic
            independent_variable.append(cell_b)
            independent_variable_type.append(72)
            independent_variable.append(cell_c)
            independent_variable_type.append(73)
        elif 142 < space_group_number < 168:  # Rhombohedral
            err = 1  # Need to work out R- vs H- settings!!!
            print("Rhombohedral R- and H- cells not yet implemented \
                  for unit cell refinement")
        elif (167 < space_group_number < 195) or \
             (74 < space_group_number < 143):  # Hexagonal or Tetragonal
            independent_variable.append(cell_c)
            independent_variable_type.append(73)

    if 'G' in refine_mode:  # Unit cell angles
        # Not yet implemented!!!
        raise ValueError("Unit cell angle refinement not yet implemented")

    if 'H' in refine_mode:  # Convergence angle
        independent_variable.append(convergence_angle)
        independent_variable_type.append(9)

    if 'I' in refine_mode:  # accelerating_voltage_kv
        independent_variable.append(accelerating_voltage_kv)
        independent_variable_type.append(10)
        # Not yet implemented!!!
        raise ValueError("accelerating_voltage_kv refinement not yet implemented")

# Total number of independent variables
n_variables = len(independent_variable)
if n_variables == 0:
    raise ValueError("No refinement variables! \
    Check refine_mode flag in felix.inp. \
        Valid refine modes are A,B,C,D,F,H,S")
if n_variables == 1:
    print("Only one independent variable")
else:
    print(f"Number of independent variables = {n_variables}")

independent_variable = np.array(independent_variable)
independent_delta = np.zeros(n_variables)
independent_variable_type = np.array(independent_variable_type)
independent_variable_atom = np.array(atom_refine_flag[:n_variables])


# # %% set up Ug refinement
# if 'A' in refine_mode:  # Ug refinement
#     print("Refining Structure Factors, A")
#     # needs error check for any other refinement
#     # raise ValueError("Structure factor refinement incompatible
#     # with anything else")
#     # we refine magnitude and phase for each Ug.  However for space groups
#     # with a centre of symmetry phases are fixed at 0 or pi, so only
#     # amplitude is refined (1 independent variable per Ug)
#     # Identify the 92 centrosymmetric space groups
#     centrosymmetric = [2, 10, 11, 12, 13, 14, 15, 47, 48, 49, 50, 51, 52,
#                        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
#                        66, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85,
#                        86, 87, 88, 123, 124, 125, 126, 127, 128, 129, 130,
#                        131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
#                        141, 142, 147, 148, 162, 163, 164, 165, 166, 167,
#                        175, 176, 191, 192, 193, 194, 200, 201, 202,
#                        203, 204, 205, 206, 221, 222, 223, 224, 225, 226,
#                        227, 228, 229, 230]
#     if space_group_number in centrosymmetric:
#         vars_per_ug = 1
#     else:
#         vars_per_ug = 2

#     # set up Ug refinement
#     # equivalent g's identified by abs(h)+abs(k)+abs(l)+a*h^2+b*k^2+c*l^2
#     g_eqv = (10000*(np.sum(np.abs(g_matrix), axis=2) +
#                     g_magnitude**2)).astype(int)
#     # we keep track of individual Ug's in a matrix ug_eqv
#     ug_eqv = np.zeros([n_hkl, n_hkl], dtype=int)
#     # bit of a hack here - can skip Ug's in refinement using ug_offset, but
#     # should really be an input in felix.inp if it's going to be used
#     ug_offset = 0
#     # Ug number (position in ug_matrix)
#     i_ug = 1 + ug_offset
#     # the first column of the Ug matrix has g-vectors in ascending order
#     # we work through this list until we have identified the required no.
#     # of Ugs. The matrix ug_eqv identifies equivalent Ug's with an integer
#     # whose sign is that of the imaginary part of Ug. (may need to take
#     # care of floating point residuals where im(Ug) is nominally zero???)
#     j = 1  # number of Ug's processed
#     while j < no_of_ugs+1:
#         if ug_eqv[i_ug, 0] != 0:  # already in the list, skip it
#             i_ug += 1
#             continue
#         g_id = abs(g_eqv[i_ug, 0])
#         # update relevant locations in ug_eqv
#         ug_eqv[np.abs(g_eqv) == g_id] = j*np.sign(
#             np.imag(ug_matrix[i_ug, 0]))
#         # amplitude is type 1, always a variable
#         independent_variable.append(ug_matrix[i_ug, 0])
#         independent_variable_type.append(0)
#         if vars_per_ug == 2:  # we also adjust phase
#             independent_variable.append(ug_matrix[i_ug, 0])
#             independent_variable_type.append(1)
#         j += 1


# %% simulate

# variable passing needs to be changed to something more efficient!
hkl, g_output, lacbed_sim, setup, bwc = \
    sim.simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
                 symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                 cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
                 basis_B_iso, basis_occupancy, scatter_factor_method, accelerating_voltage_kv,
                 x_direction, incident_beam_direction, normal_direction,
                 min_reflection_pool, min_strong_beams, g_limit, input_hkls,
                 absorption_method, absorption_per, convergence_angle,
                 image_radius, thickness)

# %% read in experimental images
if 'S' not in refine_mode:
    lacbed_expt = np.zeros([2*image_radius, 2*image_radius, n_out])
    # get the list of available images
    x_str = str(2*image_radius)
    dm3_folder = None
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            # Check if 'dm3' and the number x are in the folder name
            if 'dm3' in dirname.lower() and x_str in dirname:
                # Return the full path of the matching folder
                dm3_folder = os.path.join(dirpath, dirname)
    if dm3_folder is not None:
        dm3_files = [file for file in os.listdir(dm3_folder)
                     if file.lower().endswith('.dm3')]
        # just match the indices in the filename to felix.hkl, expect the user
        # to ensure the data is of the right material!
        n_expt = n_out
        for i in range(n_out):
            g_string = px.hkl_string(hkl[g_output[i]])
            found = False
            for file_name in dm3_files:
                if g_string in file_name:
                    file_path = os.path.join(dm3_folder, file_name)
                    lacbed_expt[:, :, i] = px.read_dm3(file_path, 2*image_radius,
                                                       debug)
                    found = True
            if not found:
                n_expt -= 1
                print(f"{g_string} not found")

        # print experimental LACBED patterns
        w = int(np.ceil(np.sqrt(n_out)))
        h = int(np.ceil(n_out/w))
        fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
        text_effect = withStroke(linewidth=3, foreground='black')
        axes = axes.flatten()
        for i in range(n_out):
            axes[i].imshow(lacbed_expt[:, :, i], cmap='gist_earth')
            axes[i].axis('off')
            annotation = f"{hkl[g_output[i], 0]}{hkl[g_output[i], 1]}{hkl[g_output[i], 2]}"
            axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                             size=30, color='w', path_effects=[text_effect])
        for i in range(n_out, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        # initialise correlation
        best_corr = np.ones(n_out)


# %% output simulated LACBED patterns
w = int(np.ceil(np.sqrt(n_out)))
h = int(np.ceil(n_out/w))
for j in range(n_thickness):
    fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
    text_effect = withStroke(linewidth=3, foreground='black')
    axes = axes.flatten()
    for i in range(n_out):
        axes[i].imshow(lacbed_sim[j, :, :, i], cmap='pink')
        axes[i].axis('off')
        annotation = f"{hkl[g_output[i], 0]}{hkl[g_output[i], 1]}{hkl[g_output[i], 2]}"
        axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                         size=30, color='w', path_effects=[text_effect])
    for i in range(n_out, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


# %% figure of merit and best thickness

# figures of merit - might need a NaN check? size [n_thick, n_out]
fom_array = px.figure_of_merit(lacbed_sim, lacbed_expt, image_processing,
                         blur_radius, correlation_type, plot)
best_t = np.argmin(np.mean(fom_array, axis=1))
# mean figure of merit
fom = np.mean(fom_array[best_t])
print(f"Figure of merit {100*fom:.2f}%")
# not sure if this is really needed?
best_corr = np.minimum(best_corr, fom_array[best_t])

# plot
if plot:
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    plt.plot(thickness/10, np.mean(fom_array, axis=1))
    ax.set_xlabel('Thickness (nm)', size=24)
    ax.set_ylabel('Figure of merit', size=24)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.show()


# %% update variables
j, k, m = 1, 1, 1  # Counting indices for refinement types
basis_atom_delta.fill(0)  # Reset atom coordinate uncertainties to zero

for i in range(n_variables):
    # Check the type of variable by the last digit of independent_variable_type
    variable_type = independent_variable_type[i] % 10

    if variable_type == 0:
        # Structure factor refinement (handled elsewhere)
        variable_check = 1

    elif variable_type == 2:
        # Atomic coordinates
        atom_id = atom_refine_flag[j]

        # # Update position: r' = r - v*(r.v) + v*independent_variable
        # dot_product = np.dot(basis_atom_position[atom_id, :], vector[j - 1, :])
        # basis_atom_position[atom_id, :] = np.mod(
        #     basis_atom_position[atom_id, :] - vector[j - 1, :] * dot_product + 
        #     vector[jnd - 1, :] * independent_variable[i], 1
        # )

        # # Update uncertainty if iependent_delta is non-zero
        # if abs(independent_delta[i]) > 1e-10:  # Tiny threshold
        #     basis_atom_delta[atom_id, :] += vector[j - 1, :] * independent_delta[i]
        # j += 1

    elif variable_type == 3:
        # Occupancy
        basis_occupancy[independent_variable_atom[k]] = independent_variable[i]
        k += 1

    elif variable_type == 4:
        # Iso Debye-Waller factor
        basis_B_iso[atom_refine_flag[m]] = independent_variable[i]
        m += 1

    elif variable_type == 5:
        # Aniso Debye-Waller factor (not implemented)
        raise NotImplementedError("Anisotropic DWF not implemented")

    elif variable_type == 6:
        # Lattice parameters a, b, c
        if independent_variable_type[i] == 6:
            length_x = length_y = length_z = independent_variable[i]
        elif independent_variable_type[i] == 16:
            length_y = independent_variable[i]
        elif independent_variable_type[i] == 26:
            length_z = independent_variable[i]

    elif variable_type == 7:
        # Lattice angles alpha, beta, gamma
        variable_check[6] = 1
        if j == 1:
            cell_alpha = independent_variable[i]
        elif j == 2:
            cell_beta = independent_variable[i]
        elif j == 3:
            cell_gamma = independent_variable[i]

    elif variable_type == 8:
        # Convergence angle
        convergence_angle = independent_variable[i]

    elif variable_type == 9:
        # Accelerating voltage
        accelerating_voltage = independent_variable[i]


# %% final print
print("-----------------------------------------------------------------")
print(f"Beam pool calculation took {setup:.3f} seconds")
print(f"Bloch wave calculation in {bwc:.1f} s ({1000*(bwc)/(4*image_radius**2):.2f} ms/pixel)")
print("-----------------------------------------------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
