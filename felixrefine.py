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
refinement_scale = None
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


# %% set up refinement
# --------------------------------------------------------------------
# n_variables calculated depending upon Ug and non-Ug refinement
# --------------------------------------------------------------------
# Ug refinement is a special case, cannot do any other refinement alongside
# We count the independent variables:
# variable = variable to be refined
# variable_type = what kind of variable, as follows
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
variable = ([])
variable_type = ([])
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
                variable.append(r_dot_v)
                variable_type.append(2)
                atom_refine_flag.append(atomic_sites[i])

    if 'C' in refine_mode:  # Occupancy
        for i in range(len(atomic_sites)):
            variable.append(basis_occupancy[atomic_sites[i]])
            variable_type.append(3)
            atom_refine_flag.append(atomic_sites[i])

    if 'D' in refine_mode:  # Isotropic DW
        for i in range(len(atomic_sites)):
            variable.append(basis_B_iso[atomic_sites[i]])
            variable_type.append(4)
            atom_refine_flag.append(atomic_sites[i])

    if 'E' in refine_mode:  # Anisotropic DW
        # Not yet implemented!!! variable_type 5
        raise ValueError("Anisotropic Debye-Waller factor refinement \
                         not yet implemented")

    if 'F' in refine_mode:  # Lattice parameters
        # variable_type first digit=6 indicates lattice parameter
        # second digit=1,2,3 indicates a,b,c
        # This section needs work to include rhombohedral cells and
        # non-standard settings!!!
        variable.append(cell_a)  # is in all lattice types
        variable_type.append(61)
        atom_refine_flag.append(-1)  # -1 indicates not an atom
        if space_group_number < 75:  # Triclinic, monoclinic, orthorhombic
            variable.append(cell_b)
            variable_type.append(62)
            atom_refine_flag.append(-1)
            variable.append(cell_c)
            variable_type.append(63)
            atom_refine_flag.append(-1)
        elif 142 < space_group_number < 168:  # Rhombohedral
            # Need to work out R- vs H- settings!!!
            raise ValueError("Rhombohedral R- vs H- not yet implemented")
        elif (167 < space_group_number < 195) or \
             (74 < space_group_number < 143):  # Hexagonal or Tetragonal
            variable.append(cell_c)
            variable_type.append(63)
            atom_refine_flag.append(-1)

    if 'G' in refine_mode:  # Unit cell angles
        # Not yet implemented!!! variable_type 7
        raise ValueError("Unit cell angle refinement not yet implemented")

    if 'H' in refine_mode:  # Convergence angle
        variable.append(convergence_angle)
        variable_type.append(8)
        atom_refine_flag.append(-1)

    if 'I' in refine_mode:  # accelerating_voltage_kv
        variable.append(accelerating_voltage_kv)
        variable_type.append(9)
        atom_refine_flag.append(-1)
        # Not yet implemented!!!
        raise ValueError("accelerating_voltage_kv refinement not yet implemented")

# Total number of independent variables
n_variables = len(variable)
if n_variables == 0:
    raise ValueError("No refinement variables! \
    Check refine_mode flag in felix.inp. \
        Valid refine modes are A,B,C,D,F,H,S")
if n_variables == 1:
    print("Only one independent variable")
else:
    print(f"Number of independent variables = {n_variables}")

variable = np.array(variable)
independent_delta = np.zeros(n_variables)
variable_type = np.array(variable_type)
variable_atom = np.array(atom_refine_flag[:n_variables])


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
#         variable.append(ug_matrix[i_ug, 0])
#         variable_type.append(0)
#         if vars_per_ug == 2:  # we also adjust phase
#             variable.append(ug_matrix[i_ug, 0])
#             variable_type.append(1)
#         j += 1


# %% baseline simulation
print("-------------------------------")
print("Baseline simulation:")
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


# %% start refinement loop

# figure of merit
fom = px.figure_of_merit(lacbed_sim, thickness, lacbed_expt,
                               image_processing, blur_radius, correlation_type,
                               plot)
print(f"  Figure of merit {100*fom:.2f}%")
print("-------------------------------")

# Initialise variables for refinement
iter_count = 0
best_fit = fom
last_fit = best_fit
last_p = np.ones(n_variables)
p = np.ones(n_variables)
current_var = np.ones(n_variables)
df = 1.0
r3_var = np.zeros(3)  # for parabolic minimum
r3_fit = np.zeros(3)
independent_delta = 0.0

# Refinement loop
while df >= exit_criteria:
    # reset the variables (n-dimensional parameter space)
    current_var = np.copy(variable) # used for simulations, start at current best point
    var0 = np.copy(variable)  #running best fit during this refinement cycle
    fit0 = fom  # incoming fit

    # if all parameters have been refined and we're still in the loop, reset
    if np.sum(np.abs(last_p)) < 1e-10:
        last_p = np.ones(n_variables)

    #===========individual variable minimisation
    # we go through the variables - if there's an easy minimisation
    # we take it and remove it from the list of variables to refine in
    # vector descent.  If it's been refined, last_p[i] = 0
    for i in range(n_variables):
        # Skip variables already optimized in previous refinements
        if abs(last_p[i]) < 1e-10:
            p[i] = 0.0
            continue
        # Check if Debye-Waller factor is zero, skip if so
        if current_var[i] <= 1e-10 and variable_type == 4:
            p[i] = 1e-10
            continue

        # Update iteration
        iter_count += 1

        # Display messages based on variable type (last digit)
        if variable_type[i] % 10 == 1:
            print("Changing Ug")
        elif variable_type[i] % 10 == 2:
            print("Changing atomic coordinate")
        elif variable_type[i] % 10 == 3:
            print("Changing occupancy")
        elif variable_type[i] % 10 == 4:
            print("Changing isotropic Debye-Waller factor")
        elif variable_type[i] % 10 == 5:
            print("Changing anisotropic Debye-Waller factor")
        elif variable_type[i] % 10 == 6:
            print("Changing lattice parameter")
        elif variable_type[i] % 10 == 8:
            print("Changing convergence angle")

        print(f"Finding gradient, {i+1} of {n_variables}")

        # dx is a small change in the current variable determined by RScale
        # which is either refinement_scale for atomic coordinates and
        # refinement_scale*variable for everything else
        dx = abs(refinement_scale * current_var[i])
        if variable_type == 2:
            dx = abs(refinement_scale)

        # Three-point gradient measurement, starting with plus
        current_var[i] = var0[i] + dx
        #update variables
        [basis_atom_position, basis_atom_delta, basis_occupancy, basis_B_iso,
                cell_a, cell_b, cell_c,
                convergence_angle, accelerating_voltage_kv] = \
            px.update_variables(current_var, variable_type,
                                atom_refine_flag,
                                basis_atom_position, basis_atom_delta,
                                basis_occupancy, basis_B_iso, cell_a, cell_b,
                                cell_c, convergence_angle, accelerating_voltage_kv)
        # simulate
        hkl, g_output, lacbed_sim, setup, bwc = \
            sim.simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
                         symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                         cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
                         basis_B_iso, basis_occupancy, scatter_factor_method, accelerating_voltage_kv,
                         x_direction, incident_beam_direction, normal_direction,
                         min_reflection_pool, min_strong_beams, g_limit, input_hkls,
                         absorption_method, absorption_per, convergence_angle,
                         image_radius, thickness)
        # figure of merit
        fom = px.figure_of_merit(lacbed_sim, thickness, lacbed_expt,
                                       image_processing, blur_radius, correlation_type,
                                       plot)        
        print(f"  Figure of merit {100*fom:.2f}%")
        if (fom < best_fit):
            best_fit = fom
        print(f"    Best figure of merit {100*best_fit:.2f}%")
        print("-------------------------------")
        r_plus = fom
    
        # Now minus dx
        current_var[i] = var0[i] - dx

        #update variables
        [basis_atom_position, basis_atom_delta, basis_occupancy, basis_B_iso,
                cell_a, cell_b, cell_c,
                convergence_angle, accelerating_voltage_kv] = \
            px.update_variables(current_var, variable_type,
                                atom_refine_flag,
                                basis_atom_position, basis_atom_delta,
                                basis_occupancy, basis_B_iso, cell_a, cell_b,
                                cell_c, convergence_angle, accelerating_voltage_kv)
        # simulate
        hkl, g_output, lacbed_sim, setup, bwc = \
            sim.simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
                         symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                         cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
                         basis_B_iso, basis_occupancy, scatter_factor_method, accelerating_voltage_kv,
                         x_direction, incident_beam_direction, normal_direction,
                         min_reflection_pool, min_strong_beams, g_limit, input_hkls,
                         absorption_method, absorption_per, convergence_angle,
                         image_radius, thickness)
        # figure of merit
        fom = px.figure_of_merit(lacbed_sim, thickness, lacbed_expt,
                                       image_processing, blur_radius, correlation_type,
                                       plot)        
        print(f"  Figure of merit {100*fom:.2f}%")
        if (fom < best_fit):
            best_fit = fom
        print(f"    Best figure of merit {100*best_fit:.2f}%")
        print("-------------------------------")
        r_minus = fom

        # Reset current point so it's correct for the next variable
        current_var[i] = var0[i]
        # If the three points contain a minimum, predict its position using Kramer's rule
        if min(fit0, r_plus, r_minus) == fit0:
            r3_var = np.array([var0[i] - dx, var0[i], var0[i] + dx])
            r3_fit = np.array([r_minus, fit0, r_plus])

            var_min, fit_min = px.parabo3(r3_var, r3_fit)
            
            # error estimate goes here
            # independent_delta[i] = delta_x(r3_var, r3_fit, precision, err)
            # uncert_brak(var_min, independent_delta[i])

            # We update RVar0 with the best points as we go along
            # But keep RCurrentVar the same so that the measurements of gradient
            # are accurate.  
            var0[i] = var_min
            p[i] = 0.0
            print(f"Expect minimum at {var_min} with fit index {100*fit_min:.2f}%")
        else:  # this variable will be part of vector gradient descent
            p[i] = -(r_plus - r_minus) / (2 * dx)  # -df/dx
            if min(fit0, r_plus, r_minus) == r_plus:
                var0[i] = var0[i] + dx
            elif min(fit0, r_plus, r_minus) == r_minus:
                var0[i] = var0[i] - dx

    #===========vector descent
    # point 1 of 3 using the prediction var0
    print("Refining, point 1 of 3")
    current_var = np.copy(var0)
    #update variables
    [basis_atom_position, basis_atom_delta, basis_occupancy, basis_B_iso,
            cell_a, cell_b, cell_c,
            convergence_angle, accelerating_voltage_kv] = \
        px.update_variables(current_var, variable_type,
                            atom_refine_flag,
                            basis_atom_position, basis_atom_delta,
                            basis_occupancy, basis_B_iso, cell_a, cell_b,
                            cell_c, convergence_angle, accelerating_voltage_kv)
    # simulate
    hkl, g_output, lacbed_sim, setup, bwc = \
        sim.simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
                     symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                     cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
                     basis_B_iso, basis_occupancy, scatter_factor_method, accelerating_voltage_kv,
                     x_direction, incident_beam_direction, normal_direction,
                     min_reflection_pool, min_strong_beams, g_limit, input_hkls,
                     absorption_method, absorption_per, convergence_angle,
                     image_radius, thickness)
    # figure of merit
    fom = px.figure_of_merit(lacbed_sim, thickness, lacbed_expt,
                                   image_processing, blur_radius, correlation_type,
                                   plot)        
    print(f"  Figure of merit {100*fom:.2f}%")
    if (fom < best_fit):
        best_fit = fom
    print(f"    Best figure of merit {100*best_fit:.2f}%")
    print("-------------------------------")
    
    # Reset the gradient vector magnitude and initialize vector descent
    p_mag = np.linalg.norm(p)
    if np.isinf(p_mag) or np.isnan(p_mag):
        error = 1
        raise ValueError(f"Infinite or NaN gradient! Refinement vector = {p}")

    if abs(p_mag) > 1e-10:
        p = p / p_mag   # Normalize direction of max/min gradient
        print(f"Refinement vector [{p:.2f}]")
        # Find index of the first non-zero element in the gradient vector
        j = np.where(np.abs(p) >= 1e-10)[0][0]
        # reset the refinement scale
        p_mag = var0[j] * refinement_scale

        # Three points for concavity test
        r3_var[0] = var0[j]
        r3_fit[0] = fom

        # point 2
        print("Refining, point 2 of 3")
        current_var = var0 + p * p_mag
        #update variables
        [basis_atom_position, basis_atom_delta, basis_occupancy, basis_B_iso,
                cell_a, cell_b, cell_c,
                convergence_angle, accelerating_voltage_kv] = \
            px.update_variables(current_var, variable_type,
                                atom_refine_flag,
                                basis_atom_position, basis_atom_delta,
                                basis_occupancy, basis_B_iso, cell_a, cell_b,
                                cell_c, convergence_angle, accelerating_voltage_kv)
        # simulate
        hkl, g_output, lacbed_sim, setup, bwc = \
            sim.simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
                         symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                         cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
                         basis_B_iso, basis_occupancy, scatter_factor_method, accelerating_voltage_kv,
                         x_direction, incident_beam_direction, normal_direction,
                         min_reflection_pool, min_strong_beams, g_limit, input_hkls,
                         absorption_method, absorption_per, convergence_angle,
                         image_radius, thickness)
        # figure of merit
        fom = px.figure_of_merit(lacbed_sim, thickness, lacbed_expt,
                                       image_processing, blur_radius, correlation_type,
                                       plot)        
        print(f"  Figure of merit {100*fom:.2f}%")
        if (fom < best_fit):
            best_fit = fom
        print(f"    Best figure of merit {100*best_fit:.2f}%")
        print("-------------------------------")
        r3_var[1] = current_var[j]
        r3_fit[1] = fom
        
        # Point 3 of 3
        print("Refining, point 3 of 3")
        if r3_fit[1] > r3_fit[0]:  #if second point is worse
            p_mag = -p_mag  # Go in the opposite direction
        else:  # keep going
            var0 = np.copy(current_var)
        current_var = var0 + p * p_mag

        #update variables
        [basis_atom_position, basis_atom_delta, basis_occupancy, basis_B_iso,
                cell_a, cell_b, cell_c,
                convergence_angle, accelerating_voltage_kv] = \
            px.update_variables(current_var, variable_type,
                                atom_refine_flag,
                                basis_atom_position, basis_atom_delta,
                                basis_occupancy, basis_B_iso, cell_a, cell_b,
                                cell_c, convergence_angle, accelerating_voltage_kv)
        # simulate
        hkl, g_output, lacbed_sim, setup, bwc = \
            sim.simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
                         symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                         cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
                         basis_B_iso, basis_occupancy, scatter_factor_method, accelerating_voltage_kv,
                         x_direction, incident_beam_direction, normal_direction,
                         min_reflection_pool, min_strong_beams, g_limit, input_hkls,
                         absorption_method, absorption_per, convergence_angle,
                         image_radius, thickness)
        # figure of merit
        fom = px.figure_of_merit(lacbed_sim, thickness, lacbed_expt,
                                       image_processing, blur_radius, correlation_type,
                                       plot)        
        print(f"  Figure of merit {100*fom:.2f}%")
        if (fom < best_fit):
            best_fit = fom
        print(f"    Best figure of merit {100*best_fit:.2f}%")
        print("-------------------------------")
        r3_var[2] = current_var[j]
        r3_fit[2] = fom

        # Concavity check and further refinement
        max_var_idx = np.argmax(r3_var)
        min_var_idx = np.argmin(r3_var)
        if max_var_idx != min_var_idx:
            mid_idx = 3 - max_var_idx - min_var_idx
            convexity_test = -abs(r3_fit[max_var_idx] - r3_fit[min_var_idx])
            convexity_adjust = r3_fit[mid_idx] - (
                r3_fit[min_var_idx] + (r3_var[mid_idx] - r3_var[min_var_idx]) * 
                (r3_fit[max_var_idx] - r3_fit[min_var_idx]) / 
                (r3_var[max_var_idx] - r3_var[min_var_idx])
            )

            while convexity_adjust > 0.1 * convexity_test:
                print("Convex, continuing")
                max_fit_idx = np.argmax(r3_fit)
                min_fit_idx = np.argmin(r3_fit)
                if max_fit_idx != min_fit_idx:
                    mid_fit_idx = 3 - max_fit_idx - min_fit_idx
                    p_mag *= 2
                    # if abs(p_mag) > max_ug_step and refine_mode[0] == 1:
                    #     p_mag = np.sign(p_mag) * max_ug_step

                    current_var = var0 + p * p_mag
                    r3_var[mid_fit_idx] = current_var[j]

                    #update variables
                    [basis_atom_position, basis_atom_delta, basis_occupancy, basis_B_iso,
                            cell_a, cell_b, cell_c,
                            convergence_angle, accelerating_voltage_kv] = \
                        px.update_variables(current_var, variable_type,
                                            atom_refine_flag,
                                            basis_atom_position, basis_atom_delta,
                                            basis_occupancy, basis_B_iso, cell_a, cell_b,
                                            cell_c, convergence_angle, accelerating_voltage_kv)
                    # simulate
                    hkl, g_output, lacbed_sim, setup, bwc = \
                        sim.simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
                                     symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                                     cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
                                     basis_B_iso, basis_occupancy, scatter_factor_method, accelerating_voltage_kv,
                                     x_direction, incident_beam_direction, normal_direction,
                                     min_reflection_pool, min_strong_beams, g_limit, input_hkls,
                                     absorption_method, absorption_per, convergence_angle,
                                     image_radius, thickness)
                    # figure of merit
                    fom = px.figure_of_merit(lacbed_sim, thickness, lacbed_expt,
                                                   image_processing, blur_radius, correlation_type,
                                                   plot)        
                    print(f"  Figure of merit {100*fom:.2f}%")
                    if (fom < best_fit):
                        best_fit = fom
                        variable=np.copy(current_var)
                    print(f"    Best figure of merit {100*best_fit:.2f}%")
                    print("-------------------------------")
                    r3_fit[mid_fit_idx] = fom
                    max_var_idx = np.argmax(r3_var)
                    min_var_idx = np.argmin(r3_var)
                    if max_var_idx != min_var_idx:
                        mid_idx = 3 - max_var_idx - min_var_idx
                        convexity_adjust = r3_fit[mid_idx] - (
                            r3_fit[min_var_idx] + 
                            (r3_var[mid_idx] - r3_var[min_var_idx]) * 
                            (r3_fit[max_var_idx] - r3_fit[min_var_idx]) / 
                            (r3_var[max_var_idx] - r3_var[min_var_idx])
                        )
                        convexity_test = -abs(r3_fit[max_var_idx] - r3_fit[min_var_idx])

        # Predict minimum fit point
        var_min, fit_min = px.parabo3(r3_var, r3_fit)
        print(f"Concave set, predict minimum at {var_min} with fit index {fit_min}")
        current_var = var0 + p * (var_min - var0[j]) / p[j]
    else:
        current_var = var0

    # Simulate with updated parameters

    #update variables
    [basis_atom_position, basis_atom_delta, basis_occupancy, basis_B_iso,
            cell_a, cell_b, cell_c,
            convergence_angle, accelerating_voltage_kv] = \
        px.update_variables(current_var, variable_type,
                            atom_refine_flag,
                            basis_atom_position, basis_atom_delta,
                            basis_occupancy, basis_B_iso, cell_a, cell_b,
                            cell_c, convergence_angle, accelerating_voltage_kv)
    # simulate
    hkl, g_output, lacbed_sim, setup, bwc = \
        sim.simulate(plot, debug, space_group, lattice_type, symmetry_matrix,
                     symmetry_vector, cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                     cell_gamma, basis_atom_label, basis_atom_name, basis_atom_position,
                     basis_B_iso, basis_occupancy, scatter_factor_method, accelerating_voltage_kv,
                     x_direction, incident_beam_direction, normal_direction,
                     min_reflection_pool, min_strong_beams, g_limit, input_hkls,
                     absorption_method, absorption_per, convergence_angle,
                     image_radius, thickness)
    # figure of merit
    fom = px.figure_of_merit(lacbed_sim, thickness, lacbed_expt,
                                   image_processing, blur_radius, correlation_type,
                                   plot)        
    print(f"  Figure of merit {100*fom:.2f}%")
    if (fom < best_fit):
        best_fit = fom
        variable=np.copy(current_var)
    print(f"    Best figure of merit {100*best_fit:.2f}%")

    # Update for next iteration
    last_p = p
    df = last_fit - best_fit
    last_fit = best_fit

    print(f"Improvement in fit {df}, will stop at {exit_criteria}")
    refinement_scale *= (1 - 1 / (2 * n_variables))
    
    df = 0



# %% final print
print("-----------------------------------------------------------------")
print(f"Beam pool calculation took {setup:.3f} seconds")
print(f"Bloch wave calculation in {bwc:.1f} s ({1000*(bwc)/(4*image_radius**2):.2f} ms/pixel)")
print("-----------------------------------------------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
