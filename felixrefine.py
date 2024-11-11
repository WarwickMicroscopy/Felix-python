# -*- coding: utf-8 -*-
# %% modules and subroutines
"""
Created August 2024

@author: R Beanland

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import time

# felix modules
from pylix_modules import pylix as px
from pylix_modules import simulate as sim
from pylix_modules import pylix_dicts as fu
from pylix_modules import pylix_class as pc

path = os.getcwd()
start = time.time()
latest_commit_id = px.get_git()
# outputs
print("-----------------------------------------------------------------")
print(f"felixrefine:  version {latest_commit_id[:8]}")
print("felixrefine:  https://github.com/WarwickMicroscopy/Felix-python")
print("-----------------------------------------------------------------")

# initialise class objects
v = pc.Var()  # working variables used in the simulation
# initialise iteration count
v.iter_count = 0


# %% read felix.cif

# cif_dict is a dictionary of value-key pairs.  values are given as tuples
# with the second number the uncertainty in the first.  Nothing is currently
# done with these uncertainties...
cif_dict = px.read_cif('felix.cif')
v.update_from_dict(cif_dict)
# ====== extract cif data into working variables v
v.space_group = v.symmetry_space_group_name_h_m
if v.chemical_formula_structural is not None:
    v.chemical_formula = v.chemical_formula_structural
elif v.chemical_formula_sum is not None:
    v.chemical_formula = v.chemical_formula_sum
elif v.chemical_formula_iupac is not None:
    v.chemical_formula = v.chemical_formula_iupac
print("Material: " + v.chemical_formula)

# space group number and lattice type
if "space_group_symbol" in cif_dict:
    v.space_group = v.space_group_symbol.replace(' ', '')
elif "space_group_name_h_m_alt" in cif_dict:
    v.space_group = v.space_group_name_h_m_alt.replace(' ', '')
elif "symmetry_space_group_name_h_m" in cif_dict:
    v.space_group = v.symmetry_space_group_name_h_m.replace(' ', '')
elif "space_group_it_number" in cif_dict:
    v.space_group_number = int(v.space_group_it_number[0])
    reverse_space_groups = {v: k for k, v in fu.space_groups.items()}
    v.space_group = reverse_space_groups.get(v.space_group_number, "Unknown")
else:
    error_flag = True
    raise ValueError("No space group found in .cif")
v.lattice_type = v.space_group[0]
v.space_group_number = fu.space_groups[v.space_group]

# cell
v.cell_a = v.cell_length_a[0]
v.cell_b = v.cell_length_b[0]
v.cell_c = v.cell_length_c[0]
v.cell_alpha = v.cell_angle_alpha[0]*np.pi/180.0  # angles in radians
v.cell_beta = v.cell_angle_beta[0]*np.pi/180.0
v.cell_gamma = v.cell_angle_gamma[0]*np.pi/180.0
n_basis = len(v.atom_site_label)

# symmetry operations
if "space_group_symop_operation_xyz" in cif_dict:
    v.symmetry_matrix, v.symmetry_vector = px.symop_convert(
        v.space_group_symop_operation_xyz)
elif "symmetry_equiv_pos_as_xyz" in cif_dict:
    v.symmetry_matrix, v.symmetry_vector = px.symop_convert(
        v.symmetry_equiv_pos_as_xyz)
else:
    error_flag = True
    raise ValueError("Symmetry operations not found in .cif")

# extract the basis from the raw cif values
# take basis atom labels as given
v.basis_atom_label = v.atom_site_label
# atom symbols, stripping any charge etc.
v.basis_atom_name = [''.join(filter(str.isalpha, name))
                   for name in v.atom_site_type_symbol]
# take care of any odd symbols, get the case right
for i in range(n_basis):
    name = v.basis_atom_name[i]
    if len(name) == 1:
        name = name.upper()
    elif len(name) > 1:
        name = name[0].upper() + name[1:].lower()
    v.basis_atom_name[i] = name
# take basis Wyckoff letters as given (maybe check they are only letters?)
v.basis_wyckoff = v.atom_site_wyckoff_symbol

# basis_atom_position = np.zeros([basis_count, 3])
v.basis_atom_position = \
    np.column_stack((np.array([tup[0] for tup in v.atom_site_fract_x]),
                     np.array([tup[0] for tup in v.atom_site_fract_y]),
                     np.array([tup[0] for tup in v.atom_site_fract_z])))

# Debye-Waller factor
if "atom_site_b_iso_or_equiv" in cif_dict:
    v.basis_B_iso = np.array([tup[0] for tup in v.atom_site_b_iso_or_equiv])
elif "atom_site_u_iso_or_equiv" in cif_dict:
    v.basis_B_iso = np.array([tup[0] for tup in
                            v.atom_site_u_iso_or_equiv])*8*(np.pi**2)
    
# occupancy, assume it's unity if not specified
if v.atom_site_occupancy is not None:
    v.basis_occupancy = np.array([tup[0] for tup in v.atom_site_occupancy])
else:
    v.basis_occupancy = np.ones([n_basis])

v.basis_atom_delta = np.zeros([n_basis, 3])  # ***********what's this


# %% read felix.inp
inp_dict = px.read_inp_file('felix.inp')
v.update_from_dict(inp_dict)

# thickness array
if (v.final_thickness > v.initial_thickness + v.delta_thickness):
    v.thickness = np.arange(v.initial_thickness, v.final_thickness,
                            v.delta_thickness)
    v.n_thickness = len(v.thickness)
else:
    # need np.array rather than float so wave_functions works for 1 or many t's
    v.thickness = np.array(v.initial_thickness)
    v.n_thickness = 1

# convert arrays to numpy
v.incident_beam_direction = np.array(v.incident_beam_direction, dtype='float')
v.normal_direction = np.array(v.normal_direction, dtype='float')
v.x_direction = np.array(v.x_direction, dtype='float')
v.atomic_sites = np.array(v.atomic_sites, dtype='int')

# crystallography exp(2*pi*i*g.r) to physics convention exp(i*g.r)
v.g_limit = v.g_limit * 2 * np.pi

# other refinement variables (is there a neater way of reading these across? I expect so)
# v.refine_mode = v.refine_mode
# v.scatter_factor_method = v.scatter_factor_method
# v.accelerating_voltage_kv = v.accelerating_voltage_kv
# v.min_reflection_pool = v.min_reflection_pool
# v.plot = v.plot
# v.debug = v.debug

# output
print(f"Zone axis: {v.incident_beam_direction.astype(int)}")
if v.n_thickness ==1:
    print(f"Specimen thickness {v.initial_thickness/10} nm")
else:
    print(f"{v.n_thickness} thicknesses: {', '.join(map(str, v.thickness/10))} nm")

if v.scatter_factor_method == 0:
    print("Using Kirkland scattering factors")
elif v.scatter_factor_method == 1:
    print("Using Lobato scattering factors")
elif v.scatter_factor_method == 2:
    print("Using Peng scattering factors")
elif v.scatter_factor_method == 3:
    print("Using Doyle & Turner scattering factors")
else:
    raise ValueError("No scattering factors chosen in felix.inp")

if 'S' in v.refine_mode:
    print("Simulation only, S")
elif 'A' in v.refine_mode:
    print("Refining Structure Factors, A")
    # needs error check for any other refinement
    # raise ValueError("Structure factor refinement
    # incompatible with anything else")
else:
    if 'B' in v.refine_mode:
        print("Refining Atomic Coordinates, B")
        # redefine the basis if necessary to allow coordinate refinement
        v.basis_atom_position = px.preferred_basis(v.space_group_number,
                                                   v.basis_atom_position,
                                                   v.basis_wyckoff)
    if 'C' in v.refine_mode:
        print("Refining Occupancies, C")
    if 'D' in v.refine_mode:
        print("Refining Isotropic Debye Waller Factors, D")
    if 'E' in v.refine_mode:
        print("Refining Anisotropic Debye Waller Factors, E")
        raise ValueError("Refinement mode E not implemented")
    if (len(v.atomic_sites) > n_basis):
        raise ValueError("Number of atomic sites to refine is larger than the \
                         number of atoms")
if 'F' in v.refine_mode:
    print("Refining Lattice Parameters, F")
if 'G' in v.refine_mode:
    print("Refining Lattice Angles, G")
if 'H' in v.refine_mode:
    print("Refining Convergence Angle, H")
if 'I' in v.refine_mode:
    print("Refining Accelerating Voltage, I")


# %% read felix.hkl
v.input_hkls, v.i_obs, v.sigma_obs = px.read_hkl_file("felix.hkl")
v.n_out = len(v.input_hkls)+1  # we expect 000 NOT to be in the hkl list


# %% set up refinement
# --------------------------------------------------------------------
# n_variables calculated depending upon Ug and non-Ug refinement
# --------------------------------------------------------------------
# Ug refinement is a special case, cannot do any other refinement alongside
# We count the independent variables:
# v.refined_variable = variable to be refined
# v.refined_variable_type = what kind of variable, as follows
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
v.refined_variable = ([])
v.refined_variable_type = ([])
v.atom_refine_flag = ([])
if 'S' not in v.refine_mode:
    v.n_variables = 0
    # count refinement variables
    if 'B' in v.refine_mode:  # Atom coordinate refinement
        for i in range(len(v.atomic_sites)):
            # the [3, 3] matrix 'moves' returned by atom_move gives the
            # allowed movements for an atom (depending on its Wyckoff
            # symbol and space group) as row vectors with magnitude 1.
            # ***NB NOT ALL SPACE GROUPS IMPLEMENTED ***
            moves = px.atom_move(v.space_group_number, v.basis_wyckoff[i])
            degrees_of_freedom = int(np.sum(moves**2))
            if degrees_of_freedom == 0:
                raise ValueError("Atom coord refinement not possible")
            for j in range(degrees_of_freedom):
                r_dot_v = np.dot(v.basis_atom_position[v.atomic_sites[i]],
                                 moves[j, :])
                v.refined_variable.append(r_dot_v)
                v.refined_variable_type.append(2)
                v.atom_refine_flag.append(v.atomic_sites[i])

    if 'C' in v.refine_mode:  # Occupancy
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_occupancy[v.atomic_sites[i]])
            v.refined_variable_type.append(3)
            v.atom_refine_flag.append(v.atomic_sites[i])

    if 'D' in v.refine_mode:  # Isotropic DW
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_B_iso[v.atomic_sites[i]])
            v.refined_variable_type.append(4)
            v.atom_refine_flag.append(v.atomic_sites[i])

    if 'E' in v.refine_mode:  # Anisotropic DW
        # Not yet implemented!!! variable_type 5
        raise ValueError("Anisotropic Debye-Waller factor refinement \
                         not yet implemented")

    if 'F' in v.refine_mode:  # Lattice parameters
        # variable_type first digit=6 indicates lattice parameter
        # second digit=1,2,3 indicates a,b,c
        # This section needs work to include rhombohedral cells and
        # non-standard settings!!!
        v.refined_variable.append(v.cell_a)  # is in all lattice types
        v.refined_variable_type.append(61)
        v.atom_refine_flag.append(-1)  # -1 indicates not an atom
        if v.space_group_number < 75:  # Triclinic, monoclinic, orthorhombic
            v.refined_variable.append(v.cell_b)
            v.refined_variable_type.append(62)
            v.atom_refine_flag.append(-1)
            v.refined_variable.append(v.cell_c)
            v.refined_variable_type.append(63)
            v.atom_refine_flag.append(-1)
        elif 142 < v.space_group_number < 168:  # Rhombohedral
            # Need to work out R- vs H- settings!!!
            raise ValueError("Rhombohedral R- vs H- not yet implemented")
        elif (167 < v.space_group_number < 195) or \
             (74 < v.space_group_number < 143):  # Hexagonal or Tetragonal
            v.refined_variable.append(v.cell_c)
            v.refined_variable_type.append(63)
            v.atom_refine_flag.append(-1)

    if 'G' in v.refine_mode:  # Unit cell angles
        # Not yet implemented!!! variable_type 7
        raise ValueError("Unit cell angle refinement not yet implemented")

    if 'H' in v.refine_mode:  # Convergence angle
        v.refined_variable.append(v.convergence_angle)
        v.refined_variable_type.append(8)
        v.atom_refine_flag.append(-1)
        print(f"Starting convergence angle {v.convergence_angle} Å^-1")

    if 'I' in v.refine_mode:  # accelerating_voltage_kv
        v.refined_variable.append(v.accelerating_voltage_kv)
        v.refined_variable_type.append(9)
        v.atom_refine_flag.append(-1)

# Total number of independent variables
v.n_variables = len(v.refined_variable)
if v.n_variables == 0:
    raise ValueError("No refinement variables! \
    Check refine_mode flag in felix.v. \
        Valid refine modes are A,B,C,D,F,H,S")
if v.n_variables == 1:
    print("Only one independent variable")
else:
    print(f"Number of independent variables = {v.n_variables}")

v.refined_variable = np.array(v.refined_variable)
independent_delta = np.zeros(v.n_variables)
v.refined_variable_type = np.array(v.refined_variable_type)
v.refined_variable_atom = np.array(v.atom_refine_flag[:v.n_variables])


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
# uses the whole v=Var class
setup, bwc = sim.simulate(v)

# %% read in experimental images
if 'S' not in v.refine_mode:
    v.lacbed_expt = np.zeros([2*v.image_radius, 2*v.image_radius, v.n_out])
    # get the list of available images
    x_str = str(2*v.image_radius)
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
        n_expt = v.n_out
        for i in range(v.n_out):
            g_string = px.hkl_string(v.hkl[v.g_output[i]])
            found = False
            for file_name in dm3_files:
                if g_string in file_name:
                    file_path = os.path.join(dm3_folder, file_name)
                    v.lacbed_expt[:, :, i] = px.read_dm3(file_path,
                                                       2*v.image_radius,
                                                       v.debug)
                    found = True
            if not found:
                n_expt -= 1
                print(f"{g_string} not found")

        # print experimental LACBED patterns
        w = int(np.ceil(np.sqrt(v.n_out)))
        h = int(np.ceil(v.n_out/w))
        fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
        text_effect = withStroke(linewidth=3, foreground='black')
        axes = axes.flatten()
        for i in range(v.n_out):
            axes[i].imshow(v.lacbed_expt[:, :, i], cmap='gist_earth')
            axes[i].axis('off')
            annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
            axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                             size=30, color='w', path_effects=[text_effect])
        for i in range(v.n_out, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        # initialise correlation
        best_corr = np.ones(v.n_out)


# %% output simulated LACBED patterns
sim.print_LACBED(v)


# %% start refinement loop

# figure of merit
fom = sim.figure_of_merit(v)
print(f"  Figure of merit {100*fom:.2f}%")
print("-------------------------------")

# Initialise variables for refinement
fit0 = fom*1.0
best_fit = fom*1.0
last_fit = fom*1.0
# p is a vector along the gradient in n-dimensional space
p = np.ones(v.n_variables)
last_p = np.ones(v.n_variables)
df = 1.0
r3_var = np.zeros(3)  # for parabolic minimum
r3_fom = np.zeros(3)
# dunno what this is
independent_delta = 0.0

# for a plot
var_pl = ([])
fit_pl = ([])

# Refinement loop
while df >= v.exit_criteria:
    # v.refined_variable is the working set of variables going into a sim
    # best_var is the running best fit during this refinement cycle
    best_var = np.copy(v.refined_variable)
    # next_var is the predicted next (best) point
    next_var = np.copy(v.refined_variable)
    # if all parameters have been refined and we're still in the loop, reset
    if np.sum(np.abs(last_p)) < 1e-10:
        last_p = np.ones(v.n_variables)

    # ===========individual variable minimisation
    # we go through the variables - if there's an easy minimisation
    # we take it and remove it from the list of variables to refine in
    # vector descent.  If it's been fixed, last_p[i] = 0
    # at the end of this for loop we have a predicted best starting point
    # for gradient descent best_var
    for i in range(v.n_variables):
        # Skip variables already optimized in previous refinements
        if abs(last_p[i]) < 1e-10:
            p[i] = 0.0
            continue
        v.current_variable_type = v.refined_variable_type[i]
        # Check if Debye-Waller factor is zero, skip if so
        if v.current_variable_type == 4 and v.refined_variable[i] <= 1e-10:
            p[i] = 0.0
            continue

        # middle point is the previous best simulation
        r3_var[1] = v.refined_variable[i]*1.0
        r3_fom[1] = last_fit*1.0
        var_pl.append(v.refined_variable[i])
        fit_pl.append(last_fit)

        # Update iteration
        

        print(f"Finding gradient, variable {i+1} of {v.n_variables}")

        # Display messages based on variable type (last digit)
        if v.current_variable_type % 10 == 1:
            print("Changing Ug")
        elif v.current_variable_type % 10 == 2:
            print("Changing atomic coordinate")
        elif v.current_variable_type % 10 == 3:
            print("Changing occupancy")
        elif v.current_variable_type % 10 == 4:
            print("Changing isotropic Debye-Waller factor")
        elif v.current_variable_type % 10 == 5:
            print("Changing anisotropic Debye-Waller factor")
        elif v.current_variable_type % 10 == 6:
            print("Changing lattice parameter")
        elif v.current_variable_type % 10 == 8:
            print("Changing convergence angle")

        # dx is a small change in the current variable determined by RScale
        # which is either refinement_scale for atomic coordinates and
        # refinement_scale*variable for everything else
        dx = abs(v.refinement_scale * v.refined_variable[i])
        if v.refined_variable_type[i] == 2:
            dx = abs(v.refinement_scale)

        # Three-point gradient measurement, starting with plus
        v.refined_variable[i] += dx
        # update variables
        sim.update_variables(v)
        sim.print_current_var(v, v.refined_variable[i])
        # simulate
        setup, bwc = sim.simulate(v)
        # figure of merit
        fom = sim.figure_of_merit(v)
        if (fom < best_fit):
            best_fit = fom*1.0
            best_var = np.copy(v.refined_variable)
        r3_var[2] = v.refined_variable[i]*1.0
        r3_fom[2] = fom*1.0
        print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
        print(f"-1-----------------------------")  # {r3_var},{r3_fom}")
        var_pl.append(v.refined_variable[i])
        fit_pl.append(fom)

        # keep going or turn round?
        if r3_fom[2] < r3_fom[1]:  # keep going
            v.refined_variable[i] += np.exp(0.5) * dx
        else:  # turn round
            dx = - dx
            v.refined_variable[i] += np.exp(0.75) * dx

        # update variables
        sim.update_variables(v)
        sim.print_current_var(v, v.refined_variable[i])
        # simulate
        setup, bwc = sim.simulate(v)
        # figure of merit
        fom = sim.figure_of_merit(v)
        if (fom < best_fit):
            best_fit = fom*1.0
            best_var = np.copy(v.refined_variable)
        r3_var[0] = v.refined_variable[i]*1.0
        r3_fom[0] = fom*1.0
        print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
        print(f"-2-----------------------------")  # {r3_var},{r3_fom}")
        var_pl.append(v.refined_variable[i])
        fit_pl.append(fom)

        # predict the next point as a minimum or a step on
        next_var[i], exclude = px.convex(r3_var, r3_fom)
        if exclude:
            p[i] = 0.0  # this variable doesn't get included in vector downhill
        else:
            # we weight the variable by -df/dx
            p[i] = -(r3_fom[2] - r3_fom[0]) / (2 * dx)
            # error estimate goes here
            # independent_delta[i] = delta_x(r3_var, r3_fom, precision, err)
            # uncert_brak(var_min, independent_delta[i])

    # ===========vector descent
    # either: point 1 of 3, or final simulation using the prediction next_var
    v.refined_variable = np.copy(next_var)
    # simulation
    sim.update_variables(v)
    setup, bwc = sim.simulate(v)
    fom = sim.figure_of_merit(v)
    if (fom < best_fit):
        best_fit = fom*1.0
        best_var = np.copy(v.refined_variable)
    print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")

    # Reset the gradient vector magnitude and initialize vector descent
    p_mag = np.linalg.norm(p)
    if np.isinf(p_mag) or np.isnan(p_mag):
        raise ValueError(f"Infinite or NaN gradient! Refinement vector = {p}")
    if abs(p_mag) > 1e-10:  # There are gradients, do the vector descent
        p = p / p_mag   # Normalized direction of max/min gradient
        j = np.where(np.abs(p) >= 1e-10)[0][0]
        v.current_variable_type = v.refined_variable_type[j]
        sim.print_current_var(v, v.refined_variable[i])
        print(f"Refining, refinement vector {p}")
        # Find index of the first non-zero element in the gradient vector
        # reset the refinement scale (last term reverses sign if we overshot)
        p_mag = -best_var[j] * v.refinement_scale  #* (2*(fom < best_fit)-1)
        # First of three points for concavity test is the best simulation
        r3_var[0] = best_var[j]*1.0
        r3_fom[0] = fom*1.0
        print(f"-a-----------------------------")  # {r3_var},{r3_fom}")

        # Second point
        print("Refining, point 2 of 3")
        # NB vectors here, not individual variables
        v.refined_variable = v.refined_variable + p * p_mag
        # simulation
        sim.print_current_var(v, v.refined_variable[j])
        sim.update_variables(v)
        setup, bwc = sim.simulate(v)
        fom = sim.figure_of_merit(v)
        if (fom < best_fit):
            best_fit = fom*1.0
            best_var = np.copy(v.refined_variable[j])
        r3_var[1] = v.refined_variable[j]*1.0
        r3_fom[1] = fom*1.0
        print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
        print(f"-b-----------------------------")  # {r3_var},{r3_fom}")
        var_pl.append(v.refined_variable[j])
        fit_pl.append(fom)

        # Third point
        print("Refining, point 3 of 3")
        if r3_fom[1] > r3_fom[0]:  # if second point is worse
            p_mag = -p_mag
            v.refined_variable += np.exp(0.6)*p*p_mag  # Go in the opposite direction
        else:  # keep going
            v.refined_variable += p*p_mag

        # simulation
        sim.print_current_var(v, v.refined_variable[j])
        sim.update_variables(v)
        setup, bwc = sim.simulate(v)
        fom = sim.figure_of_merit(v)
        if (fom < best_fit):
            best_fit = fom*1.0
            best_var = np.copy(v.refined_variable)
        r3_var[2] = v.refined_variable[j]*1.0
        r3_fom[2] = fom*1.0
        print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
        print(f"-c-----------------------------")  # {r3_var},{r3_fom}")
        var_pl.append(v.refined_variable[j])
        fit_pl.append(fom)

        # we continue until we get a predicted minnymum
        minny = False
        while minny is False:
            last_x = v.refined_variable[j]*1.0
            # predict the next point as a minimum or a step on
            next_x, minny = px.convex(r3_var, r3_fom)
            v.refined_variable += p * (next_x-last_x) / p[j]
            # simulation
            sim.print_current_var(v, v.refined_variable[j])
            sim.update_variables(v)
            setup, bwc = sim.simulate(v)
            fom = sim.figure_of_merit(v)
            if (fom < best_fit):
                best_fit = fom*1.0
                best_var = np.copy(v.refined_variable)
                # replace worst point with this one
                i = np.argmax(r3_fom)
                r3_var[i] = v.refined_variable[j]
                r3_fom[i] = fom*1.0
            else:
                minny = True
            print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
            print(f"-.-----------------------------")  # {r3_var},{r3_fom}")
            var_pl.append(v.refined_variable[j])
            fit_pl.append(fom)

        # End of this cycle
        print("Refinement cycle complete")
        # simulation
        # v.refined_variable += p * (next_x-last_x) / p[j]
        # sim.print_current_var(v, v.refined_variable[j])
        # sim.update_variables(v)
        # setup, bwc = sim.simulate(v)
        # fom = sim.figure_of_merit(v)
        # if (fom < best_fit):
        #     best_fit = fom
        #     best_var = np.copy(v.refined_variable)
        # print(f"  Figure of merit {100*fom:.2f}% (best {100*best_fit:.2f}%)")
        # print(f"-------------------------------")  # {r3_var},{r3_fom}")
        # var_pl.append(v.refined_variable[j])
        # fit_pl.append(fom)

    # Update for next iteration
    last_p = p
    df = last_fit - best_fit
    last_fit = best_fit*1.0
    v.refined_variable = np.copy(best_var)
    v.refinement_scale *= (1 - 1 / (2 * v.n_variables))
    print(f"Improvement in fit {100*df:.2f}%, will stop at {100*v.exit_criteria:.2f}%")
    print("-------------------------------")
    plt.plot(fit_pl)
    # plt.scatter(var_pl, fit_pl)
    plt.show()

print(f"Refinement complete after {v.iter_count} simulations.  Refined values: {best_var}")

# %% final print
sim.print_LACBED(v)
total_time = time.time() - start
print("-----------------------------------------------------------------")
print(f"Beam pool calculation took {setup:.3f} seconds")
print(f"Bloch wave calculation in {bwc:.1f} s ({1000*(bwc)/(4*v.image_radius**2):.2f} ms/pixel)")
print(f"Total time {total_time:.1f} s")
print("-----------------------------------------------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
