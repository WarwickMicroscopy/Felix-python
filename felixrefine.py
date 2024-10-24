# -*- coding: utf-8 -*-
# %% modules and subroutines
"""
Created August 2024

@author: R Beanland

"""
import os
import re
import numpy as np
import cupy as cp
from scipy.constants import c, h, e, m_e, angstrom
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter
import time

start = time.time()
# %% Main felix program
# go to the pylix folder
# path = r"C:\Users\rbean\Documents\GitHub\Felix-python"
# path = r"C:\Users\Richard\Documents\GitHub\Felix-python"
# os.chdir(path)
path = os.getcwd()

# felix modules
# from pylix_modules import pylix_dicts as fu
from pylix_modules import pylix as px
from pylix_modules import pylix_dicts as fu
latest_commit_id = px.get_git()
# outputs
print("-----------------------------------------------------------------")
print(f"felixrefine:  version {latest_commit_id[:8]}")
print("felixrefine: see https://github.com/WarwickMicroscopy/Felix-python")
print("-----------------------------------------------------------------")

# variables to get from felix.icp
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

cell_alpha = px.dlst(cell_angle_alpha)*cp.pi/180.0  # angles in radians
cell_beta = px.dlst(cell_angle_beta)*cp.pi/180.0
cell_gamma = px.dlst(cell_angle_gamma)*cp.pi/180.0
basis_count = len(atom_site_label)
# check things make sense
if "cell_volume" in cif_dict:
    cell_volume = px.dlst(cell_volume)
else:
    cell_volume = cell_a*cell_b*cell_c*cp.sqrt(1.0-cp.cos(cell_alpha)**2
                  - cp.cos(cell_beta)**2 - cp.cos(cell_gamma)**2
                  +2.0*cp.cos(cell_alpha)*cp.cos(cell_beta)*cp.cos(cell_gamma))

# Conversion from scattering factor to volts
scatt_fac_to_volts = ((h**2) /
                      (2.0*cp.pi * m_e * e * cell_volume * (angstrom**2)))

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

# basis_atom_position = cp.zeros([basis_count, 3])
# x_ = cp.array([tup[0] for tup in atom_site_fract_x])
# y_ = cp.array([tup[0] for tup in atom_site_fract_y])
# z_ = cp.array([tup[0] for tup in atom_site_fract_z])
basis_atom_position = cp.column_stack((cp.array([tup[0] for tup in atom_site_fract_x]),
                                       cp.array([tup[0] for tup in atom_site_fract_y]),
                                       cp.array([tup[0] for tup in atom_site_fract_z])))
# Debye-Waller factor
if "atom_site_b_iso_or_equiv" in cif_dict:
    basis_B_iso = cp.array([tup[0] for tup in atom_site_b_iso_or_equiv])
elif "atom_site_u_iso_or_equiv" in cif_dict:
    basis_B_iso = cp.array([tup[0] for tup in
                            atom_site_u_iso_or_equiv])*8*(cp.pi**2)
# occupancy, assume it's unity if not specified
if atom_site_occupancy is not None:
    basis_occupancy = cp.array([tup[0] for tup in atom_site_occupancy])
else:
    basis_occupancy = cp.ones([basis_count])

basis_atom_delta = cp.zeros([basis_count, 3])  # ***********what's this

# %% read felix.inp
inp_dict, error_flag = px.read_inp_file('felix.inp')
for var_name, var_value in inp_dict.items():
    globals()[var_name] = var_value  # Create global variables
if error_flag is True:
    raise ValueError("can't read felix.inp")

# pixels
image_width = 2*image_radius
n_pixels = (image_width)**2

# thickness array
if (final_thickness > initial_thickness + delta_thickness):
    thickness = cp.arange(initial_thickness, final_thickness, delta_thickness)
    n_thickness = len(thickness)
else:
    # need cp.array rather than float so wave_functions works for 1 or many t's
    thickness = cp.array(initial_thickness)
    n_thickness = 1

# convert arrays to numpy
incident_beam_direction = cp.array(incident_beam_direction, dtype='float')
normal_direction = cp.array(normal_direction, dtype='float')
x_direction = cp.array(x_direction, dtype='float')
atomic_sites = cp.array(atomic_sites, dtype='int')

# crystallography exp(2*pi*i*g.r) to physics convention exp(i*g.r)
g_limit = g_limit * 2 * cp.pi

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

# some setup calculations
# Electron velocity in metres per second
electron_velocity = (c * cp.sqrt(1.0 - ((m_e * c**2) /
                     (e * accelerating_voltage_kv*1000.0 + m_e * c**2))**2))
# Electron wavelength in Angstroms
electron_wavelength = h / (
    cp.sqrt(2.0 * m_e * e * accelerating_voltage_kv*1000.0) *
    cp.sqrt(1.0 + (e * accelerating_voltage_kv*1000.0) /
            (2.0 * m_e * c**2))) / angstrom
# Wavevector magnitude k
electron_wave_vector_magnitude = 2.0 * cp.pi / electron_wavelength
# Relativistic correction
relativistic_correction = 1.0 / cp.sqrt(1.0 - (electron_velocity / c)**2)
# Relativistic mass
relativistic_mass = relativistic_correction * m_e

# %% read felix.hkl
input_hkls, i_obs, sigma_obs = px.read_hkl_file("felix.hkl")
n_out = len(input_hkls)+1  # we expect 000 NOT to be in the hkl list


# %% fill the unit cell and get mean inner potential
atom_position, atom_label, atom_name, B_iso, occupancy = \
    px.unique_atom_positions(
        symmetry_matrix, symmetry_vector, basis_atom_label, basis_atom_name,
        basis_atom_position, basis_B_iso, basis_occupancy)

# Generate atomic numbers based on the elemental symbols
atomic_number = cp.array([fu.atomic_number_map[name] for name in atom_name])

n_atoms = len(atom_label)
print("There are "+str(n_atoms)+" atoms in the unit cell")
# plot
if plot:
    atom_cvals = mcolors.Normalize(vmin=1, vmax=103)
    atom_cmap = plt.cm.viridis
    atom_colours = atom_cmap(atom_cvals(atomic_number.get()))  # cp to np
    border_cvals = mcolors.Normalize(vmin=0, vmax=1)
    border_cmap = plt.cm.plasma
    border_colours = border_cmap(border_cvals(atom_position[:, 2].get()))  # cp to np
    bb = 5
    fig, ax = plt.subplots(figsize=(bb, bb))
    plt.scatter(atom_position[:, 0].get(), atom_position[:, 1].get(),  # cp to np
                color=atom_colours, edgecolor=border_colours,
                linewidth=1, s=100)
    plt.xlim(left=0.0, right=1.0)
    plt.ylim(bottom=0.0, top=1.0)
    ax.set_axis_off
    plt.grid(True)
if debug:
    print("Symmetry operations:")
    for i in range(len(symmetry_matrix)):
        print(f"{i+1}: {symmetry_matrix[i]}, {symmetry_vector[i]}")
    print("atomic coordinates")
    for i in range(n_atoms):
        print(f"{atom_label[i]} {atom_name[i]}: {atom_position[i]}")
# mean inner potential as the sum of scattering factors at g=0 multiplied by
# h^2/(2pi*m0*e*CellVolume)
mip = 0.0
for i in range(n_atoms):  # get the scattering factor
    if scatter_factor_method == 0:
        mip += px.f_kirkland(atomic_number[i], 0.0)
    elif scatter_factor_method == 1:
        mip += px.f_lobato(atomic_number[i], 0.0)
    elif scatter_factor_method == 2:
        mip += px.f_peng(atomic_number[i], 0.0)
    elif scatter_factor_method == 3:
        mip += px.f_doyle_turner(atomic_number[i], 0.0)
    else:
        error_flag = True
        raise ValueError("No scattering factors chosen in felix.inp")
mip = mip.item()*scatt_fac_to_volts  # comes back as an array, convert to float
print(f"Mean inner potential = {mip:.1f} Volts")
# Wave vector magnitude in crystal
# high-energy approximation (not HOLZ compatible)
# K^2=k^2+U0
big_k_mag = np.sqrt(electron_wave_vector_magnitude**2+mip)
# k-vector for the incident beam (k is along z in the microscope frame)
big_k = cp.array([0.0, 0.0, big_k_mag.get()], dtype=float)


# %% set up reference frames
a_vec_m, b_vec_m, c_vec_m, ar_vec_m, br_vec_m, cr_vec_m, norm_dir_m = \
    px.reference_frames(cell_a, cell_b, cell_c, cell_alpha, cell_beta,
                        cell_gamma, space_group, x_direction,
                        incident_beam_direction, normal_direction)

# put the crystal in the micrcoscope reference frame, in Å
atom_coordinate = (atom_position[:, 0, cp.newaxis] * a_vec_m +
                   atom_position[:, 1, cp.newaxis] * b_vec_m +
                   atom_position[:, 2, cp.newaxis] * c_vec_m)

# %% set up beam pool
setup = time.time()
# NB g_pool are in reciprocal Angstroms in the microscope reference frame
hkl, g_pool, g_pool_mag, g_output = px.hkl_make(ar_vec_m, br_vec_m, cr_vec_m,
                                                big_k, lattice_type,
                                                min_reflection_pool,
                                                min_strong_beams, g_limit,
                                                input_hkls,
                                                electron_wave_vector_magnitude)
n_hkl = len(g_pool)
n_out = len(g_output)  # redefined to match things we can actually output
# outputs
print(f"Beam pool: {n_hkl} reflexions ({min_strong_beams} strong beams)")
# we will have larger g-vectors in g_matrix since this has differences g - h
# but the maximum of the g pool is probably a more useful thing to know
print(f"Maximum |g| = {cp.max(g_pool_mag)/(2*cp.pi):.3f} 1/Å")

# plot
if plot:
    xm = np.ceil(np.max(g_pool_mag.get()/(2*np.pi)))
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    ax.set_facecolor('black')
    # colour according to Laue zone
    lz_cvals = mcolors.Normalize(vmin=cp.min(g_pool[:, 2]),
                                 vmax=cp.max(g_pool[:, 2]))
    lz_cmap = plt.cm.brg
    lz_colours = lz_cmap(lz_cvals(g_pool[:, 2].get()))
    # plots the g-vectors in the pool, colours for different Laue zones
    plt.scatter(g_pool[:, 0].get()/(2*np.pi), g_pool[:, 1].get()/(2*np.pi),
                s=20, color=lz_colours)
    # title
    plt.annotate("Beam pool", xy=(5, 5), color='white',
                 xycoords='axes pixels', size=24)
    # major grid at 1 1/Å
    plt.grid(True,  which='major', color='lightgrey',
             linestyle='-', linewidth=1.0)
    plt.gca().set_xticks(np.arange(-xm, xm, 1))
    plt.gca().set_yticks(np.arange(-xm, xm, 1))
    plt.grid(True, which='minor', color='grey', linestyle='--',
             linewidth=0.5)
    # minor grid at 0.2 1/Å
    plt.gca().set_xticks(np.arange(-xm, xm, 0.2), minor=True)
    plt.gca().set_yticks(np.arange(-xm, xm, 0.2), minor=True)
    # remove axis labels
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    left=False, right=False,
                    labelbottom=False, labelleft=False)
    plt.show()

# g-vector matrix
g_matrix = cp.zeros((n_hkl, n_hkl, 3))
g_matrix = g_pool[:, cp.newaxis, :] - g_pool[cp.newaxis, :, :]
# g-vector magnitudes
g_magnitude = cp.sqrt(cp.sum(g_matrix**2, axis=2))

# Conversion factor from F_g to U_g
Fg_to_Ug = relativistic_correction / (cp.pi * cell_volume)

# now make the Ug matrix, i.e. calculate the structure factor Fg for all
# g-vectors in g_matrix and convert using the above factor
ug_matrix = Fg_to_Ug * px.Fg_matrix(n_hkl, scatter_factor_method, n_atoms,
                                    atom_coordinate, atomic_number, occupancy,
                                    B_iso, g_matrix, g_magnitude,
                                    absorption_method, absorption_per,
                                    electron_velocity)
# ug_matrix = 10 ug_matrix
# matrix of dot products with the surface normal
g_dot_norm = cp.dot(g_pool, norm_dir_m)

print("Ug matrix constructed")
if debug:
    cp.set_printoptions(precision=3, suppress=True)
    print(100*ug_matrix[:5, :5])


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
# 9 = kV *** NOT YET IMPLEMENTED ***

if 'S' not in refine_mode:
    # read in experimental images that will go in
    lacbed_expt = cp.zeros([image_width, image_width, n_out])
    # get the list of available images
    x_str = str(image_width)
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
                    lacbed_expt[:, :, i] = px.read_dm3(file_path, image_width,
                                                       debug)
                    found = True
            if not found:
                n_expt -= 1
                print(f"{g_string} not found")

        # output experimental LACBED patterns
        w = int(cp.ceil(cp.sqrt(n_out)))
        h = int(cp.ceil(n_out/w))
        fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
        text_effect = withStroke(linewidth=3, foreground='black')
        axes = axes.flatten()
        for i in range(n_out):
            axes[i].imshow(lacbed_expt[:, :, i].get(), cmap='gist_earth')
            axes[i].axis('off')
            annotation = f"{hkl[g_output[i], 0]}{hkl[g_output[i], 1]}{hkl[g_output[i], 2]}"
            axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                             size=30, color='w', path_effects=[text_effect])
        for i in range(n_out, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    independent_variable = ([])
    independent_variable_type = ([])
    atom_refine_flag = ([])
    n_vars = 0
    if 'A' in refine_mode:  # Ug refinement
        print("Refining Structure Factors, A")
        # needs error check for any other refinement
        # raise ValueError("Structure factor refinement incompatible
        # with anything else")
        # we refine magnitude and phase for each Ug.  However for space groups
        # with a centre of symmetry phases are fixed at 0 or pi, so only
        # amplitude is refined (1 independent variable per Ug)
        # Identify the 92 centrosymmetric space groups
        centrosymmetric = [2, 10, 11, 12, 13, 14, 15, 47, 48, 49, 50, 51, 52,
                           53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                           66, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85,
                           86, 87, 88, 123, 124, 125, 126, 127, 128, 129, 130,
                           131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
                           141, 142, 147, 148, 162, 163, 164, 165, 166, 167,
                           175, 176, 191, 192, 193, 194, 200, 201, 202,
                           203, 204, 205, 206, 221, 222, 223, 224, 225, 226,
                           227, 228, 229, 230]
        if space_group_number in centrosymmetric:
            vars_per_ug = 1
        else:
            vars_per_ug = 2

        # set up Ug refinement
        # equivalent g's identified by abs(h)+abs(k)+abs(l)+a*h^2+b*k^2+c*l^2
        g_eqv = (10000*(cp.sum(cp.abs(g_matrix), axis=2) +
                        g_magnitude**2)).astype(int)
        # we keep track of individual Ug's in a matrix ug_eqv
        ug_eqv = cp.zeros([n_hkl, n_hkl], dtype=int)
        # bit of a hack here - can skip Ug's in refinement using ug_offset, but
        # should really be an input in felix.inp if it's going to be used
        ug_offset = 0
        # Ug number (position in ug_matrix)
        i_ug = 1 + ug_offset
        # the first column of the Ug matrix has g-vectors in ascending order
        # we work through this list until we have identified the required no.
        # of Ugs. The matrix ug_eqv identifies equivalent Ug's with an integer
        # whose sign is that of the imaginary part of Ug. (may need to take
        # care of floating point residuals where im(Ug) is nominally zero???)
        j = 1  # number of Ug's processed
        while j < no_of_ugs+1:
            if ug_eqv[i_ug, 0] != 0:  # already in the list, skip it
                i_ug += 1
                continue
            g_id = abs(g_eqv[i_ug, 0])
            # update relevant locations in ug_eqv
            ug_eqv[cp.abs(g_eqv) == g_id] = j*cp.sign(
                cp.imag(ug_matrix[i_ug, 0]))
            # amplitude is type 1, always a variable
            independent_variable.append(ug_matrix[i_ug, 0])
            independent_variable_type.append(0)
            if vars_per_ug == 2:  # we also adjust phase
                independent_variable.append(ug_matrix[i_ug, 0])
                independent_variable_type.append(1)
            j += 1

    else:  # Not a Ug refinement, count refinement variables
        if 'B' in refine_mode:  # Atom coordinate refinement
            for i in range(len(atomic_sites)):
                # the [3, 3] matrix 'moves' returned by atom_move gives the
                # allowed movements for an atom (depending on its Wyckoff
                # symbol and space group) as row vectors with magnitude 1.
                # ***NB NOT ALL SPACE GROUPS IMPLEMENTED ***
                moves = px.atom_move(space_group_number, basis_wyckoff[i])
                degrees_of_freedom = int(cp.sum(moves**2))
                if degrees_of_freedom == 0:
                    raise ValueError("Atom coord refinement not possible")
                for j in range(degrees_of_freedom):
                    r_dot_v = cp.dot(basis_atom_position[atomic_sites[i]],
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

        if 'I' in refine_mode:  # kV
            independent_variable.append(accelerating_voltage_kv)
            independent_variable_type.append(10)
            # Not yet implemented!!!
            raise ValueError("kV refinement not yet implemented")

    # Total number of independent variables
    n_variables = len(independent_variable)
    if n_variables == 0:
        raise ValueError("No refinement variables! \
        Check refine_mode flag in felix.icp. \
            Valid refine modes are A,B,C,D,F,H,S")
    if n_variables == 1:
        print("Only one independent variable")
    else:
        print(f"Number of independent variables = {n_variables}")

    independent_variable = cp.array(independent_variable)
    independent_delta = cp.zeros(n_variables)
    independent_variable_type = cp.array(independent_variable_type)
    independent_variable_atom = cp.array(atom_refine_flag[:n_variables])


# %% deviation parameter for each pixel and g-vector
# s_g [n_hkl, image diameter, image diameter]
# and k vector for each pixel, tilted_k [image diameter, image diameter, 3]
s_g, tilted_k = px.deviation_parameter(convergence_angle, image_radius,
                                       big_k_mag, g_pool, g_pool_mag)


# %% Bloch wave calculation
pool = time.time()
# Dot product of k with surface normal, size [image diameter, image diameter]
k_dot_n = cp.tensordot(tilted_k, norm_dir_m, axes=([2], [0]))

lacbed_sim = cp.zeros([n_thickness, image_width, image_width, len(g_output)],
                      dtype=float)

print("Bloch wave calculation...", end=' ')
if debug:
    print("")
    print("output indices")
    print(g_output[:15])
# pixel by pixel calculations from here
for pix_x in range(image_width):
    # progess
    print(f"\rBloch wave calculation... {50*pix_x/image_radius:.0f}%", end="")

    for pix_y in range(image_width):
        s_g_pix = cp.squeeze(s_g[pix_x, pix_y, :])
        k_dot_n_pix = k_dot_n[pix_x, pix_y]

        # works for multiple thicknesses
        wave_functions = px.wave_functions(
            g_output, s_g_pix, ug_matrix, min_strong_beams, n_hkl, big_k_mag,
            g_dot_norm, k_dot_n_pix, thickness, debug)

        intensity = cp.abs(wave_functions)**2

        # Map diffracted intensity to required output g vectors
        # note x and y swapped!
        lacbed_sim[:, -pix_y, pix_x, :] = intensity[:, :len(g_output)]

print("\rBloch wave calculation... done    ")
done = time.time()


# %% output simulated LACBED patterns
w = int(cp.ceil(cp.sqrt(n_out)))
h = int(cp.ceil(n_out/w))
for j in range(n_thickness):
    fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
    text_effect = withStroke(linewidth=3, foreground='black')
    axes = axes.flatten()
    for i in range(n_out):
        axes[i].imshow(lacbed_sim[j, :, :, i].get(), cmap='pink')
        axes[i].axis('off')
        annotation = f"{hkl[g_output[i], 0]}{hkl[g_output[i], 1]}{hkl[g_output[i], 2]}"
        axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                         size=30, color='w', path_effects=[text_effect])
    for i in range(n_out, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()



# %% figure of merit and best thickness

fom = px.figure_of_merit(lacbed_sim, lacbed_expt, image_processing,
                         blur_radius, correlation_type, plot)
# plot
if plot:
    fig, ax = plt.subplots(1, 1)
    w_f = 10
    fig.set_size_inches(w_f, w_f)
    plt.plot(thickness.get()/10, cp.mean(fom.get(), axis=0))
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
        # dot_product = cp.dot(basis_atom_position[atom_id, :], vector[j - 1, :])
        # basis_atom_position[atom_id, :] = cp.mod(
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
print(f"Beam pool calculation took {pool-setup:.3f} seconds")
print(f"Bloch wave calculation in {done-pool:.1f} s ({1000*(done-pool)/(4*image_radius**2):.2f} ms/pixel)")
print("-----------------------------------------------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
