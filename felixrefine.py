# -*- coding: utf-8 -*-
"""
Started August 2024, converted from MPI Fortran version

@author: R.Beanland@warwick.ac.uk

!  Felix is free software: you can redistribute it and/or modify
!  it under the terms of the GNU General Public License as published by
!  the Free Software Foundation, either version 3 of the License, or
!  (at your option) any later version.
!
!  Felix is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!  GNU General Public License for more details.
!
!  You should have received a copy of the GNU General Public License
!  along with Felix.  If not, see <http://www.gnu.org/licenses/>

"""
# %% modules and subroutines

import os
import numpy as np
import matplotlib.pyplot as plt
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
# a small number
eps = 1e-10


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
print(f"Material: {v.chemical_formula}")

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
# take basis atom labels as given, removing any trailing blanks
v.basis_atom_label = [s.rstrip() for s in v.atom_site_label]
# atom symbols, stripping any charge etc.
v.basis_atom_name = [''.join(filter(str.isalpha, name))
                     for name in v.atom_site_type_symbol]
v.basis_atomic_number = np.array([fu.atomic_number_map[s]
                                  for s in v.basis_atom_name])

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
# redefine the basis to allow coordinate refinement and check for occupancy
v.basis_atom_position = px.preferred_basis(v.space_group_number,
                                           v.basis_atom_position,
                                           v.basis_wyckoff)

# occupancy, assume it's unity if not specified
if v.atom_site_occupancy is not None:
    v.basis_occupancy = np.array([tup[0] for tup in v.atom_site_occupancy])
else:
    v.basis_occupancy = np.ones([n_basis])

# check for multiple occupancy on the same site
tol = 0.000001  # tolerance for saying atoms are the same
diff = v.basis_atom_position[:, None, :] - v.basis_atom_position[None, :, :]
dist2 = np.sum(diff**2, axis=2)
close = (dist2 <= tol**2) & (~np.eye(n_basis, dtype=bool))
# basis_mult_occ has 0 if no shared occupancy, increasing numbers otherwise
v.basis_mult_occ = np.zeros(n_basis, dtype=int)
visited = np.zeros(n_basis, dtype=bool)
group_id = 0
for i in range(n_basis):
    if visited[i]:
        continue
    stack = [i]  # Find all atoms at position i
    cluster = []
    while stack:
        j = stack.pop()
        if visited[j]:
            continue
        visited[j] = True
        cluster.append(j)
        neighbors = np.where(close[j])[0]
        stack.extend(neighbors)
    if len(cluster) > 1:
        group_id += 1
        v.basis_mult_occ[cluster] = group_id

# Thermal displacement parameters, we work with u_ij
# ADP tensor Uij with isotropic components on the diagonal
v.basis_u_ij = np.zeros((n_basis, 3, 3))
idx = np.arange(3)
if "atom_site_b_iso_or_equiv" in cif_dict:
    v.basis_B_iso = np.array([tup[0] for tup in v.atom_site_b_iso_or_equiv])
    v.basis_u_iso = v.basis_B_iso/(8 * np.pi**2)
    v.basis_u_ij[:, idx, idx] = v.basis_u_iso[:, None]
elif "atom_site_u_iso_or_equiv" in cif_dict:
    v.basis_u_iso = np.array([tup[0] for tup in
                              v.atom_site_u_iso_or_equiv])
    v.basis_B_iso = v.basis_u_iso * 8 * np.pi**2  # *** TO BE DELETED ***
    v.basis_u_ij[:, idx, idx] = v.basis_u_iso[:, None]

# check for anisotropic displacement parameters
# and if they exist match them with the correct basis atom
if "atom_site_aniso_label" in cif_dict:
    # remove any trailing blanks
    v.atom_site_aniso_label = [s.rstrip() for s in v.atom_site_aniso_label]
    # link to the basis labels
    for i in range(n_basis):
        for j in range(len(v.atom_site_aniso_label)):
            if v.atom_site_aniso_label[j] == v.basis_atom_label[i]:
                print(f"  Using anisotropic atomic displacement parameters for atom {i}")
                # the data is in 2-tuples (second value is the error)
                v.basis_u_ij[i, 0, 0] = v.atom_site_aniso_u_11[j][0]
                v.basis_u_ij[i, 1, 1] = v.atom_site_aniso_u_22[j][0]
                v.basis_u_ij[i, 2, 2] = v.atom_site_aniso_u_33[j][0]
                v.basis_u_ij[i, 0, 1] = v.atom_site_aniso_u_12[j][0]
                v.basis_u_ij[i, 1, 0] = v.atom_site_aniso_u_12[j][0]
                v.basis_u_ij[i, 0, 2] = v.atom_site_aniso_u_13[j][0]
                v.basis_u_ij[i, 2, 0] = v.atom_site_aniso_u_13[j][0]
                v.basis_u_ij[i, 1, 2] = v.atom_site_aniso_u_23[j][0]
                v.basis_u_ij[i, 2, 1] = v.atom_site_aniso_u_23[j][0]

# np.set_printoptions(precision=5, suppress=True)
# print(f"Anisotropic ADPs: {v.basis_u_ij}")

# oxidation state
if "atom_type_symbol" in cif_dict:
    # remove any trailing blanks
    v.atom_type_symbol = [s.rstrip() for s in v.atom_type_symbol]
    v.basis_oxno = np.zeros(n_basis, dtype=int)
    # is there an oxidation number given in the cif
    if v.atom_type_oxidation_number is not None:
        # link to the basis labels
        for i in range(n_basis):
            for j in range(len(v.atom_type_symbol)):
                if v.atom_type_symbol[j] == v.atom_site_type_symbol[i]:
                    v.basis_oxno[i] = v.atom_type_oxidation_number[j][0]
    # is there a case where oxidation state is just extracted from the end
    # of atom_type_symbol?  If so, it should go here

v.basis_atom_delta = np.zeros([n_basis, 3])  # ***********what's this


# %% read felix.inp
inp_dict = px.read_inp_file('felix.inp')
v.update_from_dict(inp_dict)

if v.debug:
    np.set_printoptions(precision=5, suppress=True)
    for i in range(n_basis):
        print(f"{v.basis_atom_label[i]}:  u_ij =\n {v.basis_u_ij[i, :, :]}")

# thickness array
if (v.final_thickness > v.initial_thickness + v.delta_thickness):
    v.thickness = np.arange(v.initial_thickness, v.final_thickness,
                            v.delta_thickness)
    v.n_thickness = len(v.thickness)
else:
    # need np.array rather than float so wave_functions works for 1 or many t's
    v.thickness = np.atleast_1d(v.initial_thickness)
    v.n_thickness = 1

# convert arrays to numpy
v.incident_beam_direction = np.array(v.incident_beam_direction, dtype='float')
v.normal_direction = np.array(v.normal_direction, dtype='float')
v.x_direction = np.array(v.x_direction, dtype='float')
v.atomic_sites = np.array(v.atomic_sites, dtype='int')

# crystallography exp(2*pi*i*g.r) to physics convention exp(i*g.r)
v.g_limit = v.g_limit * 2 * np.pi

# output
print(f"Zone axis: {v.incident_beam_direction.astype(int)}")
if v.n_thickness == 1:
    print(f"Specimen thickness {v.initial_thickness/10} nm")
else:
    print(f"{v.n_thickness} thicknesses: {', '.join(map(str, v.thickness/10))} nm")

if v.scatter_factor_method == 0:
    print("  Using Kirkland scattering factors")
elif v.scatter_factor_method == 1:
    print("  Using Lobato scattering factors")
elif v.scatter_factor_method == 2:
    print("  Using Peng scattering factors")
elif v.scatter_factor_method == 3:
    print("  Using Doyle & Turner scattering factors")
elif v.scatter_factor_method == 4:
    print("  Using orbital Hartree-Fock scattering factors with Kappa")
    print("    Precomputing atom core and valence densities")
    # initialise pv, pc, kappa and r2
    v.basis_pv = np.zeros(n_basis, dtype=float)
    v.basis_pc = np.zeros(n_basis, dtype=float)
    # initial kappa is 1.0 for a neutral atom
    v.basis_kappa = np.ones(n_basis, dtype=float)
    # number of points in the core/valence calculation
    v.n_points = 1000
    v.r_max = 20  # Angstroms
    v.basis_core = np.zeros([n_basis, v.n_points], dtype=float)
    v.basis_valence = np.zeros([n_basis, v.n_points], dtype=float)
    v.basis_r2 = np.zeros(n_basis, dtype=float)
    for i in range(n_basis):
        orbi = px.orb(v.basis_atomic_number[i])
        v.basis_pv[i] = orbi["pv"]
        v.basis_pc[i] = orbi["pc"]
        v.basis_core[i, :], v.basis_valence[i, :], v.basis_r2[i] = \
            px.precompute_densities(v.basis_atomic_number[i],
                                    v.basis_kappa[i], v.basis_pv[i])
    print(f"    kappa = {v.basis_kappa}")
    print(f"    pv = {v.basis_pv}")
else:
    raise ValueError("No scattering factors chosen in felix.inp")

if v.absorption_method == 0:
    print("  No absorption")
elif v.absorption_method == 1:
    print(f"  Proportional absorption model, set at {v.absorption_per}%")
elif v.absorption_method == 2:
    print("  Bird and King absorption model, with Thomas parameterisation")
else:
    raise ValueError("Invalid absorption method (0,1,2) chosen in felix.inp")

if 'S' in v.refine_mode:
    print("Simulation only, S")
elif 'A' in v.refine_mode:
    print("Refining Structure Factors, A")
    # needs error check for any other refinement
    # raise ValueError("Structure factor refinement
    # incompatible with anything else")
else:  # atom-specific refinements can be done simultaneously
    atm = 0  # flag for atom-specific refinements
    if (len(v.atomic_sites) > n_basis):
        raise ValueError("Number of atomic sites to refine is larger than the \
                         number of atoms")
    if 'B' in v.refine_mode:
        atm = 1
        print("Refining Atomic Coordinates, B")
    if 'C' in v.refine_mode:
        atm = 1
        print("Refining Occupancies, C")
    if 'D' in v.refine_mode:
        atm = 1
        print("Refining Isotropic atomic displacement parameters, D")
    if 'E' in v.refine_mode:
        atm = 1
        print("Refining Anisotropic atomic displacement parameters, E")
    if 'J' in v.refine_mode:
        atm = 1
        print("Refining Kappa, J")
    if 'K' in v.refine_mode:
        atm = 1
        # v.basis_pv[0]= 0.9994
        # v.basis_pv[1] = 4.997
        # v.basis_pv[2]= 5.985
        # v.basis_kappa[0]= 1.3
        # v.basis_kappa[1]= 1.01
        # v.basis_kappa[2]= 1.01
        # kappas (default 1.0)
        # refined kappa : [1.21517673 1.12267508 0.93547286]
        # expand per atom in full unit cell
        # print(unique_aniso_matrixes)
        # print(unique_aniso_matrixes.shape)
        # print(v.basis_kappa)
        # Step 1: define a dictionary of initial P_v guesses per element
        # For LiNbO3 using formal charges as we discussed
        # we just need a dictionary of the valence states of the atoms
        print("Refining valence electrons, K")
    if atm == 1:
        # error check - do specified atom sites make sense
        for i in range(len(v.atomic_sites)):
            if v.atomic_sites[i] >= len(v.basis_atom_name):
                raise ValueError(f"atomic_site {v.atomic_sites[i]} selected for refinement but does not exist")
            print(f"  Refining basis atom {v.atomic_sites[i]}, {v.basis_atom_name[v.atomic_sites[i]]}")

    # non atom-specific refinements
    if 'F' in v.refine_mode:
        print("Refining Lattice Parameters, F")
    if 'G' in v.refine_mode:
        print("Refining Lattice Angles, G")
    if 'H' in v.refine_mode:
        print("Refining Convergence Angle, H")
    if 'I' in v.refine_mode:
        print("Refining Accelerating Voltage, I")
    if 'O' in v.refine_mode:
        print("Beam pool optimisation, O")

    if v.correlation_type == 0:
        print("  Using Pearson correlation")
    elif v.correlation_type == 1:
        print("  Using phase correlation")
    elif v.correlation_type == 2:
        print("  Using Pearson correlation, applying sub-pixel alignment")


# %% read felix.hkl
v.input_hkls, v.i_obs, v.sigma_obs = px.read_hkl_file("felix.hkl")
v.n_out = len(v.input_hkls)+1  # we expect 000 NOT to be in the hkl list


# %% set up refinement
# --------------------------------------------------------------------
# n_variables calculated depending upon Ug and non-Ug refinement
# --------------------------------------------------------------------
# Ug refinement is a special case, cannot do any other refinement alongside
# We count the independent variables:
# v.refined_variable = array of variables to be refined
# v.refined_variable_type = what kind of variable, as follows
# 10 A1 = Ug amplitude *** NOT YET IMPLEMENTED ***
# 11 A2 = Ug phase *** NOT YET IMPLEMENTED ***
# 20 B = atom coordinate *** PARTIALLY IMPLEMENTED *** not all space groups
# 21 C = occupancy
# 22 D = isotropic atomic displacement parameters (ADPs)
# 23,24,25,26,27,28 E = anisotropic atomic displacement parameters (ADPs)
# 30,31,32 F = lattice parameters ***PARTIALLY IMPLEMENTED*** not rhombohedral
# 33,34,35 G = unit cell angles *** NOT YET IMPLEMENTED ***
# 40 H = convergence angle
# 41 I = accelerating_voltage_kv *** NOT YET IMPLEMENTED ***
# 50 J = Kappa
# 51 K = valence electrons
v.refined_variable = ([])  # array of floats, values to be refined
v.refined_variable_type = ([])  # array of integers corresponding to above
v.atom_refine_flag = ([])  # the index of the atom in the .cif, -1 if none
v.atom_refine_vec = ([])  # the direction of atom movement, [0,0,0] if none
nullvec = np.array([0, 0, 0])  # null vector for above
if 'S' not in v.refine_mode:
    v.n_variables = 0
    # count refinement variables
    if 'B' in v.refine_mode:  # Atom coordinate refinement
        # the input v.atomic_sites gives the index of the atom in the cif
        for i in range(len(v.atomic_sites)):
            # the [3, 3] matrix 'moves' returned by atom_move gives the
            # allowed movements for an atom (depending on its Wyckoff
            # symbol and space group) as row vectors with magnitude 1.
            # ***NB NOT ALL SPACE GROUPS IMPLEMENTED ***
            moves = px.atom_move(v.space_group_number,
                                 v.basis_wyckoff[v.atomic_sites[i]])
            degrees_of_freedom = np.sum(np.any(moves, axis=1))
            if degrees_of_freedom == 0:
                raise ValueError(f"Coordinate refinement of atom \
                                 {v.atomic_sites[i]} not possible")
            for j in range(degrees_of_freedom):
                v.atom_coord_vec = moves[j, :]  # the vector of movement
                # we refine the coordinate along the appropriate vector
                r_dot_v = np.dot(v.basis_atom_position[v.atomic_sites[i]],
                                 moves[j, :])
                v.refined_variable.append(r_dot_v)
                v.refined_variable_type.append(20)  # flag to say it's a coord
                v.atom_refine_flag.append(v.atomic_sites[i])  # atom index
                v.atom_refine_vec.append(moves[j, :])  # atom movement

    if 'C' in v.refine_mode:  # Occupancy
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_occupancy[v.atomic_sites[i]])
            v.refined_variable_type.append(21)
            v.atom_refine_flag.append(v.atomic_sites[i])
            v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'D' in v.refine_mode:  # Isotropic ADPs
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_B_iso[v.atomic_sites[i]])
            v.refined_variable_type.append(22)
            v.atom_refine_flag.append(v.atomic_sites[i])
            v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'E' in v.refine_mode:  # Anisotropic ADPs, only refine non-zero
        for i in range(len(v.atomic_sites)):
            U = v.basis_u_ij[v.atomic_sites[i]]
            # Extract symmetric independent components
            aniso_params = np.array([U[0, 0], U[1, 1], U[2, 2],
                                     U[0, 1], U[0, 2], U[1, 2]])
            anisotypes = [23, 24, 25, 26, 27, 28]
            for param, t in zip(aniso_params, anisotypes):
                if abs(param) > eps:
                    v.refined_variable.append(param)
                    v.refined_variable_type.append(t)
                    v.atom_refine_flag.append(v.atomic_sites[i])
                    v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'F' in v.refine_mode:  # Lattice parameters
        # variable_type first digit=6 indicates lattice parameter
        # second digit=1,2,3 indicates a,b,c
        # This section needs work to include rhombohedral cells and
        # non-standard settings!!!
        v.refined_variable.append(v.cell_a)  # is in all lattice types
        v.refined_variable_type.append(30)
        v.atom_refine_flag.append(-1)  # -1 indicates not an atom
        v.atom_refine_vec.append(nullvec)  # no atom movement
        if v.space_group_number < 75:  # Triclinic, monoclinic, orthorhombic
            v.refined_variable.append(v.cell_b)
            v.refined_variable_type.append(31)
            v.atom_refine_flag.append(-1)
            v.refined_variable.append(v.cell_c)
            v.refined_variable_type.append(32)
            v.atom_refine_flag.append(-1)
            v.atom_refine_vec.append(nullvec)  # no atom movement
        elif 142 < v.space_group_number < 160:  # Rhombohedral 168
            # Need to work out R- vs H- settings!!!
            raise ValueError("Rhombohedral R- vs H- not yet implemented")
        elif (160 < v.space_group_number < 195) or \
             (74 < v.space_group_number < 143):  # Hexagonal or Tetragonal 167
            v.refined_variable.append(v.cell_c)
            v.refined_variable_type.append(32)
            v.atom_refine_flag.append(-1)
            v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'G' in v.refine_mode:  # Unit cell angles
        # Not yet implemented!!! variable_type 33,34,35
        raise ValueError("Unit cell angle refinement not yet implemented")

    if 'H' in v.refine_mode:  # Convergence angle
        v.refined_variable.append(v.convergence_angle)
        v.refined_variable_type.append(40)
        v.atom_refine_flag.append(-1)
        v.atom_refine_vec.append(nullvec)  # no atom movement
        print(f"Starting convergence angle {v.convergence_angle} Å^-1")

    if 'I' in v.refine_mode:  # accelerating_voltage_kv
        v.refined_variable.append(v.accelerating_voltage_kv)
        v.refined_variable_type.append(41)
        v.atom_refine_flag.append(-1)
        v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'J' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_kappa[v.atomic_sites[i]])
            v.refined_variable_type.append(50)
            v.atom_refine_flag.append(v.atomic_sites[i])
            v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'K' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_pv[v.atomic_sites[i]])
            v.refined_variable_type.append(51)
            v.atom_refine_flag.append(v.atomic_sites[i])
            v.atom_refine_vec.append(nullvec)  # no atom movement

    # Total number of independent variables
    v.n_variables = len(v.refined_variable)
    if v.n_variables == 0 and v.refine_mode != 'O':
        raise ValueError("No refinement variables! \
        Check refine_mode flag in felix.inp. \
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


# %% baseline simulation or beam pool optimisation
print("-------------------------------")
if 'O' in v.refine_mode:
    diff_max, diff_mean, times = sim.optimise_pool(v)
else:
    print("Baseline simulation:")
    # uses the whole v=Var class
    sim.simulate(v)
    # print_LACBED has options 0=sim, 1=expt, 2=difference
    sim.print_LACBED(v, 0)

# %% read in experimental images
if 'S' not in v.refine_mode:
    v.lacbed_expt_raw = np.zeros([2*v.image_radius, 2*v.image_radius, v.n_out])
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
                    v.lacbed_expt_raw[:, :, i] = px.read_dm3(file_path,
                                                             2*v.image_radius,
                                                             v.debug)
                    found = True
            if not found:
                n_expt -= 1  # *_* we don't actually do anything with this!
                print(f"{g_string} not found")
                
        # print experimental LACBED patterns
        v.lacbed_expt = np.copy(v.lacbed_expt_raw)
        # print_LACBED has options 0=sim, 1=expt, 2=difference
        sim.print_LACBED(v, 1)
        # initialise correlation
        best_corr = np.ones(v.n_out)
    

# output LACBED patterns and figure of merit
if v.image_processing == 1:
    print(f"  Blur radius {v.blur_radius} pixels")
if 'S' not in v.refine_mode:
    # figure of merit
    fom = sim.figure_of_merit(v)
    print(f"  Figure of merit {100*fom:.2f}%")
    print("-------------------------------")


# %% start refinement loop *** needs work
if 'S' not in v.refine_mode:
    # Initialise variables for refinement
    fit0 = fom*1.0
    v.best_fit = fom*1.0
    last_fit = fom*1.0
    r3_var = np.zeros(3)  # for parabolic minimum
    r3_fom = np.zeros(3)
    # *_*dunno what this is
    independent_delta = 0.0

    # for a plot
    v.fit_log = ([last_fit])

    # Refinement loop
    df = 1.0
    while df >= v.exit_criteria:
        # v.refined_variable is the working array of variables
        # best_var is the best array of variables during this refinement cycle
        v.best_var = np.copy(v.refined_variable)
        # next_var is the predicted next (best) point
        v.next_var = np.copy(v.refined_variable)

        if v.refine_method == 0:
            print("Gradient descent, one parameter at a time")
            # dydx is a vector along the gradient in n-dimensional space
            dydx = np.zeros(v.n_variables)
            for i in range(v.n_variables):
                dydx[i] = 1.0
                print(f"Refinement vector {dydx}")
                # single is just multiparameter with one non-zero value
                # v.next_var = v.best_var - dydx*v.refinement_scale
                dydx = sim.refine_multi_variable(v, dydx)

        elif v.refine_method == 1:
            print("Multiparameter refinement, finding parameter gradients")
            # =========== step 1: individual variable minimisation
            # if all variables have been refined, reset
            if np.sum(np.abs(dydx)) < 1e-10:
                dydx = np.ones(v.n_variables)
            # Go through the variables looking at three points in the hope
            # of capturing a minimum - if there is one, we take it and remove
            # that variable from multidimensional refinement, dydx[i] = 0.
            # Otherwise dydx[i] is the gradient for that variable.
            # We also get a predicted best starting point
            # for gradient descent, v.next_var
            for i in range(v.n_variables):
                # Skip variables already optimized
                if abs(dydx[i]) < 1e-10:
                    dydx[i] = 0.0
                    continue
                dydx[i] = sim.refine_single_variable(v, i)

            # all variables have updated/predicted so do a final simulation
            # if it's better, update v.best_fit and v.best_var accordingly
            if np.count_nonzero(dydx) == 0:
                print("Closing simulation for this cycle")
                v.refined_variable = np.copy(v.next_var)
                fom = sim.sim_fom(v, 0)
                if (fom < v.best_fit):
                    v.best_fit = fom*1.0
                    v.best_var = np.copy(v.refined_variable)
            print("Vector gradient descent")
            # ===========step 2: vector descent
            # Downhill minimisation until we eliminate all variables
            while np.sum(np.abs(dydx)) > 1e-10:
                # the returned dydx will have an extra zero!
                dydx = sim.refine_multi_variable(v, dydx, False)
        else:
            raise ValueError("No valid refine method (0,1) in felix.inp")

        # Update for next iteration
        df = last_fit - v.best_fit

        last_fit = np.copy(v.best_fit)
        v.refined_variable = np.copy(v.best_var)
        # reduce refinement scale for next round
        v.refinement_scale *= (1 - 1 / (1 + v.n_variables))
        print(f"Improvement in fit {100*df:.2f}%, will stop at {100*v.exit_criteria:.2f}%")
        if df >= v.exit_criteria:
            print(f"Step size reduced to {v.refinement_scale:.6f}")
        print("-------------------------------")
        if v.plot >= 1:
            plt.plot(v.fit_log)
            # plt.scatter(var_pl, fit_pl)
            plt.show()

    print(f"Refinement complete after {v.iter_count} simulations.  Refined values: {v.best_var}")

# %% final print
sim.print_LACBED(v, 0)
sim.save_LACBED(v)
total_time = time.time() - start
print("-----------------------------------------------------------------")
print(f"Total time {total_time:.1f} s")
print("-----------------------------------------------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
