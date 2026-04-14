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
import time
import numpy as np

# felix modules
from pylix_modules import pylix as px  # most calculations found here
from pylix_modules import simulate as sim  # simulation control and output
from pylix_modules import pylix_dicts as fu  # dictionaries
from pylix_modules import pylix_class as pc  # classes

path = os.getcwd()
start = time.time()
latest_commit_id = px.get_git()
# outputs
print("-----------------------------------------------------------------")
print(f"felixrefine:  version {latest_commit_id[:8]}")
print("felixrefine:  https://github.com/WarwickMicroscopy/Felix-python")
print("-----------------------------------------------------------------")

# initialise class objects
rc = pc.RunControl()  # felix.inp and derived variables for run control
cif = pc.Cif()  # cif variables
hkl = pc.Hkl()  # hkl file variables
xtal = pc.Crystal()  # crystal variables extracted from cif
basis = pc.Basis()  # basis variables extracted from cif
cell = pc.Cell()  # filled cell variables, from px.unique_atom_positions
bloch = pc.Bloch()  # variables in Bloch wave calculation
cbed = pc.Cbed()  # images

# initialise iteration count
rc.iter_count = 0
# a small number
eps = 1e-10


# %% read felix.cif
# cif_dict is a dictionary of value-key pairs.  values are given as tuples
# with the second number the uncertainty in the first.  Nothing is currently
# done with these uncertainties...
cif_dict = px.read_cif('felix.cif')
cif.update_from_dict(cif_dict)
# ====== extract cif data into working variables
xtal.space_group = cif.symmetry_space_group_name_h_m
if cif.chemical_formula_structural is not None:
    xtal.chemical_formula = cif.chemical_formula_structural
elif cif.chemical_formula_sum is not None:
    xtal.chemical_formula = cif.chemical_formula_sum
elif cif.chemical_formula_iupac is not None:
    xtal.chemical_formula = cif.chemical_formula_iupac
print(f"Material: {xtal.chemical_formula}")

# space group number and lattice type
if "space_group_symbol" in cif_dict:
    xtal.space_group = cif.space_group_symbol.replace(' ', '')
elif "space_group_name_h_m_alt" in cif_dict:
    xtal.space_group = cif.space_group_name_h_m_alt.replace(' ', '')
elif "symmetry_space_group_name_h_m" in cif_dict:
    xtal.space_group = cif.symmetry_space_group_name_h_m.replace(' ', '')
elif "space_group_it_number" in cif_dict:
    xtal.space_group_number = int(cif.space_group_it_number[0])
    reverse_space_groups = {rc: k for k, rc in fu.space_groups.items()}
    xtal.space_group = reverse_space_groups.get(xtal.space_group_number, "Unknown")
else:
    error_flag = True
    raise ValueError("No space group found in .cif")
xtal.lattice_type = xtal.space_group[0]
xtal.space_group_number = fu.space_groups[xtal.space_group]

# cell
xtal.cell_a = cif.cell_length_a[0]
xtal.cell_b = cif.cell_length_b[0]
xtal.cell_c = cif.cell_length_c[0]
xtal.cell_alpha = cif.cell_angle_alpha[0]*np.pi/180.0  # angles in radians
xtal.cell_beta = cif.cell_angle_beta[0]*np.pi/180.0
xtal.cell_gamma = cif.cell_angle_gamma[0]*np.pi/180.0
basis.n_atoms = len(cif.atom_site_label)

# symmetry operations
if "space_group_symop_operation_xyz" in cif_dict:
    xtal.symmetry_matrix, xtal.symmetry_vector = px.symop_convert(
        cif.space_group_symop_operation_xyz)
elif "symmetry_equiv_pos_as_xyz" in cif_dict:
    xtal.symmetry_matrix, xtal.symmetry_vector = px.symop_convert(
        cif.symmetry_equiv_pos_as_xyz)
else:
    error_flag = True
    raise ValueError("Symmetry operations not found in .cif")

# extract the basis from the raw cif values
# take basis atom labels as given, removing any trailing blanks
basis.atom_label = [s.rstrip() for s in cif.atom_site_label]
# atom symbols, stripping any charge etc.
basis.atom_name = [''.join(filter(str.isalpha, name))
                   for name in cif.atom_site_type_symbol]
basis.atomic_number = np.array([fu.atomic_number_map[s]
                                for s in basis.atom_name])

# take care of any odd symbols, get the case right
for i in range(basis.n_atoms):
    name = basis.atom_name[i]
    if len(name) == 1:
        name = name.upper()
    elif len(name) > 1:
        name = name[0].upper() + name[1:].lower()
    basis.atom_name[i] = name
# take basis Wyckoff letters as given (maybe check they are only letters?)
basis.wyckoff = cif.atom_site_wyckoff_symbol

basis.atom_position = \
    np.column_stack((np.array([tup[0] for tup in cif.atom_site_fract_x]),
                     np.array([tup[0] for tup in cif.atom_site_fract_y]),
                     np.array([tup[0] for tup in cif.atom_site_fract_z])))
# redefine the basis to allow coordinate refinement and check for occupancy
px.preferred_basis(basis, xtal.space_group_number)

# occupancy, assume it's unity if not specified
if cif.atom_site_occupancy is not None:
    basis.occupancy = np.array([tup[0] for tup in cif.atom_site_occupancy])
else:
    basis.occupancy = np.ones([basis.n_atoms])

# check for multiple occupancy on the same site
tol = 0.0001  # tolerance for saying atoms are the same
diff = basis.atom_position[:, None, :] - basis.atom_position[None, :, :]
dist2 = np.sum(diff**2, axis=2)
close = (dist2 <= tol**2) & (~np.eye(basis.n_atoms, dtype=bool))
# mult_occ has 0 if no shared occupancy, increasing numbers otherwise
basis.mult_occ = np.zeros(basis.n_atoms, dtype=int)
visited = np.zeros(basis.n_atoms, dtype=bool)
group_id = 0
for i in range(basis.n_atoms):
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
        basis.mult_occ[cluster] = group_id

# Thermal displacement parameters, we work with u_ij
# ADP tensor Uij with isotropic components on the diagonal
basis.u_aniso = np.zeros((basis.n_atoms, 3, 3))
idx = np.arange(3)
if "atom_site_b_iso_or_equiv" in cif_dict:
    basis.B_iso = np.array([tup[0] for tup in cif.atom_site_b_iso_or_equiv])
    basis.u_iso = basis.B_iso/(8 * np.pi**2)
    basis.u_aniso[:, idx, idx] = basis.u_iso[:, None]
elif "atom_site_u_iso_or_equiv" in cif_dict:
    basis.u_iso = np.array([tup[0] for tup in cif.atom_site_u_iso_or_equiv])
    basis.B_iso = basis.u_iso * 8 * np.pi**2  # *** TO BE DELETED? ***
    basis.u_aniso[:, idx, idx] = basis.u_iso[:, None]

# check for anisotropic displacement parameters
# and if they exist match them with the correct basis atom
if "atom_site_aniso_label" in cif_dict:
    # remove any trailing blanks
    cif.atom_site_aniso_label = [s.rstrip() for s in cif.atom_site_aniso_label]
    # link to the basis labels
    for i in range(basis.n_atoms):
        for j in range(len(cif.atom_site_aniso_label)):
            if cif.atom_site_aniso_label[j] == basis.atom_label[i]:
                print(f"  Using anisotropic atomic displacement parameters for atom {i}")
                # the data is in 2-tuples (second value is the error)
                basis.u_aniso[i, 0, 0] = cif.atom_site_aniso_u_11[j][0]
                basis.u_aniso[i, 1, 1] = cif.atom_site_aniso_u_22[j][0]
                basis.u_aniso[i, 2, 2] = cif.atom_site_aniso_u_33[j][0]
                basis.u_aniso[i, 0, 1] = cif.atom_site_aniso_u_12[j][0]
                basis.u_aniso[i, 1, 0] = cif.atom_site_aniso_u_12[j][0]
                basis.u_aniso[i, 0, 2] = cif.atom_site_aniso_u_13[j][0]
                basis.u_aniso[i, 2, 0] = cif.atom_site_aniso_u_13[j][0]
                basis.u_aniso[i, 1, 2] = cif.atom_site_aniso_u_23[j][0]
                basis.u_aniso[i, 2, 1] = cif.atom_site_aniso_u_23[j][0]

# np.set_printoptions(precision=5, suppress=True)
# print(f"Anisotropic ADPs: {basis.u_aniso}")

# oxidation state
if "atom_type_symbol" in cif_dict:
    # remove any trailing blanks
    cif.atom_type_symbol = [s.rstrip() for s in cif.atom_type_symbol]
    basis.oxno = np.zeros(basis.n_atoms, dtype=int)
    # is there an oxidation number given in the cif
    if cif.atom_type_oxidation_number is not None:
        # link to the basis labels
        for i in range(basis.n_atoms):
            for j in range(len(cif.atom_type_symbol)):
                if cif.atom_type_symbol[j] == cif.atom_site_type_symbol[i]:
                    basis.oxno[i] = cif.atom_type_oxidation_number[j][0]
    # is there a case where oxidation state is just extracted from the end
    # of atom_type_symbol?  If so, it should go here

basis.atom_delta = np.zeros([basis.n_atoms, 3])  # direction of movement


# %% read felix.inp
inp_dict = px.read_inp_file('felix.inp')
rc.update_from_dict(inp_dict)

if rc.debug:
    np.set_printoptions(precision=5, suppress=True)
    for i in range(basis.n_atoms):
        print(f"{basis.atom_label[i]}:  u_ij =\n {basis.u_aniso[i, :, :]}")

# thickness array
if rc.final_thickness > rc.initial_thickness + rc.delta_thickness:
    rc.thickness = np.arange(rc.initial_thickness, rc.final_thickness,
                             rc.delta_thickness)
    rc.n_thickness = len(rc.thickness)
else:
    # need np.array rather than float so wave_functions works for 1 or many t's
    rc.thickness = np.atleast_1d(rc.initial_thickness)
    rc.n_thickness = 1
# give best thickness a value of 0 for the case of only one t 
rc.best_t = 0

# convert arrays to numpy
rc.incident_beam_direction = np.array(rc.incident_beam_direction,
                                      dtype='float')
rc.normal = np.array(rc.normal, dtype='float')
rc.x_direction = np.array(rc.x_direction, dtype='float')
rc.atomic_sites = np.array(rc.atomic_sites, dtype='int')

# crystallography exp(2*pi*i*g.r) to physics convention exp(i*g.r)
rc.g_limit = rc.g_limit * 2 * np.pi

# output
print(f"Zone axis: {rc.incident_beam_direction.astype(int)}")
if rc.n_thickness == 1:
    print(f"Specimen thickness {rc.initial_thickness/10} nm")
else:
    print(f"{rc.n_thickness} thicknesses: {', '.join(map(str, rc.thickness/10))} nm")

if rc.scatter_factor_method == 0:
    print("  Using Kirkland scattering factors")
elif rc.scatter_factor_method == 1:
    print("  Using Lobato scattering factors")
elif rc.scatter_factor_method == 2:
    print("  Using Peng scattering factors")
elif rc.scatter_factor_method == 3:
    print("  Using Doyle & Turner scattering factors")
elif rc.scatter_factor_method > 3:
    if rc.scatter_factor_method == 4:
        print("  Using Coppens RHF scattering factors with Kappa")
    else:
        print("  Using Bunge RHF scattering factors with Kappa")
    # initialise pv, pc, kappa and electron density
    basis.pv = np.zeros(basis.n_atoms, dtype=float)
    basis.pc = np.zeros(basis.n_atoms, dtype=float)
    basis.n_electrons = np.zeros(basis.n_atoms, dtype=float)
    # initial kappa is 1.0 for a neutral atom
    basis.kappa = np.ones(basis.n_atoms, dtype=float)
    # initial calculation of orbitals
    # px.electron_density(xtal, basis, rc)
else:
    raise ValueError("No scattering factors chosen in felix.inp")

if rc.absorption_method == 0:
    print("  No absorption")
elif rc.absorption_method == 1:
    print(f"  Proportional absorption model, set at {rc.absorption_per}%")
elif rc.absorption_method == 2:
    print("  Bird and King absorption model, with Thomas parameterisation")
else:
    raise ValueError("Invalid absorption method !(0,1,2) chosen in felix.inp")

if 'S' in rc.refine_mode:
    print("Simulation only, S")
elif 'A' in rc.refine_mode:
    print("Refining Structure Factors, A")
    # needs error check for any other refinement
    # raise ValueError("Structure factor refinement
    # incompatible with anything else")
else:  # atom-specific refinements can be done simultaneously
    atm = 0  # flag for atom-specific refinements
    if 'B' in rc.refine_mode:
        atm = 1
        print("Refining Atomic Coordinates, B")
    if 'C' in rc.refine_mode:
        atm = 1
        print("Refining Occupancies, C")
    if 'D' in rc.refine_mode:
        atm = 1
        print("Refining Isotropic atomic displacement parameters, D")
    if 'E' in rc.refine_mode:
        atm = 1
        print("Refining Anisotropic atomic displacement parameters, E")
    if 'J' in rc.refine_mode:
        atm = 1
        print("Refining Kappa, J")
    if 'K' in rc.refine_mode:
        atm = 1
        print("Refining valence electron population, K")
    if 'J' in rc.refine_mode or 'K' in rc.refine_mode:
        # check we're using RHF scattering factors
        if rc.scatter_factor_method < 4:
            raise ValueError("scatter_factor_method must be 4 or 5 for kappa/Pv refinement")
        else:
            px.electron_density(xtal, basis, rc)

    if atm == 1:
        # error check - do specified atom sites make sense
        if len(rc.atomic_sites) > basis.n_atoms:
            raise ValueError("Number of atomic sites to refine is larger \
                             than the number of atoms")
        for i in range(len(rc.atomic_sites)):
            if rc.atomic_sites[i] >= basis.n_atoms:
                raise ValueError(f"atomic_site {rc.atomic_sites[i]} selected for refinement but does not exist")
            print(f"  Refining basis atom {rc.atomic_sites[i]}, {basis.atom_name[rc.atomic_sites[i]]}")

    # non atom-specific refinements
    if 'F' in rc.refine_mode:
        print("Refining Lattice Parameters, F")
    if 'G' in rc.refine_mode:
        print("Refining Lattice Angles, G")
    if 'H' in rc.refine_mode:
        print("Refining Convergence Angle, H")
    if 'I' in rc.refine_mode:
        print("Refining Accelerating Voltage, I")
    if 'O' in rc.refine_mode:
        print("Beam pool optimisation, O")

    if rc.correlation_type == 0:
        print("  Using phase correlation")
    elif rc.correlation_type == 1:
        print("  Using Pearson correlation")
    elif rc.correlation_type == 2:
        print("  Using Pearson correlation with affine transform")
    elif rc.correlation_type == 3:
        print("  Using Pearson correlation with affine transform and sub-pixel alignment")
    elif rc.correlation_type == 4:
        print("  Using Pearson correlation with Sobel filter")
    else:
        raise ValueError("Correlation type invalid in felix.inp")


# %% read felix.hkl
px.read_hkl_file(hkl, "felix.hkl")
rc.n_out = len(hkl.input_hkls)+1  # we expect 000 NOT to be in the hkl list


# %% set up refinement
# --------------------------------------------------------------------
# n_variables calculated depending upon Ug and non-Ug refinement
# --------------------------------------------------------------------
# Ug refinement is a special case, cannot do any other refinement alongside
# We count the independent variables:
# rc.refined_variable = array of variables to be refined
# rc.refined_variable_type = what kind of variable, as follows
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
rc.n_variables = 0
rc.refined_variable = []  # array of floats, values to be refined
rc.refined_variable_type = []  # array of integers corresponding to above
rc.atom_refine_flag = []  # the index of the atom in the .cif, -1 if none
rc.atom_refine_vec = []  # the direction of atom movement, [0,0,0] if none
nullvec = np.array([0, 0, 0])  # null vector for above
if 'S' not in rc.refine_mode:
    n_sites = len(rc.atomic_sites)
    # count refinement variables
    if 'B' in rc.refine_mode:  # Atom coordinate refinement
        # the input rc.atomic_sites gives the index of the atom in the cif
        for i in range(n_sites):
            # the [3, 3] matrix 'rc.moves' returned by atom_move gives the
            # allowed movements for an atom (depending on its Wyckoff
            # symbol and space group) as row vectors with magnitude 1.
            # ***NB NOT ALL SPACE GROUPS IMPLEMENTED ***
            rc.moves = px.atom_move(xtal.space_group_number,
                                      basis.wyckoff[rc.atomic_sites[i]])
            degrees_of_freedom = np.sum(np.any(rc.moves, axis=1))
            if degrees_of_freedom == 0:
                raise ValueError(f"Coordinate refinement of atom \
                                 {rc.atomic_sites[i]} not possible")
            for j in range(degrees_of_freedom):
                rc.atom_coord_vec = rc.moves[j, :]  # the vector of movement
                # we refine the coordinate along the appropriate vector
                r_dot_v = np.dot(basis.atom_position[rc.atomic_sites[i]],
                                 rc.moves[j, :])
                rc.refined_variable.append(r_dot_v)
                rc.refined_variable_type.append(20)  # flag to say it's a coord
                rc.atom_refine_flag.append(rc.atomic_sites[i])  # atom index
                rc.atom_refine_vec.append(rc.moves[j, :])  # atom movement

    if 'C' in rc.refine_mode:  # Occupancy
        refined_sites = set()
        for i in range(n_sites):
            site = basis.mult_occ[rc.atomic_sites[i]]
            # check if we already have this site
            if site in refined_sites:
                print(f"  Shared site: not refining occupancy of atom {rc.atomic_sites[i]}")
                continue
            refined_sites.add(site)
            rc.refined_variable.append(basis.occupancy[rc.atomic_sites[i]])
            rc.refined_variable_type.append(21)
            rc.atom_refine_flag.append(rc.atomic_sites[i])
            rc.atom_refine_vec.append(nullvec)  # no atom movement

    if 'D' in rc.refine_mode:  # Isotropic ADPs
        for i in range(n_sites):
            rc.refined_variable.append(basis.B_iso[rc.atomic_sites[i]])
            rc.refined_variable_type.append(22)
            rc.atom_refine_flag.append(rc.atomic_sites[i])
            rc.atom_refine_vec.append(nullvec)  # no atom movement

    if 'E' in rc.refine_mode:  # Anisotropic ADPs, only refine non-zero
        for i in range(n_sites):
            U = basis.u_ij[rc.atomic_sites[i]]
            # Extract symmetric independent components
            aniso_params = np.array([U[0, 0], U[1, 1], U[2, 2],
                                     U[0, 1], U[0, 2], U[1, 2]])
            anisotypes = [23, 24, 25, 26, 27, 28]
            for param, t in zip(aniso_params, anisotypes):
                if abs(param) > eps:
                    rc.refined_variable.append(param)
                    rc.refined_variable_type.append(t)
                    rc.atom_refine_flag.append(rc.atomic_sites[i])
                    rc.atom_refine_vec.append(nullvec)  # no atom movement

    if 'F' in rc.refine_mode:  # Lattice parameters
        # variable_type first digit=6 indicates lattice parameter
        # second digit=1,2,3 indicates a,b,c
        # This section needs work to include rhombohedral cells and
        # non-standard settings!!!
        rc.refined_variable.append(xtal.cell_a)  # is in all lattice types
        rc.refined_variable_type.append(30)
        rc.atom_refine_flag.append(-1)  # -1 indicates not an atom
        rc.atom_refine_vec.append(nullvec)  # no atom movement
        if xtal.space_group_number < 75:  # Triclinic, monoclinic, orthorhombic
            rc.refined_variable.append(xtal.cell_b)
            rc.refined_variable_type.append(31)
            rc.atom_refine_flag.append(-1)
            rc.refined_variable.append(xtal.cell_c)
            rc.refined_variable_type.append(32)
            rc.atom_refine_flag.append(-1)
            rc.atom_refine_vec.append(nullvec)  # no atom movement
        elif 142 < xtal.space_group_number < 160:  # Rhombohedral 168
            # Need to work out R- vs H- settings!!!
            raise ValueError("Rhombohedral R- vs H- not yet implemented")
        elif (160 < xtal.space_group_number < 195) or \
             (74 < xtal.space_group_number < 143):  # Hexagonal or Tetragonal 167
            rc.refined_variable.append(xtal.cell_c)
            rc.refined_variable_type.append(32)
            rc.atom_refine_flag.append(-1)
            rc.atom_refine_vec.append(nullvec)  # no atom movement

    if 'G' in rc.refine_mode:  # Unit cell angles
        # Not yet implemented!!! variable_type 33,34,35
        raise ValueError("Unit cell angle refinement not yet implemented")

    if 'H' in rc.refine_mode:  # Convergence angle
        rc.refined_variable.append(rc.convergence_angle)
        rc.refined_variable_type.append(40)
        rc.atom_refine_flag.append(-1)
        rc.atom_refine_vec.append(nullvec)  # no atom movement
        print(f"Starting convergence angle {rc.convergence_angle} Å^-1")

    if 'I' in rc.refine_mode:  # accelerating_voltage_kv
        rc.refined_variable.append(rc.accelerating_voltage_kv)
        rc.refined_variable_type.append(41)
        rc.atom_refine_flag.append(-1)
        rc.atom_refine_vec.append(nullvec)  # no atom movement

    if 'J' in rc.refine_mode:
        for i in range(n_sites):
            rc.refined_variable.append(basis.kappa[rc.atomic_sites[i]])
            rc.refined_variable_type.append(50)
            rc.atom_refine_flag.append(rc.atomic_sites[i])
            rc.atom_refine_vec.append(nullvec)  # no atom movement

    if 'K' in rc.refine_mode:
        for i in range(n_sites):
            rc.refined_variable.append(basis.pv[rc.atomic_sites[i]])
            rc.refined_variable_type.append(51)
            rc.atom_refine_flag.append(rc.atomic_sites[i])
            rc.atom_refine_vec.append(nullvec)  # no atom movement

    # Total number of independent variables
    rc.n_variables = len(rc.refined_variable)
    if rc.n_variables == 0 and rc.refine_mode != 'O':
        raise ValueError("No refinement variables! \
        Check refine_mode flag in felix.inp. \
            Valid refine modes are A,B,C,D,F,H,S")
    if rc.n_variables == 1:
        print("Only one independent variable")
    else:
        print(f"Number of independent variables = {rc.n_variables}")

    rc.refined_variable = np.array(rc.refined_variable)
    independent_delta = np.zeros(rc.n_variables)
    rc.refined_variable_type = np.array(rc.refined_variable_type)
    rc.refined_variable_atom = np.array(rc.atom_refine_flag[:rc.n_variables])
else:
    # we still need a type for later code, set it to zero for sim only
    rc.refined_variable_type = np.array([0])
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


# %% baseline simulation & beam pool optimisation if required
print("-------------------------------")
if 'O' in rc.refine_mode:
    # diff_max, diff_mean, times = sim.optimise_pool(v)
    sim.optimise_pool(xtal, basis, cell, hkl, bloch, cbed, rc)
else:
    print("Baseline simulation:")
    sim.simulate(xtal, basis, cell, hkl, bloch, cbed, rc)
    # print_LACBED has options 0=sim, 1=expt, 2=difference
    sim.print_LACBED(bloch, cbed, rc, 0)


# %% read in experimental images
if 'S' not in rc.refine_mode:
    cbed.lacbed_expt_raw = np.zeros([2*rc.image_radius, 2*rc.image_radius,
                                     rc.n_out])
    # get the list of available images
    x_str = str(2*rc.image_radius)
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
        n_expt = rc.n_out
        for i in range(rc.n_out):
            g_string = px.hkl_string(bloch.hkl_indices[bloch.hkl_output[i]])
            found = False
            for file_name in dm3_files:
                if g_string in file_name:
                    file_path = os.path.join(dm3_folder, file_name)
                    cbed.lacbed_expt_raw[:, :, i] = px.read_dm3(file_path,
                                                             2*rc.image_radius,
                                                             rc.debug)
                    found = True
            if not found:
                n_expt -= 1  # *_* we don't actually do anything with this!
                print(f"{g_string} not found")

        # print experimental LACBED patterns
        cbed.lacbed_expt = np.copy(cbed.lacbed_expt_raw)
        # print_LACBED has options 0=sim, 1=expt, 2=difference
        sim.print_LACBED(bloch, cbed, rc, 1)
        # initialise correlation
        best_corr = np.ones(rc.n_out)
    else:
        raise ValueError(f"Experimental images DM3_{x_str}x{x_str} not found")

# output LACBED patterns and figure of merit
if rc.image_processing == 1:
    print(f"  Blur radius {rc.blur_radius} pixels")
if 'S' not in rc.refine_mode:
    # figure of merit
    fom = sim.figure_of_merit(bloch, cbed, rc)
    print(f"  Figure of merit {100*fom:.2f}%")
    print("-------------------------------")


# %% start refinement loop *** needs work
if 'S' not in rc.refine_mode:
    # Initialise variables for refinement
    fit0 = fom*1.0
    rc.best_fit = fom*1.0
    rc.last_fit = fom*1.0
    r3_var = np.zeros(3)  # for parabolic minimum
    r3_fom = np.zeros(3)
    # *_*dunno what this is
    independent_delta = 0.0

    # for a plot
    rc.fit_log = ([rc.last_fit])

    # Refinement loop
    df = 1.0
    while df >= rc.exit_criteria:
        # rc.refined_variable is the working array of variables
        # best_var is the best array of variables during this refinement cycle
        rc.best_var = np.copy(rc.refined_variable)
        # next_var is the predicted next (best) point
        rc.next_var = np.copy(rc.refined_variable)

        if rc.refine_method == 0:
            print("Gradient descent, one parameter at a time")
            # dydx is a vector along the gradient in n-dimensional space
            dydx = np.zeros(rc.n_variables)
            for i in range(rc.n_variables):
                dydx[i] = 1.0
                print(f"Refinement vector {dydx}")
                # single is just multiparameter with one non-zero value
                # rc.next_var = rc.best_var - dydx*rc.refinement_scale
                dydx = sim.refine_multi_variable(xtal, basis, cell, hkl,
                                                 bloch, cbed, rc, dydx)

        elif rc.refine_method == 1:
            print("Multiparameter refinement, finding parameter gradients")
            # =========== step 1: individual variable minimisation
            # if all variables have been refined, reset
            if np.sum(np.abs(dydx)) < 1e-10:
                dydx = np.ones(rc.n_variables)
            # Go through the variables looking at three points in the hope
            # of capturing a minimum - if there is one, we take it and remove
            # that variable from multidimensional refinement, dydx[i] = 0.
            # Otherwise dydx[i] is the gradient for that variable.
            # We also get a predicted best starting point
            # for gradient descent, rc.next_var
            for i in range(rc.n_variables):
                # Skip variables already optimized
                if abs(dydx[i]) < 1e-10:
                    dydx[i] = 0.0
                    continue
                dydx[i] = sim.refine_single_variable(xtal, basis, cell, hkl,
                                                     bloch, cbed, rc, i)

            # all variables have updated/predicted so do a final simulation
            # if it's better, update rc.best_fit and rc.best_var accordingly
            if np.count_nonzero(dydx) == 0:
                print("Closing simulation for this cycle")
                rc.refined_variable = np.copy(rc.next_var)
                fom = sim.sim_fom(xtal, basis, hkl, bloch, cbed, rc, i)
                if (fom < rc.best_fit):
                    rc.best_fit = fom*1.0
                    rc.best_var = np.copy(rc.refined_variable)
            print("Vector gradient descent")
            # ===========step 2: vector descent
            # Downhill minimisation until we eliminate all variables
            while np.sum(np.abs(dydx)) > 1e-10:
                # the returned dydx will have an extra zero!
                dydx = sim.refine_multi_variable(xtal, basis, cell, hkl,
                                                 bloch, cbed, rc, dydx, False)
        else:
            raise ValueError("No valid refine method (0,1) in felix.inp")
        if rc.plot > 0:
            sim.plot_progress(rc)
            sim.print_LACBED(bloch, cbed, rc, 0)

        # Update for next iteration
        df = rc.last_fit - rc.best_fit

        rc.last_fit = np.copy(rc.best_fit)
        rc.refined_variable = np.copy(rc.best_var)
        # reduce refinement scale for next round
        rc.refinement_scale *= (1 - 1 / (1 + rc.n_variables))
        print(f"Improvement in fit {100*df:.2f}%, will stop at {100*rc.exit_criteria:.2f}%")
        if df >= rc.exit_criteria:
            print(f"Step size reduced to {rc.refinement_scale:.6f}")
        print("-------------------------------")
    print(f"Refinement complete after {rc.iter_count} simulations.  Refined values: {rc.best_var}")

# %% final print
sim.print_LACBED(bloch, cbed, rc, 0)
sim.print_LACBED(bloch, cbed, rc, 2)
sim.save_LACBED(xtal, bloch, cbed, rc)
total_time = time.time() - start
print("-----------------------------------------------------------------")
print(f"Total time {total_time:.1f} s")
print("-----------------------------------------------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
