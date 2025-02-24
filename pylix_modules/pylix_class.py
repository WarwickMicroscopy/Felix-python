# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:54:05 2024

@author: Richard
"""
# ----------------------------------------------------------------------------
# variables to get from felix.inp
# class microscope:
class Inp:
    def __init__(self):
        self.accelerating_voltage_kv = None
        self.convergence_angle = None
    
# simulation:
        self.image_radius = None
        self.min_reflection_pool = None
        self.min_strong_beams = None
        self.min_weak_beams = None
        self.g_limit = None
        self.scatter_factor_method = None
        self.absorption_method = None
        self.absorption_per = None
        self.holz_flag = None
        self.debug = None
        self.plot = None
        self.n_output_reflexions = None
        self.initial_thickness = None
        self.final_thickness = None
        self.delta_thickness = None
        self.debye_waller_constant = None

# sample:
        self.incident_beam_direction = None
        self.x_direction = None
        self.normal_direction = None

# experimental data
        self.n_frames = None
        self.frame_angle = None
        self.frame_size_x = None
        self.frame_size_y = None
        self.frame_resolution = None  # reciprocal Angstroms/pixel
        self.frame_output = None  # flag to say if we output kinematical frames

# refinement:
        self.refine_mode = None
        self.refine_method = None
        self.refinement_scale = None
        self.weighting_flag = None
        self.exit_criteria = None
        self.precision = None
        self.correlation_type = None
        self.image_processing = None
        self.atomic_sites = None
        self.no_of_ugs = None
        self.blur_radius = None
        self.print_flag = None

# class inp:
#     def __init__(self, microscope, simulation, sample, refinement):
#         self.microscope = microscope
#         self.simulation = simulation
#         self.sample = sample
#         self.refinement = refinement

    def update_from_dict(self, data):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

# ----------------------------------------------------------------------------
# variables to get from felix.cif
# class basis:
class Cif:
    def __init__(self):
        self.atom_site_b_iso_or_equiv = None
        self.atom_site_label = None
        self.atom_site_type_symbol = None
        self.atom_site_fract_x = None
        self.atom_site_fract_y = None
        self.atom_site_fract_z = None
        self.atom_site_occupancy = None
        self.atom_site_u_iso_or_equiv = None
        self.atom_site_wyckoff_symbol = None

# class cell:
#     def __init__(self):
        self.cell_angle_alpha = None
        self.cell_angle_beta = None
        self.cell_angle_gamma = None
        self.cell_length_a = None
        self.cell_length_b = None
        self.cell_length_c = None
        self.cell_volume = None

# class chemical:
#     def __init__(self):
        self.chemical_formula_iupac = None
        self. chemical_formula_structural = None
        self.chemical_formula_sum = None

# class symmetry:
#     def __init__(self):
        self.space_group_it_number = None
        self.space_group_name_h_m_alt = None
        self.space_group_symbol = None
        self.space_group_symop_operation_xyz = None
        self.symmetry_equiv_pos_as_xyz = None
        self.symmetry_space_group_name_h_m = None

# class cif:
#     def __init__(self, basis, cell, chemical, symmetry):
#         self.basis = basis
#         self.cell = cell
#         self.chemical = chemical
#         self.symmetry = symmetry
        
    def update_from_dict(self, data):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

# ----------------------------------------------------------------------------
# working variables for simulation
class Var(Inp, Cif):
    def __init__(self):
        Inp.__init__(self)
        Cif.__init__(self)
    # self.chemical_formula = None
    # self.space_group = None
    # self.space_group_number = None
    # self.lattice_type = None
    # self.cell_a = None
    # self.cell_b = None
    # self.cell_c = None
    # self.cell_alpha = None
    # self.cell_beta = None
    # self.cell_gamma = None
    # self.n_basis = None
    # self.symmetry_matrix = None
    # self.symmetry_vector = None
    # self.basis_atom_label = None
    # self.basis_atom_name = None
    # self.basis_wyckoff = None
    # self.basis_atom_position = None
    # self.basis_occupancy = None
    # self.basis_B_iso = None
    # self.thickness = None
    # self.n_thickness = None
    # self.incident_beam_direction = None
    # self.normal_direction = None
    # self.x_direction = None
    # self.convergence_angle = None
    # self.accelerating_voltage_kv = None
    # self.atomic_sites = None
    # self.g_limit = None
    # self.refine_mode = None
    # self.scatter_factor_method = None
    # self.plot = None
    # self.debug = None
    # self.input_hkls = None
    # self.i_obs = None
    # self.sigma_obs = None
    # self.refined_variable = None
    # self.refined_variable_type = None
    # self.atom_refine_flag = None
    # self.refined_variable_atom = None
    # self.thickness = None
    # self.thickness = None
    # self.thickness = None
    # self.thickness = None
    # self.thickness = None
    # self.thickness = None
    # self.thickness = None
    # self.thickness = None
        