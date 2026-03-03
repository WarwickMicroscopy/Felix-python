# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 08:54:05 2024

@author: Richard
"""
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------
# variables in felix.inp
@dataclass
class RunControl:
    # control
    write_flag = None
    scatter_factor_method = None
    holz_flag = None
    absorption_method = None
    absorption_per = None
    Debye_model = None
    accelerating_voltage_kv = None
    convergence_angle = None

    # beam selection criteria:
    min_reflection_pool = None
    min_strong_beams = None
    min_weak_beams = None
    g_limit = None

    debug = None
    plot = None
    n_output_reflexions = None

    # sample:
    incident_beam_direction = None
    x_direction = None
    normal = None
    initial_thickness = None
    final_thickness = None
    delta_thickness = None
    debye_waller_constant = None

    # microscope
    convergence_angle = None
    accelerating_voltage_kv = None
    acceptance_angle = None

    # output
    image_radius = None
    plot = None
    image_processing = None
    blur_radius = None
    print_flag = None
    debug = None

    # refinement:
    refine_mode = None
    refine_method = None
    refinement_scale = None
    weighting_flag = None
    exit_criteria = None
    precision = None
    correlation_type = None
    atomic_sites = None
    no_of_ugs = None

    def update_from_dict(self, data):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

# ----------------------------------------------------------------------------
# run control variables derived from felix.inp

    thickness: NDArray[np.floating] | None = None
    n_thickness: int = 0
    n_out: int = 0  # number of output reflections
    refined_variable: NDArray[np.floating] | None = None
    refined_variable_type: NDArray[np.integer] | None = None
    atom_refine_flag: NDArray[np.integer] | None = None
    atom_refine_vec: NDArray[np.floating] | None = None
    atom_coord_vec: NDArray[np.floating] | None = None
    moves: NDArray[np.floating] | None = None
    n_variables: int = 0
    best_fit: float = 0.0
    last_fit: float = 0.0
    fit_log: NDArray[np.floating] | None = None
    df: float = 1.0
    iter_count: int = 0
    plot: int = 0
#     p: np.ndarray = field(default_factory=lambda: np.array([]))
#     last_p: np.ndarray = field(default_factory=lambda: np.array([]))
#     var_pl: list = field(default_factory=list)
#     fit_pl: list = field(default_factory=list)


# ----------------------------------------------------------------------------
# variables used in the Bloch wave calculation
@dataclass
class Bloch:
    electron_velocity: float = 0.0
    relativistic_correction: float = 0.0
    big_k: NDArray[np.floating] | None = None
    big_k_mag: float = 0.0
    hkl_indices: NDArray[np.integer] | None = None
    n_hkl: int = 0
    hkl_output: NDArray[np.integer] | None = None
    n_out: int = 0
    g_pool: NDArray[np.floating] | None = None
    g_pool_mag: NDArray[np.floating] | None = None
    g_matrix: NDArray[np.floating] | None = None
    g_dot_norm: NDArray[np.floating] | None = None
    s_g: NDArray[np.floating] | None = None
    s_g_pix: NDArray[np.floating] | None = None
    tilted_k: NDArray[np.floating] | None = None
    k_dot_n: NDArray[np.floating] | None = None
    k_dot_n_pix: NDArray[np.floating] | None = None
    wave_function: NDArray[np.complex128] | None = None
    strong_beam: NDArray[np.integer] | None = None
    gamma: NDArray[np.complex128] | None = None
    eigenvecs: NDArray[np.complex128] | None = None
    # inv_eigenvecs: NDArray[np.floating] | None = None
    n_beams: int = 0
    strong_beam_indices: NDArray[np.integer] | None = None


# ----------------------------------------------------------------------------
# variables output
@dataclass
class Cbed:
    lacbed_sim: NDArray[np.floating] | None = None
    diff_image: NDArray[np.floating] | None = None


# ----------------------------------------------------------------------------
# variables in felix.cif
@dataclass
class Cif:
    # basis:
    atom_site_b_iso_or_equiv = None
    atom_site_aniso_label = None
    atom_site_aniso_type_symbol = None
    atom_site_label = None
    atom_site_type_symbol = None
    atom_site_symmetry_multiplicity = None
    atom_site_fract_x = None
    atom_site_fract_y = None
    atom_site_fract_z = None
    atom_site_occupancy = None
    atom_site_u_iso_or_equiv = None
    atom_site_wyckoff_symbol = None
    atom_site_aniso_u_11 = None
    atom_site_aniso_u_22 = None
    atom_site_aniso_u_33 = None
    atom_site_aniso_u_12 = None
    atom_site_aniso_u_13 = None
    atom_site_aniso_u_23 = None
    atom_site_aniso_label = None
    atom_site_aniso_type_symbol = None
    atom_type_symbol = None
    atom_type_oxidation_number = None

    # cell:
    cell_angle_alpha = None
    cell_angle_beta = None
    cell_angle_gamma = None
    cell_length_a = None
    cell_length_b = None
    cell_length_c = None
    cell_volume = None

    # chemical:
    chemical_formula_iupac = None
    chemical_formula_structural = None
    chemical_formula_sum = None

    #  symmetry:
    space_group_it_number = None
    space_group_name_h_m_alt = None
    space_group_symbol = None
    space_group_symop_id = None
    space_group_symop_operation_xyz = None
    symmetry_equiv_pos_as_xyz = None
    symmetry_space_group_name_h_m = None

    def update_from_dict(self, data):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ----------------------------------------------------------------------------
# classes derived from the .cif file
@dataclass
class Crystal:
    space_group: str = ""
    chemical_formula: str = ""
    lattice_type: str = ""
    space_group_number: int = 0
    symmetry_matrix: NDArray[np.integer] | None = None
    symmetry_vector: NDArray[np.floating] | None = None
    cell_a: float = 0.0
    cell_b: float = 0.0
    cell_c: float = 0.0
    cell_alpha: float = 0.0
    cell_beta: float = 0.0
    cell_gamma: float = 0.0
    cell_volume: float = 0.0
    a_vec_m: NDArray[np.floating] | None = None
    b_vec_m: NDArray[np.floating] | None = None
    c_vec_m: NDArray[np.floating] | None = None
    ar_vec_m: NDArray[np.floating] | None = None
    br_vec_m: NDArray[np.floating] | None = None
    cr_vec_m: NDArray[np.floating] | None = None
    norm_dir_m: NDArray[np.floating] | None = None
    t_mat_o2m: NDArray[np.floating] | None = None
    t_mat_c2o: NDArray[np.floating] | None = None

@dataclass
class Basis:
    n_atoms: int = 0
    atom_label: list | None = None
    atom_name: list | None = None
    atomic_number: NDArray[np.integer] | None = None
    wyckoff: list | None = None
    atom_position: NDArray[np.floating] | None = None
    occupancy: NDArray[np.floating] | None = None
    mult_occ: NDArray[np.integer] | None = None
    u_aniso: NDArray[np.floating] | None = None
    u_iso: NDArray[np.floating] | None = None
    B_iso: NDArray[np.floating] | None = None
    oxno: NDArray[np.integer] | None = None
    atom_delta: NDArray[np.floating] | None = None
    f_g: NDArray[np.complex128] | None = None
    f_g_prime: NDArray[np.complex128] | None = None


@dataclass
class Cell:
    n_atoms: int = 0
    atom_label: list | None = None
    atom_name: list | None = None
    atomic_number: NDArray[np.integer] | None = None
    wyckoff: list | None = None
    atom_position: NDArray[np.floating] | None = None  # in cell
    atom_coordinate: NDArray[np.floating] | None = None  # in microscope frame
    occupancy: NDArray[np.floating] | None = None
    mult_occ: NDArray[np.integer] | None = None
    u_aniso: NDArray[np.floating] | None = None
    u_iso: NDArray[np.floating] | None = None
    B_iso: NDArray[np.floating] | None = None
    oxno: NDArray[np.integer] | None = None
    atom_delta: NDArray[np.floating] | None = None
    f_g: NDArray[np.complex128] | None = None


# -----------------------------------------------
@dataclass
class Hkl:  # hkl file
    input_hkls: NDArray[np.integer] | None = None
    i_obs: NDArray[np.floating] | None = None
    sigma_obs: NDArray[np.floating] | None = None
