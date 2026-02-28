# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 09:17:25 2026

@author: rbean
"""

import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray

# classes from the .cif file
@dataclass
class CrystalInfo:
    space_group: str
    chemical_formula: str
    lattice_type: str
    space_group_number: int


@dataclass
class Symmetry:
    symmetry_matrix: NDArray[np.integer]
    symmetry_vector: NDArray[np.floating]


@dataclass
class UnitCell:
    cell_a: float
    cell_b: float
    cell_c: float
    cell_alpha: float
    cell_beta: float
    cell_gamma: float


@dataclass
class Basis:
    basis_atom_label: list
    basis_atom_name: list
    basis_atomic_number: NDArray[np.integer]

    basis_wyckoff: list
    basis_atom_position: NDArray[np.floating]
    basis_occupancy: NDArray[np.floating]
    basis_mult_occ: NDArray[np.integer]

    basis_u_ij: NDArray[np.floating]
    basis_u_iso: NDArray[np.floating] | None
    basis_B_iso: NDArray[np.floating] | None

    basis_oxno: NDArray[np.integer]
    basis_atom_delta: NDArray[np.floating]


@dataclass
class Crystal:
    info: CrystalInfo
    cell: UnitCell
    symmetry: Symmetry
    basis: Basis

#-----------------------------------------------

# classes from felix.inp
@dataclass
class RunControl:
    iter_count: int = 0
    debug: bool = False


@dataclass
class InputControl:
    refine_mode: str
    correlation_type: int

    initial_thickness: float
    final_thickness: float
    delta_thickness: float
    scatter_factor_method: int
    absorption_method: int
    absorption_per: float

    incident_beam_direction: NDArray[np.floating]
    normal_direction: NDArray[np.floating]
    x_direction: NDArray[np.floating]
    g_limit: float
    atomic_sites: NDArray[np.integer]


@dataclass
class Thickness:
    thickness: NDArray[np.floating]
    n_thickness: int


@dataclass
class ScatteringHF:
    basis_pv: NDArray[np.floating]
    basis_pc: NDArray[np.floating]
    basis_kappa: NDArray[np.floating]

    n_points: int
    r_max: float

    basis_core: NDArray[np.floating]
    basis_valence: NDArray[np.floating]
    basis_r2: NDArray[np.floating]


class SimulationState:
    ctrl: InputControl
    run: RunControl
    thick: Thickness
    scat: ScatteringHF | None


#-----------------------------------------------
# refinement classes
class RefinementState:
    refined_variable: np.ndarray
    refined_variable_type: np.ndarray
    atom_refine_flag: np.ndarray
    atom_refine_vec: np.ndarray
    n_variables: int

#-----------------------------------------------
# code snippets to go into felixrefine

run = RunControl(iter_count=0)

if (ctrl.final_thickness > ctrl.initial_thickness + ctrl.delta_thickness):
    thickness = np.arange(ctrl.initial_thickness,
                           ctrl.final_thickness,
                           ctrl.delta_thickness)
    thick = Thickness(thickness=thickness,
                      n_thickness=len(thickness))
else:
    thickness = np.atleast_1d(ctrl.initial_thickness)
    thick = Thickness(thickness=thickness,
                      n_thickness=1)


inp_dict = px.read_inp_file('felix.inp')
ctrl = InputControl(**inp_dict)

ctrl.incident_beam_direction = np.array(ctrl.incident_beam_direction, dtype=float)
ctrl.normal_direction = np.array(ctrl.normal_direction, dtype=float)
ctrl.x_direction = np.array(ctrl.x_direction, dtype=float)
ctrl.atomic_sites = np.array(ctrl.atomic_sites, dtype=int)

ctrl.g_limit *= 2 * np.pi


if ctrl.scatter_factor_method == 4:
    scat = ScatteringHF(
        basis_pv=np.zeros(n_basis),
        basis_pc=np.zeros(n_basis),
        basis_kappa=np.ones(n_basis),
        n_points=1000,
        r_max=20,
        basis_core=np.zeros((n_basis, 1000)),
        basis_valence=np.zeros((n_basis, 1000)),
        basis_r2=np.zeros(n_basis),
    )

    for i in range(n_basis):
        orbi = px.orb(basis_atomic_number[i])
        scat.basis_pv[i] = orbi["pv"]
        scat.basis_pc[i] = orbi["pc"]
        scat.basis_core[i, :], scat.basis_valence[i, :], scat.basis_r2[i] = \
            px.precompute_densities(
                basis_atomic_number[i],
                scat.basis_kappa[i],
                scat.basis_pv[i]
            )

v.refine = RefinementState()

v.refine.refined_variable = []
v.refine.refined_variable_type = []
v.refine.atom_refine_flag = []
v.refine.atom_refine_vec = []

# %% set up refinement
nullvec = np.array([0, 0, 0])

v.refine.refined_variable = []
v.refine.refined_variable_type = []
v.refine.atom_refine_flag = []
v.refine.atom_refine_vec = []

if 'S' not in v.refine_mode:
    v.refine.n_variables = 0

    if 'B' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            moves = px.atom_move(
                v.space_group_number,
                v.basis_wyckoff[v.atomic_sites[i]]
            )

            degrees_of_freedom = np.sum(np.any(moves, axis=1))
            if degrees_of_freedom == 0:
                raise ValueError(
                    f"Coordinate refinement of atom {v.atomic_sites[i]} not possible"
                )

            for j in range(degrees_of_freedom):
                r_dot_v = np.dot(
                    v.basis_atom_position[v.atomic_sites[i]],
                    moves[j, :]
                )
                v.refine.refined_variable.append(r_dot_v)
                v.refine.refined_variable_type.append(20)
                v.refine.atom_refine_flag.append(v.atomic_sites[i])
                v.refine.atom_refine_vec.append(moves[j, :])

    if 'C' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            v.refine.refined_variable.append(
                v.basis_occupancy[v.atomic_sites[i]]
            )
            v.refine.refined_variable_type.append(21)
            v.refine.atom_refine_flag.append(v.atomic_sites[i])
            v.refine.atom_refine_vec.append(nullvec)

    if 'D' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            v.refine.refined_variable.append(
                v.basis_B_iso[v.atomic_sites[i]]
            )
            v.refine.refined_variable_type.append(22)
            v.refine.atom_refine_flag.append(v.atomic_sites[i])
            v.refine.atom_refine_vec.append(nullvec)

    if 'E' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            U = v.basis_u_ij[v.atomic_sites[i]]
            aniso_params = np.array([
                U[0, 0], U[1, 1], U[2, 2],
                U[0, 1], U[0, 2], U[1, 2]
            ])
            anisotypes = [23, 24, 25, 26, 27, 28]

            for param, t in zip(aniso_params, anisotypes):
                if abs(param) > eps:
                    v.refine.refined_variable.append(param)
                    v.refine.refined_variable_type.append(t)
                    v.refine.atom_refine_flag.append(v.atomic_sites[i])
                    v.refine.atom_refine_vec.append(nullvec)

    if 'F' in v.refine_mode:
        v.refine.refined_variable.append(v.cell_a)
        v.refine.refined_variable_type.append(30)
        v.refine.atom_refine_flag.append(-1)
        v.refine.atom_refine_vec.append(nullvec)

        if v.space_group_number < 75:
            v.refine.refined_variable.append(v.cell_b)
            v.refine.refined_variable_type.append(31)
            v.refine.atom_refine_flag.append(-1)
            v.refine.atom_refine_vec.append(nullvec)

            v.refine.refined_variable.append(v.cell_c)
            v.refine.refined_variable_type.append(32)
            v.refine.atom_refine_flag.append(-1)
            v.refine.atom_refine_vec.append(nullvec)

        elif 142 < v.space_group_number < 160:
            raise ValueError("Rhombohedral R- vs H- not yet implemented")

        elif (160 < v.space_group_number < 195) or \
             (74 < v.space_group_number < 143):
            v.refine.refined_variable.append(v.cell_c)
            v.refine.refined_variable_type.append(32)
            v.refine.atom_refine_flag.append(-1)
            v.refine.atom_refine_vec.append(nullvec)

    if 'H' in v.refine_mode:
        v.refine.refined_variable.append(v.convergence_angle)
        v.refine.refined_variable_type.append(40)
        v.refine.atom_refine_flag.append(-1)
        v.refine.atom_refine_vec.append(nullvec)

    if 'I' in v.refine_mode:
        v.refine.refined_variable.append(v.accelerating_voltage_kv)
        v.refine.refined_variable_type.append(41)
        v.refine.atom_refine_flag.append(-1)
        v.refine.atom_refine_vec.append(nullvec)

    if 'J' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            v.refine.refined_variable.append(
                v.basis_kappa[v.atomic_sites[i]]
            )
            v.refine.refined_variable_type.append(50)
            v.refine.atom_refine_flag.append(v.atomic_sites[i])
            v.refine.atom_refine_vec.append(nullvec)

    if 'K' in v.refine_mode:
        for i in range(len(v.atomic_sites)):
            v.refine.refined_variable.append(
                v.basis_pv[v.atomic_sites[i]]
            )
            v.refine.refined_variable_type.append(51)
            v.refine.atom_refine_flag.append(v.atomic_sites[i])
            v.refine.atom_refine_vec.append(nullvec)

    v.refine.n_variables = len(v.refine.refined_variable)

    if v.refine.n_variables == 0 and v.refine_mode != 'O':
        raise ValueError("No refinement variables!")

    v.refine.refined_variable = np.array(v.refine.refined_variable)
    v.refine.refined_variable_type = np.array(v.refine.refined_variable_type)
    v.refine.atom_refine_flag = np.array(v.refine.atom_refine_flag)
    v.refine.atom_refine_vec = np.array(v.refine.atom_refine_vec)