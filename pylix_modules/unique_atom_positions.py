import numpy as np

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

    # Determine the size of the all_atom_position array
    n_symmetry_operations = symmetry_vector.shape[0]
    n_basis_atoms = basis_atom_position.shape[0]
    total_atoms = n_symmetry_operations * n_basis_atoms

    # Initialize arrays to store all atom positions, including duplicates
    all_atom_position = np.zeros((total_atoms, 3))
    all_atom_label = np.empty(total_atoms, dtype='<U5')
    all_atom_name = np.empty(total_atoms, dtype='<U2')
    all_occupancy = np.zeros(total_atoms)
    all_B_iso = np.zeros(total_atoms)

    # Generate all equivalent positions by applying symmetry
    k = 0
    for i in range(n_symmetry_operations):
        for j in range(n_basis_atoms):
            all_atom_position[k, :] = np.dot(symmetry_matrix[i, :, :],
                            basis_atom_position[j, :]) + symmetry_vector[i, :]
            all_atom_label[k] = basis_atom_label[j]
            all_atom_name[k] = basis_atom_name[j]
            all_occupancy[k] = basis_occupancy[j]
            all_B_iso[k] = basis_B_iso[j]
            k += 1

    # Normalize positions to be within [0, 1]
    all_atom_position %= 1.0
    # make small values precisely zero
    all_atom_position[np.abs(all_atom_position) < 1e-05] = 0.0

    # Reduce to the set of unique fractional atomic positions
    # first atom in the long list is always in the reduced list
    atom_label = [all_atom_label[0]]  # NB [] makes a list
    atom_name = [all_atom_name[0]]
    # we make the coordinates a 1D array and reshape later
    atom_position = np.squeeze(all_atom_position[0, :])
    B_iso = [all_B_iso[0]]
    occupancy = [all_occupancy[0]]

    k = 1
    for i in range(1, total_atoms):
        unique = True
        for j in range(k):
            dr = np.abs(atom_position[3*j:3*j+3] - all_atom_position[i, :])
            if np.sum(dr) <= 1e-10:  # maybe we've seen this atom already?
                if all_atom_label[i] == atom_label[j]:  # yes, move on
                    unique = False
                    break
        if unique:
            atom_label.append(all_atom_label[i])
            atom_name.append(all_atom_name[i])
            atom_position = np.append(atom_position, all_atom_position[i, :])
            B_iso.append(all_B_iso[i])
            occupancy.append(all_occupancy[i])
            k += 1
    atom_position = atom_position.reshape(len(atom_position)//3, 3)

    return atom_position, atom_label, atom_name, B_iso, occupancy

    # # Calculate atomic position vectors in the microscope reference frame (Angstrom units)
    # for i in range(n_atoms_unit_cell):
    #     atom_coordinate[i, :] = (
    #         atom_position[i, 0] * a_vec_m +
    #         atom_position[i, 1] * b_vec_m +
    #         atom_position[i, 2] * c_vec_m
    #     )

    # return atom_position, atom_label, atom_name, B_iso, occupancy