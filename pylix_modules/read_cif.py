# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:01:17 2024

@author: Jacob Watkiss
"""

from CifFile import ReadCif
import numpy as np
import re


# function to set correct data type and remove blank spaces
def clean(var, dtype):
    var=var.strip()
    if (dtype == "str"):
        var = var.replace("'","")
        var = var.replace(" ","")
        var = str(var)
    elif (dtype == "int"):
        var = int(var)
    elif (dtype == "real"):
        if re.search(r"\(.\)", var):
            var = re.sub(r"\(.\)","", var)
        elif re.search(r"\(..\)", var):
            var = re.sub(r"\(..\)","", var)
        var=float(var.strip())
    return var

# function to reads the value associated with the inputted label
def cif_find(file,target,filename,dtype):
    found = False
    for line in file:
        if target in line:
            found = True
            line = line.replace(target,"")
            line = clean(line,dtype)
            return line
    if (found == False):
        print(target,"not found in .cif")
        return "absent"


# Converts part of a symmetry operation into a vector and constant
def symop_part(sym):
    # vector part
    if ("x" in sym):
        vec = np.array([1, 0, 0])
    if ("-x" in sym):
        vec = np.array([-1, 0, 0])
    if ("y" in sym):
        vec = np.array([0, 1, 0])
    if ("-y" in sym):
        vec = np.array([0, -1, 0])
    if ("z" in sym):
        vec = np.array([0, 0, 1])
    if ("-z" in sym):
        vec = np.array([0, 0, -1])
    # constant - we expect a fraction
    if (sym.find("/") == -1):  # there is no fraction
        const = 0
    else:  # we expect single digit numerator and denominator
        sign = 1
        if ("-" in sym[sym.find("/")-2]):
            sign = -1
        const = sign * int(sym[sym.find("/")-1])/int(sym[sym.find("/")+1])

    return vec, const


# Converts symmetry operation xyz form into matrix+vector form
def symop_convert(symop_xyz):
    symmetry_count = len(symop_xyz)
    mat = np.zeros((symmetry_count,3,3),dtype = "float")
    vec = np.zeros((symmetry_count,3),dtype = "float")
    # we expect comma-delimited symmetry operations
    for i in range(symmetry_count):
        symop = symop_xyz[i][1]
        c1 = symop.find(",")
        c2 = symop[(c1+1):].find(",")
        
        (mat[i, :, 0], vec[i, 0]) = symop_part(symop[0:c1])
        (mat[i, :, 1], vec[i, 1]) = symop_part(symop[(c1+1):c1+c2+1])
        (mat[i, :, 2], vec[i, 2]) = symop_part(symop[(c1+c2+2):])
        
    return mat, vec


# %%
def read_cif():
    
    filename="felix.cif"
    cif=open(filename,"r")
    content=cif.readlines()
    
    # Reading the data from the file
    # Space Group information
    space_group_number = cif_find(content,"_symmetry_Int_Tables_number",cif,"int")
    space_group_cif = cif_find(content,"_symmetry_space_group_cif_H-M",cif,"str")
    # To remove the quotes in the space group cif:
    space_group_cif = space_group_cif.replace("'","")
    lattice_type = space_group_cif[0]
    
    # Cell dimensions
    cell_length_a = cif_find(content,"_cell_length_a",cif,"real")
    cell_length_b = cif_find(content,"_cell_length_b",cif,"real")
    cell_length_c = cif_find(content,"_cell_length_c",cif,"real")
    alpha = cif_find(content,"_cell_angle_alpha",cif,"real")
    beta = cif_find(content,"_cell_angle_beta",cif,"real")
    gamma = cif_find(content,"_cell_angle_gamma",cif,"real")
    # Convert angles from degrees to radians
    alpha = alpha*np.pi/180
    beta = beta*np.pi/180
    gamma = gamma*np.pi/180

    volume = cif_find(content,"_cell_volume",cif,"real")
    if (volume == "absent"):
        print("...calculated cell volume")
        volume1 = cell_length_a * cell_length_b * cell_length_c
        volume2 = 1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2
        volume3 = 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
        volume = volume1*np.sqrt(volume2+volume3)
    
    # Chemical Formula
    chemical_formula = cif_find(content,"_chemical_formula_sum",cif,"str")
    if chemical_formula == "absent":
        chemical_formula = cif_find(content,"_chemical_formula_moiety",cif,"str")
    LN=int(len(chemical_formula.strip()))
    
    # atoms, their positions, Debye-Waller factors and occupancy
    """
    JW:  have written a lot of this code myself, but could see
    no simple soution to overcoming the format of the atoms
    in the CIF. as a result, PyCifRW has been used where 
    can not find a solution.
    Hence the file is opened twice in two different ways.
    """
    cif_data = ReadCif(filename)
    block = list(cif_data.keys())[0]
    atoms = cif_data[block]
    
    basis_atom_label = np.array(atoms.get("_atom_site_label",[]),dtype="str")
    basis_atom_count = len(basis_atom_label)
    basis_atom_name = np.array(atoms.get("_atom_site_type_symbol",[]),dtype="str")
    
    """
    For the position, a dummy array must be created in order
    to make an array of the correct length with tuples as
    the elements. The CIF is then read, and tuples with each
    atom's position is created and then added to the correct array.
    """
    x = np.array(atoms.get("_atom_site_fract_x", [])).astype('float')
    y = np.array(atoms.get("_atom_site_fract_y", [])).astype('float')
    z = np.array(atoms.get("_atom_site_fract_z", [])).astype('float')
    basis_atom_position  =  np.array([x, y, z])
    
    basis_isoDW = np.array(atoms.get("_atom_site_U_iso_or_equiv", [])).astype('float')
    if (basis_isoDW.size == 0):
        basis_isoDW = np.array(atoms.get("_atom_site_B_iso_or_equiv", [])).astype('float')
        
    basis_occupancy = np.array(atoms.get("_atom_site_occupancy",[])).astype('float')
    if (basis_occupancy.size == 0):
        # print("No Occupancy found, assuming 100% for all atoms")
        basis_occupancy = np.ones(basis_atom_count,dtype = "float")
    
    # Symmetry Operations
    text = np.array(atoms.get("_symmetry_equiv_pos_as_xyz",[]),dtype = "str")
    if (text.size == 0):
        text = np.array(atoms.get("_space_group_symop_operation_xyz",[]),dtype = "str")
        if (text.size == 0):
            print("Error: No Symmetry Groups")    
    sym_count = len(text)
    length = 0
    for i in range(sym_count):
        if len(text[i])>length:
            length = len(text[i])
    length = str(length)
    symop_xyz = np.empty((sym_count,2), dtype = f"<U"+length)
    for i in range(sym_count):
        symop_xyz[i][0] = str(i+1)
        symop_xyz[i][1] = (text[i])
    
    name.close()

    return (space_group_number, basis_atom_count, LN, basis_atom_label,
            basis_atom_name, lattice_type, space_group_name, chemical_formula,
            symop_xyz, cell_length_a, cell_length_b, cell_length_c, alpha, beta, gamma,
            volume, basis_atom_position, basis_isoDW, basis_occupancy)


# %% tester

    
(space_group_number, basis_atom_count, LN, basis_atom_label, 
basis_atom_name, lattice_type, space_group_name, chemical_formula, 
symop_xyz, cell_length_a, cell_length_b, cell_length_c, alpha, beta, gamma, 
volume, basis_atom_position, basis_isoDW, basis_occupancy) = read_cif()