import numpy as np


def change_origin(space_group, basis_atom_position, basis_wyckoff):
    """
    Change to the preferred origin for a given space group.

    Parameters:
    space_group (int): The space group number.

    Returns:
    basis_atom_position (ndarray): the updated basis atom fractional coords.
    """
    change_flag = 0
    n_basis_atoms = basis_atom_position.shape[0]  # Number of basis atoms

    # Only implemented for space group #142 (I41/acd)
    if space_group == 142:
        # Change from choice 1 (origin -4 at [0,0,0])
        # to choice 2 (origin -1 at [0,0,0])
        # Look for an 'a' site incompatible with choice 2
        for i in range(n_basis_atoms):
            # 'a' is -4 at [000],[0,1/2,1/2],[0,1/2,1/4],[1/2,0,1/4] in 1
            # and [0,1/4,3/8], [0,3/4,5/8], [1/2,1/4,5/8], [1/2,3/4,5/8] in 2
            if basis_wyckoff[i] == 'a':
                # Check for origin 1
                # by multiplying by 4 and checking if it is an integer
                if np.mod(4 * basis_atom_position[i, 2], 1.0) < 1e-10:
                    change_flag = 1
        # add [0,1/4,3/8] if change_flag is set
        if change_flag == 1:
            basis_atom_position[:, 1] = np.mod(basis_atom_position[:, 1]
                                               + 0.25, 1.0)
            basis_atom_position[:, 2] = np.mod(basis_atom_position[:, 2]
                                               + 0.375, 1.0)
    return basis_atom_position


def preferred_basis(space_group, basis_atom_position, basis_wyckoff):

    n_basis_atoms = len(basis_atom_position)
    basis_atom_position = change_origin(space_group)

    # Loop over basis atoms
    for i in range(n_basis_atoms):
        wyckoff_symbol = basis_wyckoff[i]

        if space_group == 1:  # P1
            if wyckoff_symbol != 'a':
                raise ValueError("Wyckoff Symbol for P1 not recognised")
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
# 10
# 11
# 12
# 13
# 14
        elif space_group == 15:  # C2/c
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for C 2/c not recognised")
# 16
# 17
# 18
# 19
# 20
# 21
# 22
# 23
# 24
# 25
# 26
# 27
# 28
# 29
# 30
# 31
# 32
# 33
# 34
# 35
        elif space_group == 36:  # C m c 21
            if wyckoff_symbol in ['a', 'b']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for C m c 21 not recognised")
# 37
# 38
# 39
# 40
# 41
# 42
# 43
# 44
# 45
# 46
# 47
# 48
# 49
# 50
# 51
# 52
# 53
# 54
# 55
# 56
# 57
# 58
# 59
# 60
# 61
# 62
        elif space_group == 63:  # C m c m
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for C m c m not recognised")
        elif space_group == 64:  # C m c a
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for C m c a not recognised")
# 65
# 66
# 67
        elif space_group == 68:  # C c c a
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for Ccca not recognised")
# 69
# 70
# 71
# 72
# 73
# 74
# 75
# 76
# 77
# 78
# 79
# 80
# 81
# 82
# 83
# 84
# 85
# 86
# 87
# 88
# 89
# 90
# 91
# 92
# 93
# 94
# 95
# 96
# 97
# 98
        elif space_group == 99:  # P 4 m m
            if wyckoff_symbol in ['a', 'b', 'c', 'g']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'd':
                # Change equivalent coordinate [x,-x,z] to [x,x,z]
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'e':
                # Change equivalent coordinate [0,x,z] to [x,0,z]
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'f':
                # Change equivalent coordinate [1/2,x,z] to [x,1/2,z]
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            else:
                raise ValueError("Wyckoff Symbol for P 4 m m not recognised")
# 100
# 101
# 102
# 103
# 104
# 105
# 106
# 107
# 108
# 109
# 110
# 111
# 112
# 113
# 114
# 115
# 116
# 117
# 118
# 119
# 120
# 121
# 122
# 123
# 124
# 125
# 126
# 127
# 128
# 129
# 130
# 131
# 132
# 133
# 134
# 135
# 136
# 137
# 138
        elif space_group == 139:  # I4/m m m
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'l', 'o']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'h':
                # Change to [x,x,0]
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'i':
                # Change equivalent coordinate [0,y,0]
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'j':
                # Change equivalent coordinate [1/2,y,0]
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            elif wyckoff_symbol == 'k':
                # Change to [x,1/2+x,1/4]
                basis_atom_position[i, 1] = basis_atom_position[i, 0] + 0.5
            elif wyckoff_symbol == 'm':
                # Change to [x,x,z]
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'n':
                # Change equivalent coordinate [0,y,z]
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            else:
                raise ValueError("Wyckoff Symbol for I4/m m m not recognised")
# 140
# 141
        elif space_group == 142:  # I41/acd
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'g']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'e':
                if abs(basis_atom_position[i, 2]) < 1e-10:
                    if abs(basis_atom_position[i, 0] - 0.25) < 1e-10:
                        basis_atom_position[i, 0] = 0.25 - basis_atom_position[i, 1]
                        basis_atom_position[i, 1] = 0.0
                        basis_atom_position[i, 2] = 0.25
                    elif abs(basis_atom_position[i, 0] - 0.75) < 1e-10:
                        basis_atom_position[i, 0] = basis_atom_position[i, 1] - 0.25
                        basis_atom_position[i, 1] = 0.0
                        basis_atom_position[i, 2] = 0.25
            elif wyckoff_symbol == 'f':
                if abs(basis_atom_position[i, 2] - 0.375) < 1e-10:
                    basis_atom_position[i, 1] = basis_atom_position[i, 0] + 0.25
                    basis_atom_position[i, 2] = 0.125
                elif abs(basis_atom_position[i, 2] - 0.625) < 1e-10:
                    basis_atom_position[i, 1] = basis_atom_position[i, 0] + 0.25
                    basis_atom_position[i, 2] = 0.125
            else:
                raise ValueError("Wyckoff Symbol for 142, I41/a c d not recognised")
# 143
# 144
# 145
# 146
# 147
# 148
# 149
# 150
# 151
# 152
# 153
# 154
# 155
# 156
# 157
# 158
# 159
# 160
# 161
# 162
# 163
# 164
# 165
# 166
# 167
# 168
# 169
# 170
# 171
# 172
# 173
# 174
# 175
# 176
# 177
# 178
# 179
# 180
# 181
# 182
# 183
# 184
# 185
# 186
# 187
# 188
# 189
# 190
        elif space_group == 191:  # P 6 m m
            if wyckoff_symbol in ['a', 'd', 'g']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'b':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'c':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'e':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            elif wyckoff_symbol == 'f':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            else:
                raise ValueError("Wyckoff Symbol for P6mm not recognised")
# 192
# 193
        elif space_group == 194:  # P 6/mmm
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e',
                                  'f', 'g', 'h', 'm', 'r']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'i':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'j':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'k':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            elif wyckoff_symbol == 'l':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'n':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'o':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'p':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            elif wyckoff_symbol == 'q':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            else:
                raise ValueError("Wyckoff Symbol for P6/mmm not recognised")
# 195
# 196
# 197
# 198
# 199
# 200
# 201
# 202
# 203
# 204
# 205
# 206
# 207
# 208
# 209
# 210
# 211
# 212
# 213
# 214
# 215
# 216
# 217
# 218
# 219
# 220
        elif space_group == 221:  # Pm-3m
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e', 'f', 'm']:
                pass  # no reassignment needed
            elif wyckoff_symbol == 'g':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
                basis_atom_position[i, 2] = 0.0
            elif wyckoff_symbol == 'h':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
                basis_atom_position[i, 2] = 0.0
            elif wyckoff_symbol == 'i':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
                basis_atom_position[i, 2] = 0.0
            elif wyckoff_symbol == 'j':
                basis_atom_position[i, 1] = basis_atom_position[i, 0]
            elif wyckoff_symbol == 'k':
                if abs(basis_atom_position[i, 0]) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.0
            elif wyckoff_symbol == 'l':
                if abs(basis_atom_position[i, 0] - 0.5) < 1e-10:
                    basis_atom_position[i, 0] = basis_atom_position[i, 1]
                    basis_atom_position[i, 1] = 0.5
            else:
                raise ValueError("Wyckoff Symbol for Pm-3m not recognised")
# 222
# 223
# 224
# 225
# 226
        elif space_group == 227:  # F d -3 m
            if wyckoff_symbol in ['a', 'b', 'c', 'd', 'e',
                                  'f', 'g', 'h', 'i', 'j']:
                pass  # no reassignment needed
            else:
                raise ValueError("Wyckoff Symbol for Fd-3m not recognised")
# 228
# 229
# 230
        else:
            raise ValueError("Space group not implemented")

        return basis_atom_position

basis_atom_position = preferred_basis(space_group_number, basis_atom_position, basis_wyckoff)